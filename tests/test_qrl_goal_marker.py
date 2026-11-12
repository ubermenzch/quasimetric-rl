import copy
from types import SimpleNamespace
import unittest

import torch

from online.trainer import InteractionConf, Trainer
from tests.test_goal_set_distance import batch_data, small_agent_conf, vector_env_spec


def marker_trainer():
    conf = small_agent_conf(gsd_enabled=False)
    conf.num_critics = 1
    data = batch_data(batch_size=3)
    data.replay_episode_indices = torch.tensor([0, 0, 1])
    data.replay_timesteps = torch.tensor([0, 1, 2])
    success = torch.tensor([True, False, False, False, False, False, True, False])
    desired = torch.arange(40, dtype=torch.float32).reshape(10, 4)
    replay = SimpleNamespace(
        kind='gcrl', name='marker-test', episode_length=4,
        num_episodes_realized=0, transition_history_length=0,
        future_observation_discount=0.99, goal_set_dims=(0, 1),
        env_spec=vector_env_spec(),
        raw_data=SimpleNamespace(
            transition_infos={'is_success': success},
            observation_infos={'desired_goals': desired},
        ),
        sample=lambda *args, **kwargs: copy.deepcopy(data),
    )
    trainer = Trainer(
        agent_conf=conf, device=torch.device('cpu'), replay=replay, batch_size=3,
        interaction_conf=InteractionConf(total_env_steps=100, num_eval_episodes=1,
                                         num_test_episodes=1, validation_seed=1000,
                                         test_seed=2000, qrl_goal_marker=True),
    )
    return trainer, data, success, desired


def test_marker_encodes_task_goal_and_uses_only_successful_edges():
    trainer, raw, _, desired = marker_trainer()
    batch = trainer.sample()
    torch.testing.assert_close(batch.observations[:, :-1], raw.observations)
    assert torch.equal(batch.observations[:, -1], torch.zeros(3))
    torch.testing.assert_close(batch.goal_marker_sources[:, :-1], raw.next_observations[[0, 2]])
    assert torch.equal(batch.goal_marker_sources[:, -1], torch.zeros(2))
    torch.testing.assert_close(batch.goal_marker_targets[:, :2], desired[[1, 8], :2])
    assert torch.equal(batch.goal_marker_targets[:, 2:4], torch.zeros(2, 2))
    assert torch.equal(batch.goal_marker_targets[:, -1], torch.ones(2))

    inputs = []
    hook = trainer.agent.actor.register_forward_pre_hook(lambda _module, args: inputs.append(args))
    trainer.agent.act(raw.observations[:1], raw.future_observations[:1])
    hook.remove()
    obs, goal = inputs[0]
    torch.testing.assert_close(obs[:, :-1], raw.observations[:1])
    torch.testing.assert_close(goal[:, :2], raw.future_observations[:1, :2])
    assert obs[0, -1] == 0 and goal[0, -1] == 1
    assert torch.equal(goal[:, 2:4], torch.zeros(1, 2))

    with torch.no_grad():
        critic_infos = trainer.losses._make_critic_batch_infos(trainer.agent, batch)
        _, training_goals, actor_infos = trainer.losses.actor_loss.min_dist.gather_obs_goal_pairs(
            critic_infos, batch,
        )
        assert torch.equal(training_goals[..., -1], torch.ones(2, 3))
        assert torch.equal(training_goals[..., 2:4], torch.zeros(2, 3, 2))
        torch.testing.assert_close(
            actor_infos[0].zg, trainer.agent.critics[0].encoder(training_goals),
        )

    info = trainer.train_step(batch)
    local = info['critic_00']['local_constraint']
    assert local['goal_marker_transitions'] == 2
    assert torch.isfinite(local['goal_marker_dist'])
    assert info['critic_00']['global_push']['goal_marker_fraction'] == 1 / 3


def test_marker_training_allows_a_batch_with_no_successes():
    trainer, _, success, _ = marker_trainer()
    success[:] = False
    batch = trainer.sample()
    assert batch.goal_marker_sources.shape == (0, 5)
    info = trainer.train_step(batch)
    assert info['critic_00']['local_constraint']['goal_marker_transitions'] == 0


class GoalMarkerIntegrationTest(unittest.TestCase):
    def test_successful_marker_edges_and_policy_inputs(self):
        test_marker_encodes_task_goal_and_uses_only_successful_edges()

    def test_empty_success_batch(self):
        test_marker_training_allows_a_batch_with_no_successes()


if __name__ == '__main__':
    unittest.main()
