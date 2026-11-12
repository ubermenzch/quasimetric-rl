import contextlib
from dataclasses import replace
import inspect
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from omegaconf import OmegaConf

import online.main as training
from online.trainer import EvalEpisodeResult
from tools.generate_qrl_original_config_tasks import generate_tasks as original_tasks
from tools.run_qrl_queue import build_command, task_training_args, task_uses_native_qrl
from tools.run_native_qrl_logs_only import main as run_original_logs_only
from tools.set_pending_logs_only import logs_only_task


class LogsOnlyTrainingTest(unittest.TestCase):
    def test_best_model_is_cloned_and_selected_with_the_existing_rule(self):
        agent = torch.nn.Linear(1, 1, bias=False)
        store = training.InMemoryBestAgent()
        first = dict(success_count=2, hitting_time=1, epi_return=2, env_steps=1, optim_steps=1)
        with torch.no_grad():
            agent.weight.fill_(3)
        store.consider(agent, first)
        with torch.no_grad():
            agent.weight.fill_(9)
        store.consider(agent, dict(first, success_count=1, env_steps=2))
        store.load(agent)
        self.assertEqual(agent.weight.item(), 3)

    def test_full_logs_only_loop_writes_no_checkpoint_and_tests_best_model(self):
        class FakeTrainer:
            def __init__(self):
                self.total_env_steps = 2
                self.num_samples_per_cycle = 1
                self.validation_seed, self.test_seed = 1, 2
                self.num_eval_episodes = self.num_test_episodes = 2
                self.agent = torch.nn.Linear(1, 1, bias=False)
                self.losses = SimpleNamespace(actor_loss=None)
                self.replay = SimpleNamespace(
                    episode_length=2, num_transitions_realized=0,
                    env_spec=SimpleNamespace(observation_shape=(1,), action_shape=(1,)),
                )
                self.evaluated = []

            def iter_training_data(self, **kwargs):
                for step in (1, 2):
                    self.replay.num_transitions_realized = step
                    yield step, True, None, {}

            def train_step(self, data):
                with torch.no_grad():
                    self.agent.weight.fill_(self.replay.num_transitions_realized)
                return {}

            def evaluate(self, *, num_episodes, seed):
                value = self.agent.weight.item()
                self.evaluated.append((seed, value))
                success = value == 1
                result = torch.full((num_episodes, 2), success, dtype=torch.bool)
                return EvalEpisodeResult.from_timestep_reward_is_success(result.float(), result)

        trainer = FakeTrainer()
        writer = Mock()
        profiler = Mock()
        profiler.record.side_effect = lambda *args: contextlib.nullcontext()
        with tempfile.TemporaryDirectory() as tmp:
            cfg = SimpleNamespace(
                logs_only=True, resume_if_possible=False, eval_steps=None, save_steps=1,
                collect_rebuttal_diagnostics=True,
                log_steps=1, keep_only_best_and_final_checkpoints=True,
                agent=SimpleNamespace(training_schedule='joint'),
                device=SimpleNamespace(make=lambda: torch.device('cpu')),
                timing=SimpleNamespace(make=lambda **kwargs: profiler),
                env=SimpleNamespace(kind='gcrl', make=lambda: trainer.replay),
                setup_for_experiment=lambda: writer, output_dir=tmp,
                completion_file=str(Path(tmp) / 'COMPLETE'), batch_size=2,
                interaction=None, training_optimizations=None, seed=1000,
                dynamics_diagnostics=SimpleNamespace(enabled=False),
            )
            with patch.object(training.Conf, 'from_DictConfig', return_value=cfg), \
                    patch.object(training, 'resolve_split_encoder_goal_dims'), \
                    patch.object(training, 'Trainer', return_value=trainer), \
                    patch.object(training.utils, 'atomic_torch_save') as save:
                inspect.unwrap(training.train)(OmegaConf.create({'test_protocol': True}))
            save.assert_not_called()
            writer.flush.assert_called()
            writer.close.assert_called_once()
            self.assertFalse(list(Path(tmp).glob('*.pth')))
            self.assertTrue((Path(tmp) / 'COMPLETE').exists())
            selection = json.loads((Path(tmp) / 'best_checkpoint.json').read_text())
            self.assertEqual(selection['selected_model'], 'in_memory')
            self.assertEqual(selection['validation']['env_steps'], 1)
            self.assertEqual(trainer.evaluated, [(1, 1), (1, 2), (2, 1)])
            self.assertTrue((Path(tmp) / 'eval.log').exists())
            self.assertTrue((Path(tmp) / 'test.log').exists())
            summary = json.loads((Path(tmp) / 'diagnostics/run_summary.json').read_text())
            self.assertEqual(summary['state'], 'complete')
            self.assertEqual(summary['evaluation_count'], 3)
            self.assertEqual(len(list((Path(tmp) / 'diagnostics/evaluations').glob('*.npz'))), 3)

    def test_original_wrapper_skips_models_but_preserves_other_serialization(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / 'checkpoint_env00000001_final.pth'
            metrics = Path(tmp) / 'metrics.pth'
            original_save = torch.save
            def fake_source(*args, **kwargs):
                torch.save({'agent': torch.ones(2)}, model)
                torch.save({'result': torch.ones(2)}, metrics)
            with patch('tools.run_native_qrl_logs_only.runpy.run_module', side_effect=fake_source):
                run_original_logs_only()
            self.assertFalse(model.exists())
            self.assertTrue(metrics.exists())
            self.assertIs(torch.save, original_save)

    def test_source_repo_remains_pinned_and_queue_only_flag_is_not_a_hydra_argument(self):
        local, native = original_tasks()[0], original_tasks()[5]
        for task in (local, native):
            modified = logs_only_task(task)
            self.assertEqual(logs_only_task(modified), modified)
            self.assertFalse(any(arg.startswith('queue.') for arg in task_training_args(modified)))
            with tempfile.TemporaryDirectory() as tmp:
                command, _, cwd, _ = build_command({
                    'RESULTS_ROOT': tmp, 'RESUME_IF_POSSIBLE': '1',
                    'ONLINE_SAVE_REPLAY_BUFFER': 'True',
                }, modified, '1')
                if task_uses_native_qrl(task):
                    self.assertTrue(command[1].endswith('run_native_qrl_logs_only.py'))
                    self.assertEqual(cwd.name, 'qrl-original')
                    self.assertFalse(any(arg.startswith('logs_only=') for arg in command))
                else:
                    self.assertIn('logs_only=true', command)
                    self.assertIn('resume_if_possible=false', command)
                    self.assertIn('save_replay_buffer=false', command)


if __name__ == '__main__':
    unittest.main()
