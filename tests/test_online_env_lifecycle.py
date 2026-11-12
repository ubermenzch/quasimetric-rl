import contextlib
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from online.trainer import Trainer
from quasimetric_rl.data.online.paper_validation.reference import make_ant_maze


class EnvironmentLifecycleTest(unittest.TestCase):
    def make_trainer(self, *, fail_at=None):
        environments = []
        replay = SimpleNamespace(
            num_transitions_realized=0, num_episodes_realized=0,
            num_successful_transitions=0, episodes_capacity=4, episode_length=2,
        )

        def make_env():
            env = Mock()
            environments.append(env)
            return env

        def collect(*, env):
            if replay.num_episodes_realized == fail_at:
                raise RuntimeError('collection failed')
            replay.num_episodes_realized += 1
            replay.num_transitions_realized += 2

        trainer = SimpleNamespace(
            replay=replay, num_samples_per_cycle=1, total_env_steps=6,
            num_prefill_episodes=1, algorithm='crl', num_rollouts_per_cycle=1,
            random_policy_env_steps=0, make_collect_env=make_env,
            collect_rollout=collect, collect_random_rollout=collect,
            _record=lambda _: contextlib.nullcontext(),
            sample=lambda: SimpleNamespace(rewards=0),
        )
        return trainer, environments

    def test_collect_environments_close_before_training_yields(self):
        trainer, environments = self.make_trainer()
        steps = []
        for step, _, _, _ in Trainer.iter_training_data(trainer):
            steps.append(step)
            for env in environments:
                env.close.assert_called_once_with()
        self.assertEqual(steps, [2, 4, 6])
        self.assertEqual(len(environments), 3)

    def test_collection_error_closes_prefill_and_regular_environments(self):
        for fail_at in (0, 1):
            with self.subTest(fail_at=fail_at):
                trainer, environments = self.make_trainer(fail_at=fail_at)
                with self.assertRaisesRegex(RuntimeError, 'collection failed'):
                    list(Trainer.iter_training_data(trainer))
                for env in environments:
                    env.close.assert_called_once_with()

    def test_d4rl_temporary_xml_cleanup_on_success_and_constructor_error(self):
        d4rl = ModuleType('d4rl')
        locomotion = ModuleType('d4rl.locomotion')
        maze_env = ModuleType('d4rl.locomotion.maze_env')
        d4rl.locomotion = locomotion
        locomotion.maze_env = maze_env
        maze_env.tempfile = tempfile
        modules = {
            'd4rl': d4rl, 'd4rl.locomotion': locomotion,
            'd4rl.locomotion.maze_env': maze_env,
        }
        original_mkstemp = tempfile.mkstemp
        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as tmp:
                paths = []

                def constructor(**kwargs):
                    descriptor, name = maze_env.tempfile.mkstemp(dir=tmp, suffix='.xml')
                    path = Path(name)
                    paths.append(path)
                    with self.assertRaises(OSError):
                        os.fstat(descriptor)
                    path.write_text('<mujoco/>')
                    self.assertEqual(path.read_text(), '<mujoco/>')
                    self.assertIs(tempfile.mkstemp, original_mkstemp)
                    if fail:
                        raise RuntimeError('model load failed')
                    return 'loaded simulator'

                with patch.dict(sys.modules, modules):
                    ant_module = SimpleNamespace(AntMaze=constructor)
                    if fail:
                        with self.assertRaisesRegex(RuntimeError, 'model load failed'):
                            make_ant_maze(ant_module, map_name='umaze')
                    else:
                        self.assertEqual(make_ant_maze(ant_module, map_name='umaze'), 'loaded simulator')
                self.assertIs(maze_env.tempfile, tempfile)
                self.assertTrue(paths)
                self.assertTrue(all(not path.exists() for path in paths))
