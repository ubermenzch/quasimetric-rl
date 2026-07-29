import unittest

from tools.restart_qrl_from_checkpoints import arg_value
from tools.restart_qrl_from_checkpoints import checkpoint_issue
from tools.restart_qrl_from_checkpoints import online_training_argv
from tools.restart_qrl_from_checkpoints import resume_argv


class RestartQrlFromCheckpointsTest(unittest.TestCase):
    def test_resume_override_is_unique_and_last(self):
        argv = [
            ".venv/bin/python", "-m", "online.main",
            "resume_if_possible=False", "seed=1000",
        ]
        resumed = resume_argv(argv)
        self.assertEqual(resumed.count("resume_if_possible=True"), 1)
        self.assertNotIn("resume_if_possible=False", resumed)
        self.assertEqual(resumed[-1], "resume_if_possible=True")

    def test_recognizes_online_module_and_last_override(self):
        argv = [
            ".venv/bin/python", "-m", "online.main",
            "output_folder=old", "output_folder=current",
        ]
        self.assertTrue(online_training_argv(argv))
        self.assertEqual(arg_value(argv, "output_folder"), "current")

    def test_valid_committed_checkpoint_metadata(self):
        state = {
            "env_steps": 20_000,
            "optim_steps": 19_000,
            "checkpoint_kind": "online_committed",
            "validation_summary": {
                "env_steps": 20_000,
                "optim_steps": 19_000,
            },
            "loop_state": {
                "cycle_env_steps": 20_000,
                "next_cycle_sample": 1_000,
            },
            "agent": {},
            "losses": {},
            "rng": {},
            "replay": {},
        }
        self.assertIsNone(checkpoint_issue(state, 20_000, 19_000))

    def test_rejects_checkpoint_without_replay(self):
        state = {
            "env_steps": 20_000,
            "optim_steps": 19_000,
            "checkpoint_kind": "online_committed",
            "validation_summary": {
                "env_steps": 20_000,
                "optim_steps": 19_000,
            },
            "loop_state": {
                "cycle_env_steps": 20_000,
                "next_cycle_sample": 1_000,
            },
            "agent": {},
            "losses": {},
            "rng": {},
        }
        self.assertEqual(checkpoint_issue(state, 20_000, 19_000), "missing replay")


if __name__ == "__main__":
    unittest.main()
