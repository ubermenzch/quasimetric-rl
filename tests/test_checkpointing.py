import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from quasimetric_rl.utils.checkpointing import (
    agent_checkpoint_filename,
    agent_checkpoint_step,
    full_checkpoint_key,
    prune_checkpoints,
    validate_training_cursor,
)


class AgentCheckpointNameTest(unittest.TestCase):
    def test_round_trip(self):
        filename = agent_checkpoint_filename(10_000)
        self.assertEqual(filename, "agent_checkpoint_step00010000.pth")
        self.assertEqual(agent_checkpoint_step(filename), 10_000)

    def test_rejects_invalid_values_and_names(self):
        with self.assertRaises(ValueError):
            agent_checkpoint_filename(-1)
        self.assertIsNone(agent_checkpoint_step("checkpoint_resume_latest.pth"))


class FullCheckpointNameTest(unittest.TestCase):
    def test_parses_regular_and_final_checkpoints(self):
        self.assertEqual(
            full_checkpoint_key("checkpoint_00184_00104.pth"),
            (184, 104, 0),
        )
        self.assertEqual(
            full_checkpoint_key("checkpoint_00184_00104_final.pth"),
            (184, 104, 1),
        )

    def test_rejects_non_archival_checkpoints(self):
        self.assertIsNone(full_checkpoint_key("checkpoint_resume_latest.pth"))
        self.assertIsNone(full_checkpoint_key("checkpoint_broken.pth"))
        self.assertIsNone(full_checkpoint_key("agent_checkpoint_step00010000.pth"))


class CheckpointPruningTest(unittest.TestCase):
    def test_preserves_final_checkpoint(self):
        with TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            old = output_dir / "checkpoint_00001_00001.pth"
            keep = output_dir / "checkpoint_00002_00002.pth"
            final = output_dir / "checkpoint_00001_00002_final.pth"
            unrelated = output_dir / "metrics.json"
            for path in (old, keep, final, unrelated):
                path.touch()

            prune_checkpoints(output_dir, keep)

            self.assertFalse(old.exists())
            self.assertTrue(keep.exists())
            self.assertTrue(final.exists())
            self.assertTrue(unrelated.exists())

    def test_supports_agent_checkpoint_pattern(self):
        with TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            old = output_dir / "agent_checkpoint_step00010000.pth"
            keep = output_dir / "agent_checkpoint_step00020000.pth"
            final = output_dir / "checkpoint_00002_00000_final.pth"
            for path in (old, keep, final):
                path.touch()

            prune_checkpoints(
                output_dir,
                keep,
                pattern="agent_checkpoint_step*.pth",
                preserve_final=False,
            )

            self.assertFalse(old.exists())
            self.assertTrue(keep.exists())
            self.assertTrue(final.exists())


class CheckpointCursorTest(unittest.TestCase):
    def test_accepts_complete_step_boundary(self):
        validate_training_cursor(
            {
                "data": {"epoch": 24, "next_batch_idx": 144},
                "loop_state": {"optim_steps": 6000},
            },
            num_batches=244,
        )

    def test_rejects_mid_step_checkpoint(self):
        with self.assertRaisesRegex(RuntimeError, "complete training-step boundary"):
            validate_training_cursor(
                {
                    "data": {"epoch": 18, "next_batch_idx": 138},
                    "loop_state": {"optim_steps": 4531},
                },
                num_batches=244,
            )

    def test_allows_legacy_checkpoint_without_cursor_state(self):
        validate_training_cursor({"epoch": 2, "it": 3}, num_batches=244)


if __name__ == "__main__":
    unittest.main()
