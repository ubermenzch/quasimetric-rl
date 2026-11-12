import io
from contextlib import redirect_stdout
import os
from pathlib import Path
import subprocess
import signal
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

from tools import qrl_queue_control as control
from tools.git_metadata import git_output


class GitMetadataTest(unittest.TestCase):
    def test_nonzero_git_metadata_requires_valid_content(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".git").mkdir()
            with mock.patch("tools.git_metadata.subprocess.run", return_value=
                            subprocess.CompletedProcess([], 255, str(root) + "\n", "")):
                self.assertEqual(git_output(["rev-parse", "--show-toplevel"], cwd=root).strip(), str(root))
        with mock.patch("tools.git_metadata.subprocess.run", return_value=
                        subprocess.CompletedProcess([], 255, "a" * 40 + "\n", "")):
            self.assertEqual(git_output(["rev-parse", "HEAD"]).strip(), "a" * 40)

    def test_clean_status_has_a_branch_header_even_with_exit_255(self):
        with mock.patch("tools.git_metadata.subprocess.run", return_value=
                        subprocess.CompletedProcess([], 255, "## HEAD (no branch)\n", "")) as run:
            self.assertEqual(git_output(["status", "--porcelain", "--untracked-files=no"]), "")
            self.assertIn("--branch", run.call_args.args[0])

    def test_dirty_status_is_preserved(self):
        with mock.patch("tools.git_metadata.subprocess.run", return_value=
                        subprocess.CompletedProcess([], 255, "## main\n M tracked.py\n?? new.py\n", "")):
            self.assertEqual(git_output(["status", "--short"]), " M tracked.py\n?? new.py\n")

    def test_missing_invalid_or_error_output_is_rejected(self):
        for output, error in (("", ""), ("fatal: bad repository\n", ""),
                              ("## main\nbad row\n", ""), ("## main\n", "fatal error")):
            with self.subTest(output=output, error=error), mock.patch(
                "tools.git_metadata.subprocess.run", return_value=
                subprocess.CompletedProcess([], 255, output, error)
            ), self.assertRaises(subprocess.CalledProcessError):
                git_output(["status", "--short"])


class QueueProcessControlTest(unittest.TestCase):
    def test_start_prepares_storage_and_detaches_the_scheduler(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            lock = root / "queue.lock"
            config = root / "queue.env"
            config.write_text(f"LOG_DIR={root / 'backup' / 'logs'}\n"
                              f"RESULTS_ROOT={root / 'backup' / 'results'}\nLOCK_FILE={lock}\n")
            proc = mock.Mock(pid=999991)
            proc.poll.return_value = None

            def launch(*args, **kwargs):
                self.assertTrue((root / "backup" / "results").is_dir())
                self.assertTrue((root / "backup" / "logs" / "runner.log").is_file())
                lock.write_text("pid=999991\n")
                return proc

            with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(
                control, "lock_is_held", side_effect=[False, True]
            ), mock.patch.object(control.subprocess, "Popen", side_effect=launch) as popen, redirect_stdout(io.StringIO()):
                self.assertEqual(control.start(config, []), 0)
            self.assertTrue(popen.call_args.kwargs["start_new_session"])
            self.assertEqual(popen.call_args.kwargs["stdin"], subprocess.DEVNULL)
            self.assertEqual(popen.call_args.kwargs["env"]["LOCK_FILE"], str(lock))
            proc.terminate.assert_not_called()

    def test_start_does_not_launch_a_second_scheduler(self):
        with TemporaryDirectory() as directory:
            config = Path(directory) / "queue.env"
            config.write_text("")
            with mock.patch.object(control, "lock_is_held", return_value=True), mock.patch.object(
                control.subprocess, "Popen"
            ) as popen, self.assertRaises(RuntimeError):
                control.start(config, [])
            popen.assert_not_called()

    def test_stop_signals_only_the_scheduler_by_default(self):
        with mock.patch.object(control, "lock_is_held", side_effect=[True, False, False]), mock.patch.object(
            control, "scheduler_pids", return_value=[999991]
        ), mock.patch.object(control.os, "kill") as kill, mock.patch.object(
            control.os, "killpg"
        ) as killpg, redirect_stdout(io.StringIO()):
            self.assertEqual(control.stop(Path("unused.lock")), 0)
        kill.assert_called_once_with(999991, signal.SIGTERM)
        killpg.assert_not_called()

    def test_stop_refuses_an_unidentified_lock_holder(self):
        with mock.patch.object(control, "lock_is_held", return_value=True), mock.patch.object(
            control, "scheduler_pids", return_value=[]
        ), mock.patch.object(control.os, "kill") as kill, self.assertRaises(RuntimeError):
            control.stop(Path("unused.lock"))
        kill.assert_not_called()
        self.assertFalse(control.scheduler_pid(os.getpid()))


if __name__ == "__main__":
    unittest.main()
