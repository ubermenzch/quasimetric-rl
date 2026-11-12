import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.qrl_repro_plateau import plateau_from_eval_log, validation_at_or_after
from tools import run_qrl_queue as queue


def write_evals(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


class PlateauRuleTests(unittest.TestCase):
    def test_fixed_limit_uses_completed_validation_even_if_success_improves(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "eval.log"
            write_evals(path, [
                {"env_steps": 198000, "succ_rate": .5},
                {"env_steps": 200000, "succ_rate": .9, "split": "test"},
            ])
            with path.open("a") as handle:
                handle.write('{"env_steps": 200000')
            self.assertIsNone(validation_at_or_after(path, target_steps=200000))
            with path.open("a") as handle:
                handle.write('\n' + json.dumps({"env_steps": 200000, "succ_rate": .8}) + '\n')
                handle.write(json.dumps({"env_steps": 202000, "succ_rate": .85}) + '\n')
            evidence = validation_at_or_after(path, target_steps=200000)
            self.assertEqual(evidence["trigger_validation_env_steps"], 200000)
            self.assertEqual(evidence["trigger_validation_success"], .8)
            self.assertEqual(evidence["stop_env_steps"], 202000)

    def test_patience_resets_only_for_a_new_high(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "eval.log"
            rows = [
                {"env_steps": 0, "succ_rate": .2},
                {"env_steps": 50000, "succ_rate": .4},
                {"env_steps": 100000, "succ_rate": .4},
                {"env_steps": 140000, "succ_rate": .3},
            ]
            write_evals(path, rows)
            self.assertIsNone(plateau_from_eval_log(path, min_steps=100000, patience_steps=100000))
            rows.append({"env_steps": 150000, "succ_rate": .4})
            write_evals(path, rows)
            evidence = plateau_from_eval_log(path, min_steps=100000, patience_steps=100000)
            self.assertEqual(evidence["best_validation_env_steps"], 50000)
            self.assertEqual(evidence["stop_env_steps"], 150000)
            rows.append({"env_steps": 152000, "succ_rate": .42})
            write_evals(path, rows)
            self.assertIsNone(plateau_from_eval_log(path, min_steps=100000, patience_steps=100000))

    def test_minimum_budget_and_validation_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "eval.log"
            write_evals(path, [
                {"env_steps": 0, "succ_rate": 0},
                {"env_steps": 100000, "succ_rate": 0},
                {"env_steps": 200000, "succ_rate": 1, "split": "test"},
            ])
            with path.open("a") as handle:
                handle.write('{"env_steps": 200000')
            self.assertIsNone(plateau_from_eval_log(path, min_steps=200000, patience_steps=100000))
            with path.open("a") as handle:
                handle.write('\n' + json.dumps({"env_steps": 200000, "succ_rate": 0}) + '\n')
            evidence = plateau_from_eval_log(path, min_steps=200000, patience_steps=100000)
            self.assertEqual(evidence["stop_env_steps"], 200000)
            self.assertEqual(evidence["best_validation_success"], 0)


class QueueStopTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.task = queue.Task("test_qrl_repro", "online", "FetchPush", "1000", "1000000")
        self.status_dir = self.root / "status"
        self.output = self.root / self.task.task_id
        self.output.mkdir()
        write_evals(self.output / "eval.log", [
            {"env_steps": 0, "succ_rate": .4},
            {"env_steps": 200000, "succ_rate": .4},
        ])
        rule = self.root / "runs/qrl_queue/qrl_reproduction_plateau.json"
        rule.parent.mkdir(parents=True)
        rule.write_text(json.dumps({"enabled": True, "patience_env_steps": 100000,
                                    "tasks": {self.task.task_id: 200000}}))
        queue.write_status(self.status_dir, self.task, "RUNNING", {
            "pid": "1234", "gpu": "1", "output_dir": str(self.output), "log_file": "train.log",
        })
        self.proc = mock.Mock(pid=1234)
        self.proc.poll.return_value = None
        self.job = queue.ActiveJob(self.task, "1", self.proc, Path("train.log"), self.output)
        self.active = {self.task.task_id: self.job}

    def tearDown(self):
        self.temp.cleanup()

    def run_stop(self, termination_result):
        fake_cmdline = f"python\0output_folder={self.task.task_id}\0".encode()
        with (mock.patch.object(queue, "ROOT", self.root),
              mock.patch.object(Path, "read_bytes", return_value=fake_cmdline),
              mock.patch.object(queue, "is_queue_user_process", return_value=True),
              mock.patch.object(queue, "output_finished_for_task", return_value=False),
              mock.patch.object(queue, "terminate_process", return_value=termination_result) as terminate):
            queue.stop_qrl_reproduction_plateaus({}, [self.task], self.status_dir, self.active, False)
        return terminate

    def test_stop_preserves_evidence_and_is_terminal(self):
        terminate = self.run_stop("terminated:130")
        terminate.assert_called_once()
        status = queue.read_status(self.status_dir, self.task.task_id)
        self.assertEqual(status["state"], "EARLY_STOPPED")
        self.assertEqual(status["stop_env_steps"], "200000")
        self.assertEqual(status["independent_final_test"], "not_performed")
        self.assertTrue(queue.task_terminal(status, retry_failed=True))
        self.assertNotIn(self.task.task_id, self.active)
        self.assertTrue((self.output / "eval.log").exists())

    def test_hard_limit_overrides_plateau_patience(self):
        rule_path = self.root / "runs/qrl_queue/qrl_reproduction_plateau.json"
        rule = json.loads(rule_path.read_text())
        rule["hard_stop_env_steps"] = {self.task.task_id: 200000}
        rule_path.write_text(json.dumps(rule))
        write_evals(self.output / "eval.log", [
            {"env_steps": 198000, "succ_rate": .4},
            {"env_steps": 200000, "succ_rate": .9},
        ])
        self.run_stop("terminated:130").assert_called_once()
        status = queue.read_status(self.status_dir, self.task.task_id)
        self.assertEqual(status["state"], "EARLY_STOPPED")
        self.assertEqual(status["error"], "requested_step_limit_early_stop")
        self.assertEqual(status["target_env_steps"], "200000")
        self.assertEqual(status["trigger_validation_env_steps"], "200000")

    def test_wait_checks_hard_limits_without_waiting_for_next_allocation(self):
        with (mock.patch.object(queue, "stop_qrl_reproduction_plateaus") as stop,
              mock.patch.object(queue, "gpu_compute_apps", return_value=[]),
              mock.patch.object(queue, "update_running_gpu_memory_peaks")):
            queue.sample_gpu_memory_during_wait({}, [], self.status_dir, .03, True, active={})
        self.assertGreaterEqual(stop.call_count, 1)
        self.assertTrue(stop.call_args.kwargs["hard_limits_only"])

    def test_orphan_zombie_is_recognized_without_waiting_for_sigkill(self):
        with (mock.patch.object(queue, "pid_alive", return_value=True),
              mock.patch.object(queue, "is_queue_user_process", return_value=True),
              mock.patch.object(queue, "plateau_pid_is_running", return_value=False),
              mock.patch.object(queue.os, "kill") as kill):
            result = queue.terminate_pid("1234", {}, 10)
        self.assertEqual(result, "terminated")
        kill.assert_called_once_with(1234, queue.signal.SIGTERM)

    def test_failed_signal_does_not_change_status(self):
        self.run_stop("terminate_error:denied")
        self.assertEqual(queue.read_status(self.status_dir, self.task.task_id)["state"], "RUNNING")
        self.assertIn(self.task.task_id, self.active)

    def test_pid_must_identify_the_same_task(self):
        with (mock.patch.object(queue, "ROOT", self.root),
              mock.patch.object(Path, "read_bytes", return_value=b"python\0output_folder=other\0"),
              mock.patch.object(queue, "output_finished_for_task", return_value=False),
              mock.patch.object(queue, "terminate_process") as terminate):
            queue.stop_qrl_reproduction_plateaus({}, [self.task], self.status_dir, self.active, False)
        terminate.assert_not_called()
        self.assertEqual(queue.read_status(self.status_dir, self.task.task_id)["state"], "RUNNING")

    def test_exited_orphan_is_not_requeued_if_its_pid_is_a_zombie(self):
        self.active.clear()
        fake_cmdline = f"python\0output_folder={self.task.task_id}\0".encode()
        with (mock.patch.object(queue, "ROOT", self.root),
              mock.patch.object(Path, "read_bytes", return_value=fake_cmdline),
              mock.patch.object(queue, "is_queue_user_process", return_value=True),
              mock.patch.object(queue, "output_finished_for_task", return_value=False),
              mock.patch.object(queue, "terminate_pid", return_value="kill_timeout"),
              mock.patch.object(queue, "plateau_pid_is_running", return_value=False)):
            queue.stop_qrl_reproduction_plateaus({}, [self.task], self.status_dir, self.active, False)
        status = queue.read_status(self.status_dir, self.task.task_id)
        self.assertEqual(status["state"], "EARLY_STOPPED")
        self.assertEqual(status["terminate_result"], "terminated:zombie")


if __name__ == "__main__":
    unittest.main()
