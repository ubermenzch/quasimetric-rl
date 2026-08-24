import json
import unittest
import urllib.parse
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from tools.run_qrl_queue import Task
from tools.run_qrl_queue import notify_queue_events
from tools.run_qrl_queue import send_ntfy_notification
from tools.run_qrl_queue import send_phone_notification
from tools.run_qrl_queue import send_serverchan_notification
from tools.run_qrl_queue import write_status


class QueueNotificationsTest(unittest.TestCase):
    def setUp(self):
        self.task = Task(
            task_id="notification_task",
            mode="online",
            env_name="FetchPush",
            seed="1000",
            steps="1000",
        )

    def test_first_enable_baselines_existing_terminal_tasks(self):
        with TemporaryDirectory() as directory:
            status_dir = Path(directory)
            write_status(status_dir, self.task, "DONE", {"output_dir": "/result"})
            config = {"NTFY_TOPIC_URL": "https://ntfy.sh/private-topic"}

            with mock.patch(
                "tools.run_qrl_queue.send_phone_notification"
            ) as send:
                notify_queue_events(
                    config, [self.task], status_dir, queue_terminal=True
                )

            send.assert_not_called()
            payload = json.loads((status_dir / "notifications.json").read_text())
            self.assertEqual(payload["version"], 1)
            self.assertEqual(len(payload["events"]), 2)

    def test_new_completion_sends_task_and_queue_once(self):
        with TemporaryDirectory() as directory:
            status_dir = Path(directory)
            config = {
                "NTFY_TOPIC_URL": "https://ntfy.sh/private-topic",
                "NTFY_NOTIFY_NO_PENDING": "0",
                "NTFY_NOTIFY_TASK_DONE": "1",
                "NTFY_NOTIFY_QUEUE_DONE": "1",
            }
            write_status(status_dir, self.task, "PENDING")
            notify_queue_events(
                config, [self.task], status_dir, queue_terminal=False
            )
            write_status(status_dir, self.task, "DONE", {"output_dir": "/result"})

            with mock.patch(
                "tools.run_qrl_queue.send_phone_notification", return_value=True
            ) as send:
                notify_queue_events(
                    config, [self.task], status_dir, queue_terminal=True
                )
                notify_queue_events(
                    config, [self.task], status_dir, queue_terminal=True
                )

            self.assertEqual(send.call_count, 2)
            self.assertEqual(send.call_args_list[0].args[1], "QRL task completed")
            self.assertEqual(send.call_args_list[1].args[1], "QRL queue completed")

    def test_failed_delivery_is_retried(self):
        with TemporaryDirectory() as directory:
            status_dir = Path(directory)
            config = {
                "NTFY_TOPIC_URL": "https://ntfy.sh/private-topic",
                "NTFY_NOTIFY_NO_PENDING": "0",
                "NTFY_NOTIFY_TASK_FAILED": "1",
                "NTFY_NOTIFY_QUEUE_DONE": "0",
            }
            write_status(status_dir, self.task, "PENDING")
            notify_queue_events(
                config, [self.task], status_dir, queue_terminal=False
            )
            write_status(
                status_dir,
                self.task,
                "FAILED",
                {"error": "nonzero_exit", "exit_code": "1"},
            )

            with mock.patch(
                "tools.run_qrl_queue.send_phone_notification", return_value=False
            ) as send:
                notify_queue_events(
                    config, [self.task], status_dir, queue_terminal=True
                )
                notify_queue_events(
                    config, [self.task], status_dir, queue_terminal=True
                )

            self.assertEqual(send.call_count, 2)

    def test_running_tasks_do_not_trigger_default_completion_notification(self):
        with TemporaryDirectory() as directory:
            status_dir = Path(directory)
            config = {"NTFY_TOPIC_URL": "https://ntfy.sh/private-topic"}
            write_status(status_dir, self.task, "PENDING")

            with mock.patch(
                "tools.run_qrl_queue.send_phone_notification", return_value=True
            ) as send:
                notify_queue_events(
                    config, [self.task], status_dir, queue_terminal=False
                )
                write_status(status_dir, self.task, "RUNNING")
                notify_queue_events(
                    config, [self.task], status_dir, queue_terminal=False
                )
                notify_queue_events(
                    config, [self.task], status_dir, queue_terminal=False
                )
                write_status(status_dir, self.task, "DONE")
                notify_queue_events(
                    config, [self.task], status_dir, queue_terminal=True
                )

            send.assert_called_once()
            self.assertEqual(send.call_args.args[1], "QRL queue completed")

    def test_optional_pending_drain_notification_remains_available(self):
        with TemporaryDirectory() as directory:
            status_dir = Path(directory)
            config = {
                "NTFY_TOPIC_URL": "https://ntfy.sh/private-topic",
                "NOTIFY_HOST_LABEL": "L40",
                "NOTIFY_NO_PENDING": "1",
            }
            write_status(status_dir, self.task, "RUNNING")

            with mock.patch(
                "tools.run_qrl_queue.send_phone_notification", return_value=True
            ) as send:
                notify_queue_events(
                    config,
                    [self.task],
                    status_dir,
                    queue_terminal=False,
                    pending_was_present=True,
                )

            send.assert_called_once()
            self.assertIn("PENDING: 0", send.call_args.args[2])
            self.assertIn("Host: L40", send.call_args.args[2])

    def test_task_errors_are_batched_with_diagnostic_details(self):
        second_task = Task(
            task_id="second_notification_task",
            mode="offline",
            env_name="maze2d-umaze-v1",
            seed="1001",
            steps="1000",
        )
        with TemporaryDirectory() as directory:
            status_dir = Path(directory) / "status"
            log_file = Path(directory) / "failed.log"
            log_file.write_text("setup\nTraceback line\nCUDA failure detail\n")
            config = {
                "NTFY_TOPIC_URL": "https://ntfy.sh/private-topic",
                "NOTIFY_NO_PENDING": "0",
                "NOTIFY_TASK_FAILED": "1",
            }
            for task in (self.task, second_task):
                write_status(status_dir, task, "PENDING")
            notify_queue_events(
                config,
                [self.task, second_task],
                status_dir,
                queue_terminal=False,
            )
            write_status(
                status_dir,
                self.task,
                "FAILED",
                {
                    "error": "nonzero_exit",
                    "exit_code": "1",
                    "log_file": str(log_file),
                },
            )
            write_status(
                status_dir,
                second_task,
                "PAUSED",
                {"error": "task_definition_mismatch"},
            )

            with mock.patch(
                "tools.run_qrl_queue.send_phone_notification", return_value=True
            ) as send:
                notify_queue_events(
                    config,
                    [self.task, second_task],
                    status_dir,
                    queue_terminal=True,
                )

            self.assertEqual(send.call_count, 2)
            title = send.call_args_list[0].args[1]
            message = send.call_args_list[0].args[2]
            self.assertEqual(title, "QRL task errors/issues (2)")
            self.assertIn("notification_task", message)
            self.assertIn("second_notification_task", message)
            self.assertIn("Exit code: 1", message)
            self.assertIn("CUDA failure detail", message)
            self.assertEqual(
                send.call_args_list[1].args[1],
                "QRL queue finished with issues",
            )

    def test_pending_drain_delivery_failure_remains_armed(self):
        with TemporaryDirectory() as directory:
            status_dir = Path(directory)
            config = {
                "NTFY_TOPIC_URL": "https://ntfy.sh/private-topic",
                "NOTIFY_NO_PENDING": "1",
            }
            write_status(status_dir, self.task, "PENDING")
            notify_queue_events(
                config, [self.task], status_dir, queue_terminal=False
            )
            write_status(status_dir, self.task, "RUNNING")

            with mock.patch(
                "tools.run_qrl_queue.send_phone_notification",
                side_effect=[False, True],
            ) as send:
                notify_queue_events(
                    config, [self.task], status_dir, queue_terminal=False
                )
                notify_queue_events(
                    config, [self.task], status_dir, queue_terminal=False
                )
                notify_queue_events(
                    config, [self.task], status_dir, queue_terminal=False
                )

            self.assertEqual(send.call_count, 2)
            payload = json.loads((status_dir / "notifications.json").read_text())
            self.assertFalse(payload["pending_present"])

    def test_http_request_uses_utf8_body_and_bearer_token(self):
        response = mock.MagicMock()
        response.__enter__.return_value = response
        with mock.patch(
            "tools.run_qrl_queue.urllib.request.urlopen", return_value=response
        ) as urlopen:
            sent = send_ntfy_notification(
                {
                    "NTFY_TOPIC_URL": "https://ntfy.sh/private-topic",
                    "NTFY_TOKEN": "secret-token",
                },
                "QRL complete",
                "training complete",
            )

        self.assertTrue(sent)
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "https://ntfy.sh/private-topic")
        self.assertEqual(request.data, b"training complete")
        self.assertEqual(request.get_header("Authorization"), "Bearer secret-token")
        self.assertEqual(urlopen.call_args.kwargs["timeout"], 10.0)

    def test_serverchan_request_uses_form_body_and_checks_success_code(self):
        response = mock.MagicMock()
        response.__enter__.return_value = response
        response.read.return_value = b'{"code": 0, "message": "success"}'
        with mock.patch(
            "tools.run_qrl_queue.urllib.request.urlopen", return_value=response
        ) as urlopen:
            sent = send_serverchan_notification(
                {"SERVERCHAN_SENDKEY": "SCT-test-key"},
                "QRL complete",
                "Pending: 0",
            )

        self.assertTrue(sent)
        request = urlopen.call_args.args[0]
        self.assertEqual(
            request.full_url,
            "https://sctapi.ftqq.com/SCT-test-key.send",
        )
        form = urllib.parse.parse_qs(request.data.decode())
        self.assertEqual(form["title"], ["QRL complete"])
        self.assertEqual(form["desp"], ["Pending: 0"])
        self.assertEqual(urlopen.call_args.kwargs["timeout"], 10.0)

    def test_phone_notification_adds_host_label_to_serverchan_title(self):
        config = {
            "SERVERCHAN_SENDKEY": "SCT-test-key",
            "NOTIFY_HOST_LABEL": "L40",
        }
        with mock.patch(
            "tools.run_qrl_queue.send_serverchan_notification", return_value=True
        ) as send:
            sent = send_phone_notification(
                config,
                "QRL complete",
                "Pending: 0",
            )

        self.assertTrue(sent)
        send.assert_called_once_with(
            config,
            "[L40] QRL complete",
            "Pending: 0",
        )

    def test_phone_notification_uses_system_hostname_by_default(self):
        config = {"NTFY_TOPIC_URL": "https://ntfy.sh/private-topic"}
        with (
            mock.patch(
                "tools.run_qrl_queue.socket.gethostname", return_value="worker-07"
            ),
            mock.patch(
                "tools.run_qrl_queue.send_ntfy_notification", return_value=True
            ) as send,
        ):
            sent = send_phone_notification(
                config,
                "QRL complete",
                "Pending: 0",
                priority="high",
                tags="warning",
            )

        self.assertTrue(sent)
        send.assert_called_once_with(
            config,
            "[worker-07] QRL complete",
            "Pending: 0",
            priority="high",
            tags="warning",
        )


if __name__ == "__main__":
    unittest.main()
