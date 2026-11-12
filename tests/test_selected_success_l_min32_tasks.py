import tempfile
import unittest
from collections import Counter
from pathlib import Path

from tools.generate_selected_success_l_min32_tasks import (
    BASE_ALGORITHM_KEYS,
    CQRL_MIN32_ARGS,
    LEGACY_MAIN_MIN32_SPECS,
    MAIN_MIN32_SPECS,
    MIN32_KEY,
    PREVIOUS_MAIN_MIN32_SPECS,
    SELECTED_ALGORITHM_KEYS,
    SELECTED_PROTOCOLS,
    TRAINING_SEEDS,
    append_task_sets,
    completed_task_ids,
    extra_arg_map,
    generate_main_min32_tasks,
    generate_selected_tasks,
    replace_pending_main_min32_tasks,
)
from tools.run_qrl_queue import read_tasks


class SelectedSuccessLMin32TaskTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.selected = generate_selected_tasks()
        cls.main_min32 = generate_main_min32_tasks()

    def test_selected_matrix_is_complete_and_l_scale(self):
        expected = (
            len(SELECTED_PROTOCOLS)
            * len(SELECTED_ALGORITHM_KEYS)
            * len(TRAINING_SEEDS)
        )
        self.assertEqual(len(self.selected), expected)
        self.assertEqual(len({task.task_id for task in self.selected}), expected)
        self.assertEqual(
            Counter(task.env_name for task in self.selected),
            {
                protocol.environment_name:
                len(SELECTED_ALGORITHM_KEYS) * len(TRAINING_SEEDS)
                for protocol in SELECTED_PROTOCOLS
            },
        )
        for task in self.selected:
            self.assertRegex(task.params, r"^2[1-2]\.\d+m$")
            self.assertIn("-L", task.task_id)

    def test_selected_tasks_keep_final_pilot_budgets(self):
        protocols = {
            protocol.environment_name: protocol
            for protocol in SELECTED_PROTOCOLS
        }
        for task in self.selected:
            protocol = protocols[task.env_name]
            args = extra_arg_map(task)
            self.assertEqual(task.steps, str(protocol.total_steps))
            self.assertEqual(
                args["env.init_num_transitions"], str(protocol.initial_steps),
            )
            self.assertEqual(
                args["env.increment_num_transitions"], str(protocol.initial_steps),
            )
            self.assertEqual(args["save_steps"], str(protocol.save_steps))

    def test_selected_algorithm_cells_have_all_five_seeds(self):
        labels = {
            "td_infonce": "TD-InfoNCE-L",
            "gcsl": "GCSL-L",
            "c_learning": "C-Learning-L",
            "crl": "CRL-L",
            "qrl": "QRL-L",
            MIN32_KEY: "CQRL+Min32-L-InnerLR1",
        }
        for protocol in SELECTED_PROTOCOLS:
            for algorithm_key in (*BASE_ALGORITHM_KEYS, MIN32_KEY):
                group = [
                    task for task in self.selected
                    if task.env_name == protocol.environment_name
                    and f"_{labels[algorithm_key]}_" in task.task_id
                ]
                self.assertEqual(
                    {int(task.seed) for task in group}, set(TRAINING_SEEDS),
                )

    def test_every_min32_task_uses_min32_lr1(self):
        min32_tasks = [
            task for task in (*self.selected, *self.main_min32)
            if "CQRL+Min32" in task.task_id
        ]
        self.assertEqual(
            len(min32_tasks),
            (len(SELECTED_PROTOCOLS) + len(MAIN_MIN32_SPECS))
            * len(TRAINING_SEEDS),
        )
        for task in min32_tasks:
            args = task.extra_args.split()
            for required in CQRL_MIN32_ARGS:
                self.assertIn(required, args)

    def test_main_min32_matrix_has_requested_scales_and_budgets(self):
        self.assertEqual(
            len(self.main_min32), len(MAIN_MIN32_SPECS) * len(TRAINING_SEEDS),
        )
        by_key = {key: (size, steps) for key, size, steps in MAIN_MIN32_SPECS}
        expected_names = {
            "fetchpush": "FetchPush",
            "fetchslide": "FetchSlide",
            "fetchpickandplace": "FetchPickAndPlace",
            "reacher_hard": "reacher_hard",
            "maze2d_large": "maze2d-large",
            "pusher_v4": "Pusher-v4",
            "antnavigate_v4": "AntNavigate-v4",
        }
        for key, (size, steps) in by_key.items():
            group = [
                task for task in self.main_min32
                if task.env_name == expected_names[key]
            ]
            self.assertEqual(len(group), len(TRAINING_SEEDS))
            self.assertEqual({int(task.seed) for task in group}, set(TRAINING_SEEDS))
            self.assertEqual({task.steps for task in group}, {str(steps)})
            for task in group:
                self.assertEqual(extra_arg_map(task)["+cqrl_model_size"], size)
                self.assertEqual(size, "l")
        self.assertEqual(by_key["fetchpush"], ("l", 200_000))

    def test_pending_legacy_main_min32_tasks_can_be_replaced(self):
        old_tasks = {
            task.task_id: task
            for specs in (
                LEGACY_MAIN_MIN32_SPECS,
                PREVIOUS_MAIN_MIN32_SPECS,
            )
            for task in generate_main_min32_tasks(specs)
        }
        current_ids = {task.task_id for task in self.main_min32}
        obsolete = [
            task for task_id, task in old_tasks.items()
            if task_id not in current_ids
        ]
        with tempfile.TemporaryDirectory() as directory:
            queue_root = Path(directory)
            queue = queue_root / "tasks.tsv"
            queue.write_text(
                "# task_id\tmode\tenv_name\tseed\tsteps\tparams\textra_args\n"
                + "\n".join("\t".join((
                    task.task_id,
                    task.mode,
                    task.env_name,
                    task.seed,
                    task.steps,
                    task.params,
                    task.extra_args,
                )) for task in old_tasks.values())
                + "\n"
            )
            self.assertEqual(
                replace_pending_main_min32_tasks(queue), len(obsolete),
            )
            remaining_ids = {task.task_id for task in read_tasks(queue)}
            self.assertTrue(
                {task.task_id for task in obsolete}.isdisjoint(remaining_ids)
            )

    def test_append_is_idempotent_and_reuses_matching_tasks(self):
        with tempfile.TemporaryDirectory() as directory:
            queue = Path(directory) / "tasks.tsv"
            existing = self.selected[0]
            queue.write_text(
                "# task_id\tmode\tenv_name\tseed\tsteps\tparams\textra_args\n"
                + "\t".join((
                    existing.task_id,
                    existing.mode,
                    existing.env_name,
                    existing.seed,
                    existing.steps,
                    existing.params,
                    existing.extra_args,
                ))
                + "\n"
            )
            self.assertEqual(
                append_task_sets(queue, self.selected, self.main_min32),
                (len(self.selected) - 1, len(self.main_min32)),
            )
            self.assertEqual(
                append_task_sets(queue, self.selected, self.main_min32),
                (0, 0),
            )
            parsed = read_tasks(queue)
            self.assertEqual(
                len(parsed), len(self.selected) + len(self.main_min32),
            )

    def test_append_skips_completed_task_missing_from_queue(self):
        with tempfile.TemporaryDirectory() as directory:
            queue_root = Path(directory)
            queue = queue_root / "tasks.tsv"
            status_dir = queue_root / "status"
            status_dir.mkdir()
            queue.write_text(
                "# task_id\tmode\tenv_name\tseed\tsteps\tparams\textra_args\n"
            )
            completed = self.selected[0]
            (status_dir / f"{completed.task_id}.status").write_text(
                "state=DONE\n"
                f"task_id={completed.task_id}\n"
            )
            self.assertEqual(completed_task_ids(status_dir), {completed.task_id})
            self.assertEqual(
                append_task_sets(
                    queue, self.selected, self.main_min32,
                    status_dir=status_dir,
                ),
                (len(self.selected) - 1, len(self.main_min32)),
            )
            self.assertNotIn(
                completed.task_id,
                {task.task_id for task in read_tasks(queue)},
            )


if __name__ == "__main__":
    unittest.main()
