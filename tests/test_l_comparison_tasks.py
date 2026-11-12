import tempfile
import unittest
from collections import Counter
from pathlib import Path

from tools.generate_l_comparison_tasks import (
    ALGORITHM_KEYS,
    COMPARISON_SPECS,
    TRAINING_SEEDS,
    append_tasks,
    comparison_environments,
    generate_tasks,
)
from tools.generate_selected_success_l_min32_tasks import extra_arg_map
from tools.run_qrl_queue import read_tasks


class LComparisonTaskTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tasks = generate_tasks()

    def test_matrix_is_complete(self):
        expected = len(COMPARISON_SPECS) * len(ALGORITHM_KEYS) * len(TRAINING_SEEDS)
        self.assertEqual(len(self.tasks), expected)
        self.assertEqual(len({task.task_id for task in self.tasks}), expected)
        self.assertEqual(
            Counter(task.env_name for task in self.tasks),
            {
                environment.name: len(ALGORITHM_KEYS) * len(TRAINING_SEEDS)
                for environment in comparison_environments()
            },
        )

    def test_budgets_scales_and_checkpoint_intervals_match(self):
        environments = {
            environment.name: environment
            for environment in comparison_environments()
        }
        expected_save_steps = {
            "FetchPush": 20_000,
            "FetchSlide": 50_000,
            "FetchPickAndPlace": 20_000,
            "reacher_hard": 20_000,
            "maze2d-large": 20_000,
        }
        for task in self.tasks:
            environment = environments[task.env_name]
            args = extra_arg_map(task)
            self.assertEqual(task.steps, str(environment.total_steps))
            self.assertEqual(
                args["save_steps"], str(expected_save_steps[task.env_name]),
            )
            self.assertEqual(
                [value for key, value in args.items()
                 if key.endswith("_model_size")],
                ["l"],
            )
            self.assertRegex(task.params, r"^2[1-2]\.\d+m$")

    def test_each_cell_has_five_training_seeds(self):
        for environment in comparison_environments():
            for algorithm_key in ALGORITHM_KEYS:
                marker = {
                    "qrl": "_QRL-L_",
                    "td_infonce": "_TD-InfoNCE-L_",
                    "gcsl": "_GCSL-L_",
                    "c_learning": "_C-Learning-L_",
                    "crl": "_CRL-L_",
                }[algorithm_key]
                group = [
                    task for task in self.tasks
                    if task.env_name == environment.name
                    and marker in task.task_id
                ]
                self.assertEqual(
                    {int(task.seed) for task in group}, set(TRAINING_SEEDS),
                )

    def test_append_skips_done_tasks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queue = root / "tasks.tsv"
            status_dir = root / "status"
            status_dir.mkdir()
            queue.write_text(
                "# task_id\tmode\tenv_name\tseed\tsteps\tparams\textra_args\n"
            )
            done = self.tasks[0]
            (status_dir / f"{done.task_id}.status").write_text(
                f"state=DONE\ntask_id={done.task_id}\n"
            )
            self.assertEqual(
                append_tasks(queue, self.tasks, status_dir), len(self.tasks) - 1,
            )
            self.assertNotIn(
                done.task_id, {task.task_id for task in read_tasks(queue)},
            )


if __name__ == "__main__":
    unittest.main()
