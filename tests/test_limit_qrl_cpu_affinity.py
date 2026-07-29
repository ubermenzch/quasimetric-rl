import unittest

from tools.limit_qrl_cpu_affinity import format_cpu_list
from tools.limit_qrl_cpu_affinity import is_qrl_training_command
from tools.limit_qrl_cpu_affinity import normalize_pci_bus_id
from tools.limit_qrl_cpu_affinity import pack_affinity_groups
from tools.limit_qrl_cpu_affinity import parse_cpu_list


class LimitQrlCpuAffinityTest(unittest.TestCase):
    def test_cpu_list_round_trip(self):
        cpus = parse_cpu_list("0-3,8,10-11")
        self.assertEqual(cpus, [0, 1, 2, 3, 8, 10, 11])
        self.assertEqual(format_cpu_list(cpus), "0-3,8,10-11")

    def test_packs_whole_smt_cores_into_four_logical_cpu_groups(self):
        groups = pack_affinity_groups(
            [[0, 40], [1, 41], [2, 42], [3, 43]],
            max_logical_cpus=4,
        )
        self.assertEqual(groups, [[0, 1, 40, 41], [2, 3, 42, 43]])

    def test_recognizes_only_qrl_training_entry_points(self):
        self.assertTrue(is_qrl_training_command(
            ".venv/bin/python -m online.main env.kind=gcrl"
        ))
        self.assertTrue(is_qrl_training_command(
            "/usr/bin/python /repo/offline/main.py"
        ))
        self.assertFalse(is_qrl_training_command(
            ".venv/bin/python tools/run_qrl_queue.py"
        ))

    def test_normalizes_nvidia_pci_domain(self):
        self.assertEqual(
            normalize_pci_bus_id("00000000:1A:00.0"),
            "0000:1a:00.0",
        )


if __name__ == "__main__":
    unittest.main()
