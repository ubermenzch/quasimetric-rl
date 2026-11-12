import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import analyze_nqrl_execution_metrics as metrics


class ValidFirstArrivalTest(unittest.TestCase):
    def test_recovers_successful_mean_from_censored_time(self):
        test = {'num_episodes': 100, 'success_count': 35, 'hitting_time': 559.0}
        self.assertEqual(metrics.valid_arrival(test, 700),
                         (35, 10335 / 35, True, 'valid'))

    def test_step_one_episodes_remain_in_an_eligible_seed(self):
        # Successful arrivals are [1, 1, 3]; the failed episode contributes 5.
        test = {'num_episodes': 4, 'success_count': 3, 'hitting_time': 2.5,
                'first_step_success_count': 2, 'initial_success_count': 1}
        self.assertEqual(metrics.valid_arrival(test, 4),
                         (3, 5 / 3, True, 'valid'))

    def test_no_success_is_invalid_without_a_zero_mean(self):
        test = {'num_episodes': 2, 'success_count': 0, 'hitting_time': 5.0}
        self.assertEqual(metrics.valid_arrival(test, 4),
                         (0, None, False, 'no_successful_episodes'))

    def test_mean_of_one_is_invalid(self):
        test = {'num_episodes': 2, 'success_count': 2, 'hitting_time': 1.0}
        self.assertEqual(metrics.valid_arrival(test, 4),
                         (2, 1, False, 'mean_not_above_one'))

    def test_threshold_uses_unrounded_mean(self):
        test = {'num_episodes': 1000, 'success_count': 1000, 'hitting_time': 1.001}
        count, mean, valid, status = metrics.valid_arrival(test, 2)
        self.assertEqual(round(mean, 1), 1.0)
        self.assertEqual((count, mean, valid, status), (1000, 1.001, True, 'valid'))

    def test_invalid_censored_summary_is_rejected(self):
        test = {'num_episodes': 2, 'success_count': 0, 'hitting_time': 4.0}
        with self.assertRaises(AssertionError):
            metrics.valid_arrival(test, 4)

    def test_all_five_seed_means_are_required_and_equally_weighted(self):
        env = metrics.ENVIRONMENTS[0]
        rows = [{'environment': env.paper_name, 'method': 'NQRL', 'seed': seed,
                 'successful_episodes': 10 ** (seed - 1000), 'valid_seed': True,
                 'valid_first_arrival_steps': float(seed - 998)}
                for seed in metrics.SEEDS]
        with patch.object(metrics, 'ENVIRONMENTS', (env,)), patch.object(metrics, 'METHODS', ('NQRL',)):
            summary = metrics.summarize_valid_arrivals(rows)[0]
            self.assertEqual(summary['status'], 'reported')
            self.assertEqual(summary['valid_first_arrival_steps_mean'], 4)
            self.assertAlmostEqual(summary['valid_first_arrival_steps_std'], 2.5 ** .5)
            rows[-1].update(valid_seed=False, valid_first_arrival_steps=None)
            summary = metrics.summarize_valid_arrivals(rows)[0]
            self.assertEqual(summary['status'], 'insufficient_valid_seeds')
            self.assertEqual(summary['valid_success_seeds'], 4)
            self.assertIsNone(summary['valid_first_arrival_steps_mean'])
            self.assertIsNone(summary['valid_first_arrival_steps_std'])
            with self.assertRaises(AssertionError):
                metrics.summarize_valid_arrivals(rows[:-1])


if __name__ == '__main__':
    unittest.main()
