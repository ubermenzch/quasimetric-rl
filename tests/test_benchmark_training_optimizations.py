import json
import tempfile
import unittest
from pathlib import Path

from tools.benchmark_training_optimizations import (
    OPTIMIZATION_FEATURES,
    RunResult,
    background_command,
    benchmark_plans,
    full_ablation_rounds,
    full_ablation_variants,
    is_baseline_variant,
    parse_timing,
    render_report,
    variant_features,
    variant_optimization_args,
)


class TrainingOptimizationBenchmarkTest(unittest.TestCase):
    def test_background_command_detaches_flag_and_pins_output_directory(self):
        command = background_command(
            Path('/tmp/benchmark-result'),
            ['--gpus', '6', '7', '--suite', 'full', '--background'],
        )
        self.assertNotIn('--background', command)
        self.assertEqual(command[-2:], [
            '--output-dir', '/tmp/benchmark-result',
        ])

    def test_background_command_preserves_explicit_output_directory(self):
        command = background_command(
            Path('/tmp/resolved-result'),
            ['--background', '--output-dir=/tmp/explicit-result'],
        )
        self.assertEqual(command.count('--output-dir'), 0)
        self.assertIn('--output-dir=/tmp/explicit-result', command)

    def test_full_ablation_covers_every_nonempty_combination(self):
        variants = full_ablation_variants()
        self.assertEqual(len(variants), 15)
        self.assertEqual(len(set(variants)), 15)
        self.assertEqual(
            {variant_features(variant) for variant in variants},
            {
                frozenset(
                    feature
                    for bit, feature in enumerate(OPTIMIZATION_FEATURES)
                    if mask & (1 << bit)
                )
                for mask in range(1, 16)
            },
        )

    def test_full_ablation_plans_have_per_gpu_baselines(self):
        plans = benchmark_plans((6, 7), 'full')
        self.assertEqual([variants[0] for _, variants in plans], [
            'baseline_pre', 'baseline_pre',
        ])
        self.assertEqual([variants[-1] for _, variants in plans], [
            'baseline_post', 'baseline_post',
        ])
        self.assertEqual([len(variants) for _, variants in plans], [10, 10])
        candidates = [
            variant
            for _, variants in plans
            for variant in variants
            if not is_baseline_variant(variant)
        ]
        self.assertEqual(len(candidates), 15)
        self.assertEqual(len(set(candidates)), 15)

    def test_full_ablation_rounds_are_complementary_and_gpu_balanced(self):
        rounds = full_ablation_rounds()
        self.assertEqual(len(rounds), 10)
        factorial_rounds = rounds[1:-1]
        self.assertEqual(
            {
                variant_features(variant)
                for variants in factorial_rounds
                for variant in variants
            },
            {
                frozenset(
                    feature
                    for bit, feature in enumerate(OPTIMIZATION_FEATURES)
                    if mask & (1 << bit)
                )
                for mask in range(16)
            },
        )
        all_features = frozenset(OPTIMIZATION_FEATURES)
        for left, right in factorial_rounds:
            self.assertEqual(
                variant_features(left) ^ variant_features(right),
                all_features,
            )
        for gpu_column in (0, 1):
            for feature in OPTIMIZATION_FEATURES:
                self.assertEqual(
                    sum(
                        feature in variant_features(variants[gpu_column])
                        for variants in factorial_rounds
                    ),
                    4,
                )

    def test_combination_arguments_enable_only_selected_features(self):
        args = {
            token.split('=', 1)[0]: token.split('=', 1)[1]
            for token in variant_optimization_args('tf32+fused_adamw+compile')
        }
        self.assertEqual(args['training_optimizations.tf32'], 'true')
        self.assertEqual(args['training_optimizations.amp_dtype'], 'null')
        self.assertEqual(args['training_optimizations.fused_adamw'], 'true')
        self.assertEqual(
            args['training_optimizations.compile_heavy_modules'], 'true',
        )

    def test_timing_excludes_cold_segment_and_reports_segment_noise(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            records = [
                (40.0, 200),
                (20.0, 200),
                (22.0, 200),
                (18.0, 200),
                (20.0, 200),
            ]
            with (path / 'timing.jsonl').open('w') as handle:
                for total_s, count in records:
                    print(json.dumps({
                        'records': {
                            'train/iteration': {
                                'total_s': total_s,
                                'count': count,
                            },
                        },
                    }), file=handle)
            train_s, first_s, steady_s, cv, batches, segments = parse_timing(path)
        self.assertEqual(train_s, 120.0)
        self.assertEqual(first_s, 40.0)
        self.assertAlmostEqual(steady_s, 0.1)
        self.assertGreater(cv, 0)
        self.assertEqual(batches, 800)
        self.assertEqual(segments, 5)

    def test_full_report_ranks_combinations_and_computes_main_effects(self):
        results = []
        for gpu in (6, 7):
            for variant, steady in (
                    ('baseline_pre', 0.090), ('baseline_post', 0.110)):
                results.append(RunResult(
                    variant, gpu, 0, 10.0, 8.0, 2.0,
                    steady, 1.0, 800, 5, '/tmp/result', '/tmp/log',
                ))
        for index, variant in enumerate(full_ablation_variants()):
            gpu = 6 + index % 2
            enabled = len(variant_features(variant))
            results.append(RunResult(
                variant, gpu, 0, 9.0, 7.0, 2.0,
                0.100 / (1 + enabled * 0.05), 1.0,
                800, 5, '/tmp/result', '/tmp/log',
            ))
        report = render_report(results, 'xxxl', 'full')
        self.assertIn('## Steady-state ranking', report)
        self.assertIn('## Average main effects', report)
        self.assertIn('tf32+bf16+fused_adamw+compile', report)
        self.assertIn('arithmetic mean', report)


if __name__ == '__main__':
    unittest.main()
