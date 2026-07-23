import unittest
import math

import gym
import numpy as np
import torch
import torch.nn.functional as F

from quasimetric_rl.data.env_spec.act_distn import BoxOutputLinearNormalization
from quasimetric_rl.data.env_spec.act_distn.utils import (
    AcmeTanhTransformedDistribution,
    SampleDist,
)


class ImmutableActionBoundsTest(unittest.TestCase):
    @staticmethod
    def make_converter():
        return BoxOutputLinearNormalization(
            gym.spaces.Box(
                low=np.array([-2.0, 1.0, -5.0], dtype=np.float32),
                high=np.array([4.0, 5.0, 3.0], dtype=np.float32),
                dtype=np.float32,
            )
        )

    def test_bounds_are_python_values_materialized_per_forward(self):
        converter = self.make_converter()
        self.assertEqual(converter._mean_values, (1.0, 3.0, -1.0))
        self.assertEqual(converter._half_len_values, (3.0, 2.0, 4.0))
        self.assertEqual(dict(converter.named_buffers()), {})
        self.assertFalse(hasattr(converter, 'mean'))
        self.assertFalse(hasattr(converter, 'half_len'))

        feature = torch.zeros(2, converter.input_size, dtype=torch.float64)
        mean_1, half_len_1 = converter._action_bounds_like(feature)
        mean_2, half_len_2 = converter._action_bounds_like(feature)
        self.assertEqual(mean_1.dtype, feature.dtype)
        self.assertEqual(half_len_1.dtype, feature.dtype)
        self.assertEqual(mean_1.tolist(), [1.0, 3.0, -1.0])
        self.assertEqual(half_len_1.tolist(), [3.0, 2.0, 4.0])
        self.assertNotEqual(mean_1.data_ptr(), mean_2.data_ptr())
        self.assertNotEqual(half_len_1.data_ptr(), half_len_2.data_ptr())

    def test_forward_matches_legacy_buffer_formula_bitwise(self):
        converter = self.make_converter()
        feature = torch.linspace(-1.0, 1.0, 18).reshape(3, 6)
        gmean, grawstd = feature.view(3, 2, 3).unbind(dim=-2)
        mean, half_len = converter._action_bounds_like(feature)
        legacy = torch.distributions.Normal(
            loc=gmean,
            scale=F.softplus(grawstd) + 1e-4,
        )
        legacy = AcmeTanhTransformedDistribution(legacy)
        legacy = torch.distributions.TransformedDistribution(
            legacy,
            torch.distributions.AffineTransform(
                loc=mean,
                scale=half_len,
            ),
        )
        legacy = SampleDist(torch.distributions.Independent(
            legacy, reinterpreted_batch_ndims=1
        ))

        torch.manual_seed(53)
        expected = legacy.rsample()
        torch.manual_seed(53)
        actual = converter(feature).rsample()
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(
            converter(feature).log_prob(actual),
            legacy.log_prob(expected),
        ))

    def test_state_dict_keeps_legacy_keys_but_ignores_corrupted_values(self):
        converter = self.make_converter()
        feature = torch.zeros(4, converter.input_size)
        torch.manual_seed(23)
        expected_action = converter(feature).rsample()
        state = converter.state_dict()
        self.assertEqual(state['mean'].tolist(), [1.0, 3.0, -1.0])
        self.assertEqual(state['half_len'].tolist(), [3.0, 2.0, 4.0])
        state['mean'].zero_()
        state['half_len'].zero_()

        restored = self.make_converter()
        with self.assertWarnsRegex(
                RuntimeWarning, 'Ignoring checkpoint half_len'):
            restored.load_state_dict(state, strict=True)
        torch.manual_seed(23)
        actual_action = restored(feature).rsample()
        self.assertTrue(torch.equal(actual_action, expected_action))
        _, runtime_half_len = restored._action_bounds_like(
            torch.zeros(restored.input_size)
        )
        self.assertEqual(runtime_half_len.tolist(), [3.0, 2.0, 4.0])

    def test_multidimensional_action_shape_is_preserved(self):
        converter = BoxOutputLinearNormalization(
            gym.spaces.Box(
                low=np.array([[-1.0, -2.0], [0.0, 4.0]], dtype=np.float32),
                high=np.array([[1.0, 2.0], [2.0, 8.0]], dtype=np.float32),
                dtype=np.float32,
            )
        )
        mean, half_len = converter._action_bounds_like(
            torch.zeros(3, converter.input_size)
        )
        self.assertEqual(mean.shape, torch.Size([2, 2]))
        self.assertEqual(half_len.shape, torch.Size([2, 2]))
        action = converter(torch.zeros(3, converter.input_size)).rsample()
        self.assertEqual(action.shape, torch.Size([3, 2, 2]))

    def test_stable_entropy_keeps_gradient_for_saturated_mean(self):
        converter = BoxOutputLinearNormalization(
            gym.spaces.Box(
                low=np.array([-1.0], dtype=np.float32),
                high=np.array([1.0], dtype=np.float32),
                dtype=np.float32,
            )
        )
        target_std = torch.tensor(0.5 - 1e-4)
        raw_std = torch.log(torch.expm1(target_std))
        feature = torch.tensor([[10.0, raw_std.item()]], requires_grad=True)

        torch.manual_seed(71)
        entropy = converter(feature).entropy(num_samples=4096).mean()
        self.assertTrue(torch.isfinite(entropy))
        self.assertLess(entropy.item(), -15.0)

        entropy.backward()
        self.assertLess(feature.grad[0, 0].item(), -1.9)

    def test_stable_entropy_includes_affine_action_scale(self):
        unit_converter = BoxOutputLinearNormalization(
            gym.spaces.Box(
                low=np.array([-1.0], dtype=np.float32),
                high=np.array([1.0], dtype=np.float32),
                dtype=np.float32,
            )
        )
        double_converter = BoxOutputLinearNormalization(
            gym.spaces.Box(
                low=np.array([-2.0], dtype=np.float32),
                high=np.array([2.0], dtype=np.float32),
                dtype=np.float32,
            )
        )
        feature = torch.tensor([[0.3, -0.2]])

        torch.manual_seed(79)
        unit_entropy = unit_converter(feature).entropy(num_samples=2048)
        torch.manual_seed(79)
        double_entropy = double_converter(feature).entropy(num_samples=2048)
        torch.testing.assert_close(
            double_entropy - unit_entropy,
            torch.full_like(unit_entropy, math.log(2.0)),
        )


if __name__ == '__main__':
    unittest.main()
