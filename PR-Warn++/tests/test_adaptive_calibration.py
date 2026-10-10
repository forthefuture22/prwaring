import unittest

import numpy as np

from prwarn.calibration.adaptive import (
    AdaptiveConformalIntervals,
    ContextFallbackConformal,
)


class AdaptiveCalibrationTests(unittest.TestCase):
    def test_aci_predicts_before_using_current_target(self):
        lower = np.zeros((20, 2))
        upper = np.ones((20, 2))
        target = np.full((20, 2), 0.5)
        first = AdaptiveConformalIntervals(alpha=0.1, gamma=0.05).fit(
            lower, upper, target
        )
        test_lower = np.zeros((3, 2))
        test_upper = np.ones((3, 2))
        ordinary = np.full((3, 2), 0.5)
        shifted = ordinary.copy()
        shifted[0] = 10.0
        ordinary_result = first.transform_stream(
            test_lower[:1], test_upper[:1], ordinary[:1]
        )

        second = AdaptiveConformalIntervals(alpha=0.1, gamma=0.05).fit(
            lower, upper, target
        )
        shifted_result = second.transform_stream(test_lower, test_upper, shifted)
        np.testing.assert_array_equal(
            ordinary_result.correction[0], shifted_result.correction[0]
        )
        self.assertTrue(shifted_result.miss[0].all())
        self.assertTrue(
            np.all(shifted_result.alpha_before_update[1] < shifted_result.alpha_before_update[0])
        )

    def test_aci_ignores_invalid_updates(self):
        lo = np.zeros((10, 1))
        hi = np.ones((10, 1))
        y = np.full((10, 1), 0.5)
        model = AdaptiveConformalIntervals(alpha=0.1, gamma=0.05).fit(lo, hi, y)
        result = model.transform_stream(
            np.zeros((2, 1)),
            np.ones((2, 1)),
            np.array([[100.0], [0.5]]),
            mask=np.array([[0], [1]]),
        )
        self.assertEqual(result.alpha_before_update[0, 0], result.alpha_before_update[1, 0])

    def test_context_fallback_triggers_for_far_context(self):
        context = np.array([[0.0], [0.1], [0.2], [1.0], [1.1], [1.2]])
        lower = np.zeros((6, 1))
        upper = np.ones((6, 1))
        target = np.array([[0.5], [0.5], [0.5], [2.0], [3.0], [4.0]])
        subgroup = np.array([0, 0, 0, 1, 1, 1])
        model = ContextFallbackConformal(
            alpha=0.2, neighbors=3, ood_quantile=1.0
        ).fit(lower, upper, target, context, subgroup=subgroup)
        result = model.transform(
            np.zeros((2, 1)), np.ones((2, 1)), np.array([[0.15], [100.0]])
        )
        self.assertFalse(result.fallback_triggered[0])
        self.assertTrue(result.fallback_triggered[1])
        self.assertEqual(result.correction[1, 0], model.fallback_[0])
        self.assertGreaterEqual(result.correction[1, 0], 3.0)


if __name__ == "__main__":
    unittest.main()
