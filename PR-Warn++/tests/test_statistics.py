import unittest

import numpy as np

from prwarn.eval.statistics import (
    aggregate_seed_metrics,
    contiguous_event_ids,
    diebold_mariano,
    event_bootstrap_ci,
    newey_west_variance,
)


class StatisticalComparisonTests(unittest.TestCase):
    def test_newey_west_iid_lag_zero_matches_population_variance_over_n(self):
        values = np.array([1.0, 2.0, 4.0, 7.0])
        expected = np.var(values, ddof=0) / len(values)
        self.assertAlmostEqual(newey_west_variance(values, lags=0), expected)

    def test_dm_sign_and_zero_variance_cases(self):
        loss_a = np.array([2.0, 3.0, 4.0, 5.0])
        loss_b = loss_a - 1.0
        result = diebold_mariano(loss_a, loss_b, hac_lags=1)
        self.assertGreater(result.statistic, 0.0)
        self.assertEqual(result.p_value, 0.0)
        equal = diebold_mariano(loss_a, loss_a, hac_lags=1)
        self.assertEqual(equal.statistic, 0.0)
        self.assertEqual(equal.p_value, 1.0)

    def test_event_bootstrap_resamples_clusters_and_is_reproducible(self):
        values = np.array([0.0, 0.0, 10.0, 10.0])
        event_ids = np.array([1, 1, 2, 2])
        first = event_bootstrap_ci(
            values, event_ids, n_boot=300, seed=5, confidence=0.9
        )
        second = event_bootstrap_ci(
            values, event_ids, n_boot=300, seed=5, confidence=0.9
        )
        self.assertEqual(first, second)
        self.assertEqual(first.estimate, 5.0)
        self.assertEqual(first.n_events, 2)
        self.assertLessEqual(first.lower, first.estimate)
        self.assertGreaterEqual(first.upper, first.estimate)

    def test_seed_aggregation_flattens_nested_metrics(self):
        summary = aggregate_seed_metrics(
            [
                {"seed": 1, "point": {"mae": 2.0}, "name": "a"},
                {"seed": 2, "point": {"mae": 4.0}, "name": "b"},
            ]
        )
        self.assertEqual(summary["point.mae"]["mean"], 3.0)
        self.assertAlmostEqual(summary["point.mae"]["std"], np.sqrt(2.0))
        self.assertEqual(summary["point.mae"]["count"], 2)

    def test_contiguous_event_ids_split_separate_episodes(self):
        timestamps = np.arange(7).astype("timedelta64[m]") + np.datetime64(
            "2021-01-01"
        )
        identifiers = contiguous_event_ids(
            np.array([0, 1, 1, 0, 1, 1, 0]),
            timestamps,
            maximum_gap=np.timedelta64(1, "m"),
        )
        np.testing.assert_array_equal(identifiers, [-1, 0, 0, -1, 1, 1, -1])


if __name__ == "__main__":
    unittest.main()
