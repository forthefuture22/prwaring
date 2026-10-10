import unittest

import numpy as np
import pandas as pd

from prwarn.cli.attach_future_weather import _align_weather
from prwarn.data.imputation import impute_history
from prwarn.data.weather_protocol import (
    audit_weather_availability,
    select_latest_available_issue,
)


class WeatherProtocolTests(unittest.TestCase):
    def test_history_and_issue_time_leakage_gates(self):
        with self.assertRaises(ValueError):
            audit_weather_availability([10], [11], protocol="history_only")
        report = audit_weather_availability(
            [10, 10], [11, 12], issue_time=[8, 10],
            protocol="issue_time_forecast",
        )
        self.assertTrue(report.deployable)
        with self.assertRaises(ValueError):
            audit_weather_availability(
                [10], [12], issue_time=[11], protocol="issue_time_forecast"
            )
        oracle = audit_weather_availability([10], [12], protocol="oracle_era5")
        self.assertFalse(oracle.deployable)

    def test_latest_available_issue_never_uses_future_issue(self):
        selected = select_latest_available_issue(
            np.array([10, 20]), np.array([[30], [30]]),
            np.array([5, 15, 25]), np.array([30, 30, 30]),
        )
        np.testing.assert_array_equal(selected, np.array([[0], [1]]))

    def test_weather_alignment_selects_latest_nonfuture_issue(self):
        origin = np.array(["2021-01-01T00:00"], dtype="datetime64[ns]")
        archive = pd.DataFrame(
            {
                "TurbID": [1, 1, 1],
                "valid_time": ["2021-01-01T00:10"] * 3,
                "issue_time": [
                    "2020-12-31T23:00", "2021-01-01T00:00", "2021-01-01T00:05"
                ],
                "wind": [1.0, 2.0, 999.0],
            }
        )
        values, mask, report = _align_weather(
            origin, np.array([1]), 1, 10, archive,
            protocol="issue_time_forecast", turbine_column="TurbID",
            valid_column="valid_time", issue_column="issue_time",
            feature_columns=["wind"],
        )
        self.assertEqual(values[0, 0, 0, 0], 2.0)
        self.assertTrue(mask.all())
        self.assertEqual(report["leakage_violations"], 0)


class ImputationTests(unittest.TestCase):
    def test_all_strategies_are_finite_and_preserve_observations(self):
        values = np.array([[[1.0]], [[np.nan]], [[3.0]], [[np.nan]]])
        mask = np.isfinite(values)
        for strategy in ("train_mean", "forward_fill", "causal_linear", "grud_decay"):
            filled = impute_history(values, mask, np.array([[0.0]]), strategy=strategy)
            self.assertTrue(np.isfinite(filled).all())
            np.testing.assert_array_equal(filled[mask], values[mask])

    def test_causal_outputs_do_not_depend_on_later_observation(self):
        first = np.array([[[1.0]], [[np.nan]], [[3.0]]])
        second = np.array([[[1.0]], [[np.nan]], [[300.0]]])
        mask = np.isfinite(first)
        for strategy in ("forward_fill", "causal_linear", "grud_decay"):
            a = impute_history(first, mask, [[0.0]], strategy=strategy)
            b = impute_history(second, mask, [[0.0]], strategy=strategy)
            self.assertEqual(a[1, 0, 0], b[1, 0, 0])


if __name__ == "__main__":
    unittest.main()
