import unittest

import numpy as np
import pandas as pd

from prwarn.data.grid import (
    ReasonCode,
    build_rigid_grid,
    build_sdwpf_bundle,
    bundle_to_time_node,
)
from prwarn.data.split import chronological_split
from prwarn.data.window import make_windows
from prwarn.cli.preprocess_sdwpf import _history_only_power_curve


class DataPipelineTests(unittest.TestCase):
    def _frame(self):
        rows = []
        for turbine in (1, 2):
            for step in (0, 1, 3, 4, 5, 6, 7, 8, 9):
                rows.append(
                    {
                        "TurbID": turbine,
                        "Tmstamp": pd.Timestamp("2021-01-01") + pd.Timedelta(minutes=10 * step),
                        "Patv": 100.0 + step,
                        "Wspd": 5.0,
                        "Pab1": 1.0,
                        "Pab2": 1.0,
                        "Pab3": 1.0,
                        "Ndir": 0.0,
                        "Wdir": 0.0,
                    }
                )
        return pd.DataFrame(rows)

    def test_rigid_grid_and_bundle_preserve_missing_step(self):
        rigid = build_rigid_grid(self._frame())
        self.assertEqual(len(rigid), 20)
        means = {"Patv": 105.0, "Wspd": 5.0}
        bundle = build_sdwpf_bundle(rigid, ["Patv", "Wspd"], train_means=means)
        tensor = bundle_to_time_node(bundle)
        self.assertEqual(tensor["x_fill"].shape, (10, 2, 2))
        self.assertTrue(np.all(tensor["reason_code"][2] == ReasonCode.MISSING))
        self.assertTrue(np.all(tensor["mask"][2] == 0.0))
        self.assertTrue(np.all(tensor["delta_t"][2] == 10.0))

    def test_official_unknown_is_not_valid_zero(self):
        frame = self._frame()
        frame.loc[0, "Patv"] = 0.0
        frame.loc[0, "Wspd"] = 6.0
        rigid = build_rigid_grid(frame)
        bundle = build_sdwpf_bundle(rigid, ["Patv", "Wspd"], train_means={"Patv": 100, "Wspd": 5})
        selected = (bundle["frame"]["TurbID"] == 1) & (
            bundle["frame"]["Tmstamp"] == pd.Timestamp("2021-01-01")
        )
        self.assertEqual(bundle["reason_code"][selected.to_numpy()][0], ReasonCode.UNKNOWN)

    def test_chronological_splits_are_disjoint(self):
        rigid = build_rigid_grid(self._frame())
        splits, manifest = chronological_split(rigid)
        time_sets = [set(pd.to_datetime(s["Tmstamp"])) for s in splits.values()]
        for i in range(len(time_sets)):
            for j in range(i + 1, len(time_sets)):
                self.assertFalse(time_sets[i] & time_sets[j])
        self.assertEqual(sum(x["n_timestamps"] for x in manifest.values()), 10)

    def test_window_shapes_and_future_offset(self):
        time, nodes, features = 10, 2, 3
        x = np.arange(time * nodes * features).reshape(time, nodes, features)
        mask = np.ones_like(x)
        delta = np.zeros_like(x)
        y = np.arange(time * nodes).reshape(time, nodes)
        result = make_windows(
            x, mask, delta, y, np.ones_like(y), history_steps=4, forecast_steps=2
        )
        self.assertEqual(result["x"].shape, (5, nodes, 4, features))
        self.assertEqual(result["y"].shape, (5, nodes, 2))
        np.testing.assert_array_equal(result["y"][0], y[4:6].T)

    def test_history_only_power_curve_never_reads_future(self):
        values = np.array([[10.0], [20.0], [999.0], [888.0]])
        result = _history_only_power_curve(values, np.array([1]), horizon=2)
        np.testing.assert_array_equal(result, np.array([[[20.0, 20.0]]], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
