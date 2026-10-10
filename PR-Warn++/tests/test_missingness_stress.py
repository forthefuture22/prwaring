import unittest

import numpy as np

from prwarn.data.stress import (
    apply_missingness_stress,
    block_missingness,
    extreme_conditioned_missingness,
    mcar_missingness,
    rebuild_issue_time_derived,
    spatial_outage_missingness,
)


class MissingnessStressTests(unittest.TestCase):
    def setUp(self):
        self.x = np.arange(2 * 4 * 8 * 3, dtype=np.float32).reshape(2, 4, 8, 3)
        self.mask = np.ones_like(self.x)
        self.delta = np.zeros_like(self.x)

    def test_mcar_is_reproducible_and_only_selects_observed(self):
        self.mask[0, 0, 0, 0] = 0
        first = mcar_missingness(self.mask, rate=0.3, seed=7)
        second = mcar_missingness(self.mask, rate=0.3, seed=7)
        np.testing.assert_array_equal(first, second)
        self.assertFalse(first[0, 0, 0, 0])

    def test_block_missingness_is_contiguous(self):
        selected = block_missingness(
            self.mask, block_length=3, blocks_per_sample=1, seed=8
        )
        for b in range(selected.shape[0]):
            indices = np.argwhere(selected[b])
            self.assertEqual(indices.shape[0], 3)
            self.assertEqual(len(np.unique(indices[:, 0])), 1)
            self.assertEqual(len(np.unique(indices[:, 2])), 1)
            np.testing.assert_array_equal(np.diff(np.sort(indices[:, 1])), np.ones(2))

    def test_spatial_outage_shares_time_block_and_all_features(self):
        selected = spatial_outage_missingness(
            self.mask[:1], node_fraction=0.5, duration=2, seed=3
        )[0]
        affected_nodes = np.flatnonzero(selected.any(axis=(1, 2)))
        self.assertEqual(len(affected_nodes), 2)
        for node in affected_nodes:
            affected_times = np.flatnonzero(selected[node].any(axis=1))
            self.assertEqual(len(affected_times), 2)
            self.assertTrue(selected[node, affected_times, :].all())

    def test_apply_stress_does_not_mutate_and_updates_elapsed_time(self):
        selected = np.zeros_like(self.mask, dtype=bool)
        selected[:, :, 2:5, :] = True
        x_before = self.x.copy()
        result = apply_missingness_stress(
            self.x, self.mask, self.delta, selected, step_minutes=10.0
        )
        np.testing.assert_array_equal(self.x, x_before)
        self.assertTrue(np.all(result.x[:, :, 2:5, :] == 0.0))
        np.testing.assert_array_equal(
            result.delta_t[0, 0, 2:5, 0], np.array([10.0, 20.0, 30.0])
        )
        self.assertEqual(result.delta_t[0, 0, 5, 0], 0.0)

    def test_extreme_conditioned_can_target_only_extreme_history(self):
        extreme = np.zeros((2, 4, 8, 1), dtype=bool)
        extreme[:, :, 3:5] = True
        selected = extreme_conditioned_missingness(
            self.mask,
            extreme,
            base_rate=0.0,
            extreme_multiplier=100.0,
            seed=1,
        )
        # Zero base rate remains zero even under multiplication.
        self.assertFalse(selected.any())
        selected = extreme_conditioned_missingness(
            self.mask,
            extreme,
            base_rate=1.0,
            extreme_multiplier=0.0,
            seed=1,
        )
        self.assertFalse(selected[:, :, 3:5].any())
        self.assertTrue(selected[:, :, :3].all())

    def test_forward_fill_never_reuses_an_artificially_hidden_value(self):
        x = np.array([[[[1.0], [2.0], [3.0], [4.0]]]])
        mask = np.ones_like(x)
        selected = np.zeros_like(x, dtype=bool)
        selected[:, :, 1:3] = True
        result = apply_missingness_stress(
            x,
            mask,
            np.zeros_like(x),
            selected,
            step_minutes=10.0,
            fill_strategy="forward_fill",
        )
        np.testing.assert_array_equal(result.x.reshape(-1), [1.0, 1.0, 1.0, 4.0])

    def test_forward_fill_uses_safe_fallback_for_hidden_window_start(self):
        x = np.array([[[[99.0], [2.0]]]])
        selected = np.zeros_like(x, dtype=bool)
        selected[:, :, 0] = True
        result = apply_missingness_stress(
            x,
            np.ones_like(x),
            np.zeros_like(x),
            selected,
            step_minutes=10.0,
            fill_value=-1.0,
            fill_strategy="forward_fill",
        )
        np.testing.assert_array_equal(result.x.reshape(-1), [-1.0, 2.0])

    def test_rebuild_issue_time_quantities_uses_stressed_carrier_only(self):
        # Features: speed, pressure, temperature, power, relative direction, nacelle.
        raw = np.array(
            [[[[4.0, 101325.0, 288.15, 10.0, 20.0, 30.0],
               [8.0, 101325.0, 288.15, 20.0, 40.0, 50.0]]]],
            dtype=float,
        )
        metadata = {
            "feature_index": {"Wspd": 0, "Sp": 1, "T2m": 2, "Patv": 3, "Wdir": 4, "Ndir": 5},
            "feature_scaler": {"mean": [0.0] * 6, "std": [1.0] * 6},
            "physical_columns": {
                "wind_speed": "Wspd", "pressure": "Sp",
                "temperature": "T2m", "target": "Patv",
            },
            "physical_units": {"pressure_input": "pa", "temperature_input": "kelvin"},
            "power_curve": {
                "n_bins": 2, "minimum_bin_count": 1, "rated_power": 100.0,
                "bin_centres": [0.0, 10.0], "bin_power": [0.0, 100.0],
                "bin_counts": [10, 10],
            },
            "forecast_steps": 2,
            "density_correction": True,
            "wind_direction": {
                "mode": "relative_plus_nacelle", "wind_feature": "Wdir",
                "nacelle_feature": "Ndir",
            },
        }
        selected = np.zeros_like(raw, dtype=bool)
        selected[:, :, -1, [0, 3, 4, 5]] = True
        stressed = apply_missingness_stress(
            raw,
            np.ones_like(raw),
            np.zeros_like(raw),
            selected,
            step_minutes=10.0,
            fill_strategy="forward_fill",
        )
        derived = rebuild_issue_time_derived(stressed.x, stressed.mask, metadata)
        self.assertTrue(np.allclose(derived.p_pc[..., 0], 40.0, atol=2.0))
        self.assertEqual(derived.current_y[0, 0], 10.0)
        self.assertEqual(derived.current_y_mask[0, 0], 0.0)
        self.assertEqual(derived.wind_from[0, 0], 50.0)


if __name__ == "__main__":
    unittest.main()
