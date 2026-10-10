import unittest

import numpy as np

from prwarn.calibration.split import PerHorizonSplitConformal, scenario_interval
from prwarn.eval.metrics import (
    average_precision,
    brier_score,
    brier_skill_score,
    binary_event_metrics,
    crps_ensemble,
    crps_ensemble_values,
    energy_score,
    energy_score_values,
    interval_metrics,
    ks_distance,
    masked_r2,
    pinball_loss,
)
from prwarn.risk.proxies import (
    FrozenRiskComposer,
    MahalanobisOOD,
    aggregate_farm_scenarios,
    cvar_shortfall,
    observed_ramp_event,
    ramp_probability,
    select_cost_threshold,
    stylized_event_cost,
)


class CalibrationRiskEvaluationTests(unittest.TestCase):
    def test_conformal_only_widens_intervals(self):
        truth = np.array([[0.0, 5.0], [3.0, 6.0], [4.0, 8.0], [2.0, 7.0]])
        lower = truth + 1.0
        upper = truth + 2.0
        model = PerHorizonSplitConformal(alpha=0.25).fit(lower, upper, truth)
        lo, hi = model.transform(lower, upper)
        self.assertTrue(np.all(lo <= lower))
        self.assertTrue(np.all(hi >= upper))
        self.assertTrue(np.all(model.correction_ >= 1.0))

    def test_scenario_risk_shapes_and_exact_ramp(self):
        turbine = np.array(
            [
                [[[90.0, 70.0], [60.0, 50.0]]],
                [[[110.0, 100.0], [90.0, 80.0]]],
            ]
        )
        farm = aggregate_farm_scenarios(turbine)
        probability = ramp_probability(
            farm, threshold=20.0, current_power=np.array([200.0]), direction="down"
        )
        self.assertEqual(probability.shape, (1, 2))
        self.assertAlmostEqual(probability[0, 0], 0.5)
        lower, upper = scenario_interval(farm)
        var, cvar = cvar_shortfall(farm, np.full((1, 2), 200.0), alpha=0.5)
        self.assertEqual(lower.shape, (1, 2))
        self.assertTrue(np.all(cvar >= var))
        self.assertTrue(np.all(upper >= lower))

    def test_ood_and_metrics(self):
        rng = np.random.default_rng(3)
        train = rng.normal(size=(100, 4))
        ood = MahalanobisOOD().fit(train)
        near = ood.score(np.zeros((2, 4)))
        far = ood.score(np.full((2, 4), 10.0))
        self.assertTrue(np.all(far > near))

        truth = np.zeros((3, 2))
        scenarios = np.zeros((5, 3, 2))
        self.assertEqual(crps_ensemble(truth, scenarios), 0.0)
        self.assertEqual(energy_score(truth, scenarios), 0.0)
        np.testing.assert_array_equal(energy_score_values(truth, scenarios), np.zeros(3))
        self.assertAlmostEqual(brier_score([0, 1], [0.0, 1.0]), 0.0)
        self.assertAlmostEqual(average_precision([0, 1, 1], [0.1, 0.9, 0.8]), 1.0)
        self.assertEqual(masked_r2([1, 2, 3], [1, 2, 3]), 1.0)
        self.assertEqual(pinball_loss([1.0], [1.0], 0.5), 0.0)
        self.assertEqual(brier_skill_score([0, 1], [0, 1], [0.5, 0.5]), 1.0)
        self.assertEqual(binary_event_metrics([0, 1], [0.1, 0.9])["f1"], 1.0)
        self.assertEqual(ks_distance([0, 1], [0, 1]), 0.0)

        samples = rng.normal(size=(7, 3, 2))
        target = rng.normal(size=(3, 2))
        brute = np.mean(np.abs(samples - target[None]), axis=0) - 0.5 * np.mean(
            np.abs(samples[:, None] - samples[None, :]), axis=(0, 1)
        )
        self.assertAlmostEqual(crps_ensemble(target, samples), float(brute.mean()))
        np.testing.assert_allclose(crps_ensemble_values(target, samples), brute)

    def test_interval_and_observed_ramp_metrics(self):
        interval = interval_metrics(
            np.array([1.0, 5.0]),
            np.array([0.0, 1.0]),
            np.array([2.0, 3.0]),
            alpha=0.1,
            normalization=10.0,
        )
        self.assertAlmostEqual(interval["picp"], 0.5)
        self.assertAlmostEqual(interval["pinaw"], 0.2)
        event, valid = observed_ramp_event(
            np.array([[80.0, 50.0, 55.0]]),
            threshold=15.0,
            duration_steps=1,
            direction="down",
            current_power=np.array([100.0]),
            future_mask=np.ones((1, 3)),
            current_mask=np.ones(1),
        )
        np.testing.assert_array_equal(event, np.array([[1.0, 1.0, 0.0]]))
        self.assertTrue(valid.all())

    def test_risk_composer_is_frozen_and_cost_threshold_uses_calib(self):
        core = np.array([0.2, 0.2, 0.8, 0.8])
        ood = np.array([0.0, 1.0, 2.0, 3.0])
        composer = FrozenRiskComposer(("ood",), ood_weight=0.5).fit(ood=ood)
        augmented = composer.transform(core, ood=ood)
        self.assertLess(augmented[0], core[0])
        self.assertGreater(augmented[-1], core[-1])
        frozen_location = composer.locations_["ood"]
        composer.transform(core, ood=ood + 1000.0)
        self.assertEqual(composer.locations_["ood"], frozen_location)
        event = np.array([0, 0, 1, 1])
        threshold = select_cost_threshold(event, augmented)
        self.assertGreaterEqual(threshold, 0.0)
        self.assertLessEqual(threshold, 1.0)
        self.assertGreaterEqual(
            stylized_event_cost(event, augmented, threshold=threshold), 0.0
        )


if __name__ == "__main__":
    unittest.main()
