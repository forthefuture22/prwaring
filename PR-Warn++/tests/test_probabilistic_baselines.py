import unittest
from pathlib import Path
import tempfile

import numpy as np

from prwarn.baselines.probabilistic import (
    GaussianCopulaResidual,
    GaussianResidual,
    ResidualBootstrap,
    load_probabilistic_baseline,
    save_probabilistic_baseline,
    _normal_cdf,
    _normal_ppf,
)


class ProbabilisticBaselineTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(11)
        common = rng.normal(size=(500, 1, 1))
        self.error = np.concatenate(
            [common + 0.2 * rng.normal(size=(500, 1, 1)),
             2.0 * common + 0.2 * rng.normal(size=(500, 1, 1))],
            axis=1,
        )
        self.mask = np.ones_like(self.error)
        self.centre = np.full((4, 2, 1), 10.0)

    def test_normal_cdf_ppf_roundtrip(self):
        probability = np.array([0.001, 0.1, 0.5, 0.9, 0.999])
        np.testing.assert_allclose(_normal_cdf(_normal_ppf(probability)), probability, atol=2e-6)

    def test_gaussian_preserves_shape_and_positive_dependence(self):
        model = GaussianResidual(shrinkage=0.05).fit(self.error, self.mask)
        samples = model.sample(self.centre, 300, rated_power=100.0, seed=1)
        self.assertEqual(samples.shape, (300, 4, 2, 1))
        correlation = np.corrcoef((samples[:, 0, :, 0] - 10.0).T)[0, 1]
        self.assertGreater(correlation, 0.8)

    def test_copula_uses_empirical_marginals(self):
        model = GaussianCopulaResidual(shrinkage=0.05).fit(self.error, self.mask)
        residual = model.sample_errors(400, 2, seed=2)
        self.assertEqual(residual.shape, (400, 2, 2, 1))
        self.assertGreater(np.corrcoef(residual[:, 0, :, 0].T)[0, 1], 0.7)
        self.assertGreaterEqual(residual.min(), self.error.min() - 1e-9)
        self.assertLessEqual(residual.max(), self.error.max() + 1e-9)

    def test_bootstrap_resamples_whole_fields(self):
        model = ResidualBootstrap().fit(self.error, self.mask)
        samples = model.sample(self.centre, 20, rated_power=100.0, seed=3)
        for row in (samples[:, 0, :, 0] - 10.0):
            self.assertTrue(np.any(np.all(np.isclose(self.error[:, :, 0], row), axis=1)))

    def test_pickle_free_state_roundtrip_preserves_seeded_samples(self):
        models = (
            GaussianResidual().fit(self.error, self.mask),
            ResidualBootstrap().fit(self.error, self.mask),
            GaussianCopulaResidual().fit(self.error, self.mask),
        )
        with tempfile.TemporaryDirectory() as directory:
            for index, model in enumerate(models):
                path = Path(directory) / f"model_{index}.npz"
                save_probabilistic_baseline(model, path)
                restored = load_probabilistic_baseline(path)
                expected = model.sample(self.centre, 8, rated_power=100.0, seed=77)
                actual = restored.sample(self.centre, 8, rated_power=100.0, seed=77)
                np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
