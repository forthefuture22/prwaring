import unittest

import numpy as np

from prwarn.graphs.builders import correlation_graph, directional_graph, geographic_graph
from prwarn.physics.density import air_density, equivalent_wind_speed
from prwarn.physics.power_curve import EmpiricalPowerCurve


class PhysicsGraphTests(unittest.TestCase):
    def test_density_and_equivalent_speed(self):
        rho = air_density(101325.0, 288.15)
        self.assertAlmostEqual(float(rho), 1.225, places=2)
        speed = equivalent_wind_speed(10.0, 1.225)
        self.assertAlmostEqual(float(speed), 10.0, places=3)

    def test_power_curve_is_monotone_and_roundtrips(self):
        rng = np.random.default_rng(2)
        speed = rng.uniform(0, 20, 1000)
        power = np.clip(3.0 * speed**3 + rng.normal(0, 100, 1000), 0, 2000)
        curve = EmpiricalPowerCurve(n_bins=20, minimum_bin_count=10, rated_power=2000).fit(
            speed, power
        )
        self.assertTrue(np.all(np.diff(curve.bin_power_) >= -1e-10))
        restored = EmpiricalPowerCurve.from_dict(curve.to_dict())
        np.testing.assert_allclose(curve.predict([3, 8, 12]), restored.predict([3, 8, 12]))

    def test_graphs_are_row_normalized(self):
        coordinates = np.array([[0, 0], [100, 0], [200, 0]], dtype=float)
        geo = geographic_graph(coordinates, k=1)
        power = np.stack([np.arange(20), np.arange(20) * 2, -np.arange(20)], axis=1)
        corr = correlation_graph(power, mode="raw", k=1)
        direction = directional_graph(
            coordinates, np.array([270.0]), distance_scale=100.0, sigma_degrees=20.0
        )
        np.testing.assert_allclose(geo.sum(axis=-1), 1.0)
        np.testing.assert_allclose(corr.sum(axis=-1), 1.0)
        np.testing.assert_allclose(direction.sum(axis=-1), 1.0)
        # Wind from west propagates east: node 0 has an outgoing edge to node 1.
        self.assertGreater(direction[0, 0, 1], 0.0)


if __name__ == "__main__":
    unittest.main()
