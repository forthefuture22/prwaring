"""CPU-only smoke test for the data, physics, calibration and risk chain."""

from __future__ import annotations

import json

import numpy as np

from prwarn.calibration.split import PerHorizonSplitConformal, scenario_interval
from prwarn.eval.metrics import crps_ensemble, energy_score
from prwarn.graphs.builders import correlation_graph, directional_graph, geographic_graph
from prwarn.physics.density import air_density, equivalent_wind_speed
from prwarn.physics.power_curve import EmpiricalPowerCurve
from prwarn.risk.proxies import aggregate_farm_scenarios, cvar_shortfall, ramp_probability


def main() -> None:
    rng = np.random.default_rng(2025)
    n_train, n_batch, n_nodes, horizon, n_scenarios = 600, 8, 5, 6, 50
    coordinates = np.stack([np.arange(n_nodes) * 500.0, np.zeros(n_nodes)], axis=1)
    wind = rng.uniform(3.0, 15.0, n_train)
    pressure = rng.normal(101_325.0, 800.0, n_train)
    temperature = rng.normal(285.0, 8.0, n_train)
    density = air_density(pressure, temperature)
    equivalent = equivalent_wind_speed(wind, density)
    power = np.clip(18.0 * equivalent**3 + rng.normal(0.0, 80.0, n_train), 0.0, 2000.0)
    curve = EmpiricalPowerCurve(n_bins=20, minimum_bin_count=8, rated_power=2000.0).fit(
        equivalent, power
    )

    history_power = rng.normal(1000.0, 150.0, (200, n_nodes))
    geo = geographic_graph(coordinates, k=2)
    corr = correlation_graph(history_power, mode="difference", k=2)
    direction = directional_graph(
        coordinates,
        np.full(n_batch, 270.0),
        distance_scale=1000.0,
    )

    truth = rng.normal(1000.0, 120.0, (n_batch, n_nodes, horizon))
    scenarios = truth[None] + rng.normal(
        0.0, 140.0, (n_scenarios, n_batch, n_nodes, horizon)
    )
    farm_truth = truth.sum(axis=1)
    farm_scenarios = aggregate_farm_scenarios(scenarios)
    lower, upper = scenario_interval(farm_scenarios, alpha=0.10)
    conformal = PerHorizonSplitConformal(alpha=0.10).fit(lower, upper, farm_truth)
    calibrated_lower, calibrated_upper = conformal.transform(lower, upper)
    probability = ramp_probability(
        farm_scenarios,
        threshold=300.0,
        duration_steps=1,
        direction="down",
        current_power=farm_truth[:, 0],
    )
    var, cvar = cvar_shortfall(farm_scenarios, farm_truth, alpha=0.95)

    summary = {
        "power_curve_points": int(len(curve.bin_centres_)),
        "graph_shapes": [list(geo.shape), list(corr.shape), list(direction.shape)],
        "scenario_shape": list(scenarios.shape),
        "crps": crps_ensemble(truth, scenarios),
        "energy_score": energy_score(truth, scenarios),
        "conformal_correction": conformal.correction_.round(3).tolist(),
        "interval_shape": list(calibrated_lower.shape),
        "ramp_probability_shape": list(probability.shape),
        "cvar_shape": list(cvar.shape),
        "var_nonnegative": bool(np.all(var >= 0)),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
