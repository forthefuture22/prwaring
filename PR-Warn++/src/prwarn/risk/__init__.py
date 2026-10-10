"""Continuous wind-side risk proxies."""

from .proxies import FrozenRiskComposer, select_cost_threshold, stylized_event_cost

__all__ = ["FrozenRiskComposer", "select_cost_threshold", "stylized_event_cost"]

from .proxies import (
    MahalanobisOOD,
    aggregate_farm_scenarios,
    cvar_shortfall,
    deviation_probability,
    ramp_probability,
    observed_ramp_event,
)

__all__ = [
    "MahalanobisOOD",
    "aggregate_farm_scenarios",
    "cvar_shortfall",
    "deviation_probability",
    "ramp_probability",
    "observed_ramp_event",
]
