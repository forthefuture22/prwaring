"""Scenario-derived continuous wind-side risk proxies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def aggregate_farm_scenarios(turbine_scenarios: np.ndarray) -> np.ndarray:
    """Sum `[scenario,batch,node,horizon]` to `[scenario,batch,horizon]`."""

    scenarios = np.asarray(turbine_scenarios, dtype=float)
    if scenarios.ndim != 4:
        raise ValueError("turbine scenarios must have [scenario,batch,node,horizon]")
    return scenarios.sum(axis=2)


def _prepend_history(
    farm_scenarios: np.ndarray, current_power: np.ndarray | float | None
) -> np.ndarray:
    scenarios = np.asarray(farm_scenarios, dtype=float)
    if scenarios.ndim != 3:
        raise ValueError("farm scenarios must have [scenario,batch,horizon]")
    if current_power is None:
        return scenarios
    current = np.asarray(current_power, dtype=float)
    if current.ndim == 0:
        current = np.full(scenarios.shape[1], float(current))
    if current.shape != (scenarios.shape[1],):
        raise ValueError("current_power must be scalar or [batch]")
    prefix = np.broadcast_to(current[None, :, None], (scenarios.shape[0], scenarios.shape[1], 1))
    return np.concatenate([prefix, scenarios], axis=-1)


def ramp_probability(
    farm_scenarios: np.ndarray,
    *,
    threshold: float,
    duration_steps: int = 1,
    direction: str = "down",
    current_power: np.ndarray | float | None = None,
) -> np.ndarray:
    """Estimate per-horizon ramp probability by Monte Carlo event frequency."""

    if threshold <= 0 or duration_steps <= 0:
        raise ValueError("threshold and duration_steps must be positive")
    extended = _prepend_history(farm_scenarios, current_power)
    offset = 1 if current_power is not None else 0
    horizon = extended.shape[-1] - offset
    probability = np.full((extended.shape[1], horizon), np.nan, dtype=float)
    for h in range(horizon):
        end = h + offset
        start = end - duration_steps
        if start < 0:
            continue
        change = extended[..., end] - extended[..., start]
        if direction == "down":
            event = change < -threshold
        elif direction == "up":
            event = change > threshold
        else:
            raise ValueError("direction must be 'up' or 'down'")
        probability[:, h] = event.mean(axis=0)
    return probability


def observed_ramp_event(
    future_power: np.ndarray,
    *,
    threshold: float,
    duration_steps: int = 1,
    direction: str = "down",
    current_power: np.ndarray,
    future_mask: np.ndarray | None = None,
    current_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build future ramp truth and its validity mask without using it as input."""

    future = np.asarray(future_power, dtype=float)
    current = np.asarray(current_power, dtype=float)
    if future.ndim != 2 or current.shape != (future.shape[0],):
        raise ValueError("future_power must be [sample,horizon], current_power [sample]")
    if threshold <= 0 or duration_steps <= 0:
        raise ValueError("threshold and duration_steps must be positive")
    extended = np.concatenate([current[:, None], future], axis=1)
    if future_mask is None:
        valid_future = np.isfinite(future)
    else:
        valid_future = np.asarray(future_mask, dtype=bool) & np.isfinite(future)
    if current_mask is None:
        valid_current = np.isfinite(current)
    else:
        valid_current = np.asarray(current_mask, dtype=bool) & np.isfinite(current)
    extended_valid = np.concatenate([valid_current[:, None], valid_future], axis=1)
    event = np.zeros_like(future, dtype=np.float32)
    valid = np.zeros_like(future, dtype=bool)
    for h in range(future.shape[1]):
        end = h + 1
        start = end - duration_steps
        if start < 0:
            continue
        change = extended[:, end] - extended[:, start]
        if direction == "down":
            event[:, h] = change < -threshold
        elif direction == "up":
            event[:, h] = change > threshold
        else:
            raise ValueError("direction must be 'up' or 'down'")
        valid[:, h] = extended_valid[:, end] & extended_valid[:, start]
    return event, valid


def deviation_probability(
    farm_scenarios: np.ndarray,
    reference: np.ndarray,
    *,
    threshold: float,
    direction: str,
) -> np.ndarray:
    scenarios = np.asarray(farm_scenarios, dtype=float)
    ref = np.asarray(reference, dtype=float)
    if scenarios.ndim != 3 or ref.shape != scenarios.shape[1:]:
        raise ValueError("scenarios must be [scenario,batch,horizon], reference [batch,horizon]")
    delta = scenarios - ref[None]
    if direction == "down":
        return (delta < -threshold).mean(axis=0)
    if direction == "up":
        return (delta > threshold).mean(axis=0)
    raise ValueError("direction must be 'up' or 'down'")


def cvar_shortfall(
    farm_scenarios: np.ndarray,
    reference: np.ndarray,
    *,
    alpha: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Return VaR and CVaR of `max(0, reference - scenario)` per sample/horizon."""

    scenarios = np.asarray(farm_scenarios, dtype=float)
    ref = np.asarray(reference, dtype=float)
    if scenarios.ndim != 3 or ref.shape != scenarios.shape[1:]:
        raise ValueError("shape mismatch")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1")
    loss = np.maximum(0.0, ref[None] - scenarios)
    var = np.quantile(loss, alpha, axis=0, method="higher")
    tail = loss >= var[None]
    numerator = np.where(tail, loss, 0.0).sum(axis=0)
    denominator = tail.sum(axis=0)
    cvar = np.divide(numerator, denominator, where=denominator > 0)
    return var, cvar


@dataclass
class MahalanobisOOD:
    """Ridge-regularized Mahalanobis score fitted on Train representations."""

    ridge: float = 1e-4
    mean_: np.ndarray | None = None
    precision_: np.ndarray | None = None

    def fit(self, train_representation: np.ndarray) -> "MahalanobisOOD":
        values = np.asarray(train_representation, dtype=float)
        if values.ndim != 2 or values.shape[0] < 2:
            raise ValueError("train representation must have [sample,feature]")
        if not np.isfinite(values).all():
            raise ValueError("train representation contains non-finite values")
        self.mean_ = values.mean(axis=0)
        covariance = np.cov(values, rowvar=False, ddof=1)
        covariance = np.atleast_2d(covariance)
        scale = float(np.trace(covariance) / covariance.shape[0])
        covariance += np.eye(covariance.shape[0]) * self.ridge * max(scale, 1.0)
        self.precision_ = np.linalg.pinv(covariance)
        return self

    def score(self, representation: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.precision_ is None:
            raise RuntimeError("OOD detector is not fitted")
        values = np.asarray(representation, dtype=float)
        if values.shape[-1] != self.mean_.shape[0]:
            raise ValueError("feature dimension mismatch")
        delta = values - self.mean_
        squared = np.einsum("...i,ij,...j->...", delta, self.precision_, delta)
        return np.sqrt(np.maximum(squared, 0.0))


def data_quality_delta(clean_metric: np.ndarray, degraded_metric: np.ndarray) -> np.ndarray:
    """Controlled-intervention degradation; positive means performance worsened."""

    clean = np.asarray(clean_metric, dtype=float)
    degraded = np.asarray(degraded_metric, dtype=float)
    if clean.shape != degraded.shape:
        raise ValueError("metric shapes differ")
    return degraded - clean


def _logit(probability: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    clipped = np.clip(np.asarray(probability, dtype=float), epsilon, 1.0 - epsilon)
    return np.log(clipped) - np.log1p(-clipped)


@dataclass
class FrozenRiskComposer:
    """Combine scenario event probability with frozen auxiliary diagnostics.

    Diagnostic locations/scales are fitted on Calib only. Positive auxiliary
    weights mean that larger OOD or data-quality scores increase risk.
    """

    auxiliary_terms: tuple[str, ...] = ()
    ood_weight: float = 0.25
    data_quality_weight: float = 0.25
    locations_: dict[str, float] | None = None
    scales_: dict[str, float] | None = None

    def fit(self, **calibration_diagnostics: np.ndarray) -> "FrozenRiskComposer":
        allowed = {"ood", "data_quality"}
        unknown = set(self.auxiliary_terms) - allowed
        if unknown:
            raise ValueError(f"unsupported auxiliary terms: {sorted(unknown)}")
        self.locations_, self.scales_ = {}, {}
        for name in self.auxiliary_terms:
            if name not in calibration_diagnostics:
                raise ValueError(f"missing Calib diagnostic: {name}")
            values = np.asarray(calibration_diagnostics[name], dtype=float)
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                raise ValueError(f"Calib diagnostic {name} has no finite values")
            location = float(np.median(finite))
            mad = float(np.median(np.abs(finite - location)))
            # 1.4826 * MAD is a robust standard-deviation estimate.
            self.locations_[name] = location
            self.scales_[name] = max(1.4826 * mad, 1e-6)
        return self

    def transform(
        self, core_probability: np.ndarray, **diagnostics: np.ndarray
    ) -> np.ndarray:
        if self.locations_ is None or self.scales_ is None:
            raise RuntimeError("risk composer is not fitted")
        core = np.asarray(core_probability, dtype=float)
        score = _logit(core)
        weights = {"ood": self.ood_weight, "data_quality": self.data_quality_weight}
        for name in self.auxiliary_terms:
            if name not in diagnostics:
                raise ValueError(f"missing diagnostic: {name}")
            values = np.asarray(diagnostics[name], dtype=float)
            if values.shape != core.shape:
                try:
                    values = np.broadcast_to(values, core.shape)
                except ValueError as exc:
                    raise ValueError(f"diagnostic {name} cannot broadcast to probability") from exc
            standardized = (values - self.locations_[name]) / self.scales_[name]
            score = score + weights[name] * np.clip(standardized, -5.0, 5.0)
        return 1.0 / (1.0 + np.exp(-np.clip(score, -40.0, 40.0)))


def stylized_event_cost(
    event: np.ndarray,
    probability: np.ndarray,
    *,
    threshold: float,
    false_negative_cost: float = 5.0,
    false_positive_cost: float = 1.0,
) -> float:
    """Mean asymmetric decision cost; this is a proxy, not grid economics."""

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0,1]")
    if false_negative_cost < 0 or false_positive_cost < 0:
        raise ValueError("costs must be non-negative")
    truth = np.asarray(event, dtype=float)
    probability = np.asarray(probability, dtype=float)
    if truth.shape != probability.shape:
        raise ValueError("event and probability shapes differ")
    valid = np.isfinite(truth) & np.isfinite(probability)
    if not valid.any():
        raise ValueError("no finite event probabilities")
    truth = truth[valid].astype(bool)
    decision = probability[valid] >= threshold
    cost = false_negative_cost * (truth & ~decision)
    cost = cost + false_positive_cost * (~truth & decision)
    return float(np.mean(cost))


def select_cost_threshold(
    event: np.ndarray,
    probability: np.ndarray,
    *,
    false_negative_cost: float = 5.0,
    false_positive_cost: float = 1.0,
    grid_size: int = 201,
) -> float:
    """Select an operating threshold using Calib observations only."""

    if grid_size < 2:
        raise ValueError("grid_size must be at least 2")
    thresholds = np.linspace(0.0, 1.0, grid_size)
    costs = [
        stylized_event_cost(
            event, probability, threshold=float(value),
            false_negative_cost=false_negative_cost,
            false_positive_cost=false_positive_cost,
        )
        for value in thresholds
    ]
    return float(thresholds[int(np.argmin(costs))])
