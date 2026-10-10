"""Statistical comparison utilities for temporally dependent forecasts."""

from __future__ import annotations

from dataclasses import dataclass
from math import erfc, sqrt
from typing import Callable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class DMResult:
    """Asymptotic two-sided Diebold--Mariano test result.

    ``mean_loss_difference`` is ``loss_a - loss_b``; positive values therefore
    mean that model B has the smaller average loss.
    """

    statistic: float
    p_value: float
    mean_loss_difference: float
    hac_lags: int
    n: int


@dataclass(frozen=True)
class BootstrapCI:
    estimate: float
    lower: float
    upper: float
    confidence: float
    n_boot: int
    n_events: int


def newey_west_variance(values: np.ndarray, *, lags: int) -> float:
    """Estimate the HAC sampling variance of the one-dimensional sample mean."""

    series = np.asarray(values, dtype=np.float64).reshape(-1)
    series = series[np.isfinite(series)]
    n = series.size
    if n == 0:
        raise ValueError("values contain no finite observations")
    if lags < 0 or lags >= n:
        raise ValueError("lags must be in [0, n-1]")
    centered = series - np.mean(series)
    long_run = float(np.dot(centered, centered) / n)
    for lag in range(1, lags + 1):
        covariance = float(np.dot(centered[lag:], centered[:-lag]) / n)
        bartlett_weight = 1.0 - lag / (lags + 1.0)
        long_run += 2.0 * bartlett_weight * covariance
    return max(0.0, long_run / n)


def diebold_mariano(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    *,
    hac_lags: int,
) -> DMResult:
    """Compare paired loss sequences with a normal-reference HAC DM test."""

    a = np.asarray(loss_a, dtype=np.float64).reshape(-1)
    b = np.asarray(loss_b, dtype=np.float64).reshape(-1)
    if a.shape != b.shape:
        raise ValueError("loss_a and loss_b must have the same shape")
    finite = np.isfinite(a) & np.isfinite(b)
    difference = a[finite] - b[finite]
    if difference.size == 0:
        raise ValueError("loss pairs contain no finite observations")
    if hac_lags >= difference.size:
        raise ValueError("hac_lags must be smaller than the number of finite pairs")
    mean_difference = float(np.mean(difference))
    variance = newey_west_variance(difference, lags=hac_lags)
    if variance == 0.0:
        if mean_difference == 0.0:
            statistic, p_value = 0.0, 1.0
        else:
            statistic = float(np.copysign(np.inf, mean_difference))
            p_value = 0.0
    else:
        statistic = mean_difference / sqrt(variance)
        p_value = erfc(abs(statistic) / sqrt(2.0))
    return DMResult(
        statistic=float(statistic),
        p_value=float(p_value),
        mean_loss_difference=mean_difference,
        hac_lags=int(hac_lags),
        n=int(difference.size),
    )


def event_bootstrap_ci(
    values: np.ndarray,
    event_ids: np.ndarray,
    *,
    statistic: Callable[[np.ndarray], float] = np.mean,
    confidence: float = 0.95,
    n_boot: int = 2_000,
    seed: int = 0,
) -> BootstrapCI:
    """Bootstrap complete event clusters and return a percentile interval."""

    sample = np.asarray(values)
    ids = np.asarray(event_ids)
    if sample.ndim != 1 or ids.ndim != 1 or sample.shape != ids.shape:
        raise ValueError("values and event_ids must be one-dimensional with equal shape")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    if n_boot <= 0:
        raise ValueError("n_boot must be positive")
    finite = np.isfinite(sample.astype(np.float64, copy=False))
    sample = sample[finite]
    ids = ids[finite]
    if sample.size == 0:
        raise ValueError("values contain no finite observations")

    unique_ids = np.unique(ids)
    groups = [sample[ids == event_id] for event_id in unique_ids]
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=np.float64)
    for index in range(n_boot):
        chosen = rng.integers(0, len(groups), size=len(groups))
        resample = np.concatenate([groups[group_index] for group_index in chosen])
        draws[index] = float(statistic(resample))
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(draws, [alpha, 1.0 - alpha])
    return BootstrapCI(
        estimate=float(statistic(sample)),
        lower=float(lower),
        upper=float(upper),
        confidence=float(confidence),
        n_boot=int(n_boot),
        n_events=len(groups),
    )


def contiguous_event_ids(
    event: np.ndarray,
    timestamps: np.ndarray,
    *,
    maximum_gap: np.timedelta64,
) -> np.ndarray:
    """Assign non-negative IDs to contiguous positive event episodes.

    Non-event positions receive ``-1``. Input must already be chronological.
    """

    positive = np.asarray(event, dtype=bool).reshape(-1)
    time = np.asarray(timestamps).reshape(-1).astype("datetime64[ns]")
    if positive.shape != time.shape:
        raise ValueError("event and timestamps must have equal one-dimensional shape")
    if np.isnat(time).any() or np.any(time[1:] <= time[:-1]):
        raise ValueError("timestamps must be finite and strictly increasing")
    gap = np.asarray(maximum_gap).astype("timedelta64[ns]")
    if gap <= np.timedelta64(0, "ns"):
        raise ValueError("maximum_gap must be positive")
    identifiers = np.full(positive.shape, -1, dtype=np.int64)
    current = -1
    previous_positive_index: int | None = None
    for index in np.flatnonzero(positive):
        if (
            previous_positive_index is None
            or index != previous_positive_index + 1
            or time[index] - time[previous_positive_index] > gap
        ):
            current += 1
        identifiers[index] = current
        previous_positive_index = int(index)
    return identifiers


def _flatten_numeric(
    mapping: Mapping[str, object], prefix: str = ""
) -> dict[str, float]:
    flattened: dict[str, float] = {}
    for key, value in mapping.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flattened.update(_flatten_numeric(value, name))
        elif isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
            value, (bool, np.bool_)
        ):
            flattened[name] = float(value)
    return flattened


def aggregate_seed_metrics(
    runs: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, float | int]]:
    """Aggregate common finite numeric metrics across repeated random seeds."""

    if not runs:
        raise ValueError("runs must not be empty")
    flattened = [_flatten_numeric(run) for run in runs]
    common_keys = set(flattened[0]).intersection(*(set(run) for run in flattened[1:]))
    result: dict[str, dict[str, float | int]] = {}
    for key in sorted(common_keys):
        values = np.asarray([run[key] for run in flattened], dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        result[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "count": int(values.size),
        }
    return result
