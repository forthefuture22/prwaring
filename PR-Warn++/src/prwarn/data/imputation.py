"""Causal, mask-preserving history imputation baselines."""

from __future__ import annotations

import numpy as np


IMPUTATION_STRATEGIES = ("train_mean", "forward_fill", "causal_linear", "grud_decay")


def impute_history(
    values: np.ndarray,
    mask: np.ndarray,
    train_mean: np.ndarray,
    *,
    strategy: str,
    decay_rate: float = 0.1,
) -> np.ndarray:
    """Fill histories without changing the observation mask or reading ahead.

    Inputs may be ``[T,N,F]`` or ``[B,T,N,F]``. ``causal_linear`` extrapolates
    from the two latest observations; it deliberately does not interpolate
    from a later timestamp.
    """

    if strategy not in IMPUTATION_STRATEGIES:
        raise ValueError(f"unknown imputation strategy: {strategy}")
    if decay_rate < 0:
        raise ValueError("decay_rate must be non-negative")
    source = np.asarray(values, dtype=float)
    observed = np.asarray(mask, dtype=bool) & np.isfinite(source)
    if source.shape != observed.shape or source.ndim not in {3, 4}:
        raise ValueError("values/mask must match [T,N,F] or [B,T,N,F]")
    squeeze = source.ndim == 3
    if squeeze:
        source, observed = source[None], observed[None]
    mean = np.asarray(train_mean, dtype=float)
    try:
        mean = np.broadcast_to(mean, source.shape[2:])
    except ValueError as exc:
        raise ValueError("train_mean must broadcast to [N,F]") from exc
    output = source.copy()
    for batch in range(source.shape[0]):
        previous = mean.copy()
        previous_time = np.full(mean.shape, -1, dtype=np.int64)
        penultimate = mean.copy()
        penultimate_time = np.full(mean.shape, -1, dtype=np.int64)
        for time in range(source.shape[1]):
            current_observed = observed[batch, time]
            missing = ~current_observed
            if strategy == "train_mean":
                fill = mean
            elif strategy == "forward_fill":
                fill = previous
            elif strategy == "grud_decay":
                gap = np.maximum(time - previous_time, 1)
                gamma = np.exp(-decay_rate * gap)
                fill = gamma * previous + (1.0 - gamma) * mean
            else:
                interval = previous_time - penultimate_time
                slope = np.divide(
                    previous - penultimate,
                    interval,
                    out=np.zeros_like(previous),
                    where=interval > 0,
                )
                fill = previous + slope * np.maximum(time - previous_time, 1)
            output[batch, time][missing] = fill[missing]
            penultimate[current_observed] = previous[current_observed]
            penultimate_time[current_observed] = previous_time[current_observed]
            previous[current_observed] = source[batch, time][current_observed]
            previous_time[current_observed] = time
    return output[0] if squeeze else output
