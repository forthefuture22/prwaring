"""Per-horizon split conformal calibration for scenario-derived intervals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _finite_sample_quantile(scores: np.ndarray, alpha: float) -> float:
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        raise ValueError("no finite calibration scores")
    level = min(1.0, np.ceil((scores.size + 1) * (1.0 - alpha)) / scores.size)
    return float(np.quantile(scores, level, method="higher"))


@dataclass
class PerHorizonSplitConformal:
    """Conformalize lower/upper `[sample,horizon]` interval endpoints."""

    alpha: float = 0.10
    correction_: np.ndarray | None = None

    def fit(
        self,
        lower: np.ndarray,
        upper: np.ndarray,
        target: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> "PerHorizonSplitConformal":
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)
        target = np.asarray(target, dtype=float)
        if lower.shape != upper.shape or lower.shape != target.shape or lower.ndim != 2:
            raise ValueError("lower, upper and target must share [sample,horizon]")
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must be between 0 and 1")
        valid = np.isfinite(lower) & np.isfinite(upper) & np.isfinite(target)
        if mask is not None:
            valid &= np.asarray(mask, dtype=bool)
        scores = np.maximum.reduce([lower - target, target - upper, np.zeros_like(target)])
        self.correction_ = np.asarray(
            [_finite_sample_quantile(scores[valid[:, h], h], self.alpha) for h in range(target.shape[1])]
        )
        return self

    def transform(self, lower: np.ndarray, upper: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.correction_ is None:
            raise RuntimeError("conformalizer is not fitted")
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)
        if lower.shape != upper.shape or lower.shape[-1] != self.correction_.shape[0]:
            raise ValueError("interval shape does not match fitted horizons")
        return lower - self.correction_, upper + self.correction_

    def fit_transform(
        self,
        lower: np.ndarray,
        upper: np.ndarray,
        target: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.fit(lower, upper, target, mask).transform(lower, upper)


def scenario_interval(
    scenarios: np.ndarray,
    *,
    alpha: float = 0.10,
    scenario_axis: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract a central interval from Monte Carlo scenarios."""

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1")
    values = np.asarray(scenarios, dtype=float)
    return (
        np.quantile(values, alpha / 2.0, axis=scenario_axis),
        np.quantile(values, 1.0 - alpha / 2.0, axis=scenario_axis),
    )
