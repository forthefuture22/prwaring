"""Chronological adaptive and context-aware empirical interval calibration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .split import _finite_sample_quantile


def _scores(lower: np.ndarray, upper: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.maximum.reduce([lower - target, target - upper, np.zeros_like(target)])


@dataclass(frozen=True)
class AdaptiveCalibrationResult:
    lower: np.ndarray
    upper: np.ndarray
    correction: np.ndarray
    alpha_before_update: np.ndarray
    miss: np.ndarray
    valid: np.ndarray


class AdaptiveConformalIntervals:
    """Per-horizon ACI with strict predict-then-update chronological ordering."""

    def __init__(
        self,
        *,
        alpha: float = 0.10,
        gamma: float = 0.01,
        rolling_window: int | None = None,
        min_alpha: float = 1e-3,
        max_alpha: float = 0.999,
    ) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        if gamma <= 0.0:
            raise ValueError("gamma must be positive")
        if rolling_window is not None and rolling_window <= 0:
            raise ValueError("rolling_window must be positive")
        if not 0.0 < min_alpha <= alpha <= max_alpha < 1.0:
            raise ValueError("alpha bounds must contain alpha and lie in (0, 1)")
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.rolling_window = rolling_window
        self.min_alpha = float(min_alpha)
        self.max_alpha = float(max_alpha)
        self.score_history_: list[list[float]] | None = None
        self.alpha_state_: np.ndarray | None = None

    def fit(
        self,
        lower: np.ndarray,
        upper: np.ndarray,
        target: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> "AdaptiveConformalIntervals":
        lo, hi, y = map(lambda value: np.asarray(value, dtype=float), (lower, upper, target))
        if lo.ndim != 2 or lo.shape != hi.shape or lo.shape != y.shape:
            raise ValueError("lower, upper and target must share [sample,horizon]")
        valid = np.isfinite(lo) & np.isfinite(hi) & np.isfinite(y)
        if mask is not None:
            if np.asarray(mask).shape != y.shape:
                raise ValueError("mask shape differs from target")
            valid &= np.asarray(mask, dtype=bool)
        score = _scores(lo, hi, y)
        self.score_history_ = []
        for horizon in range(y.shape[1]):
            values = score[valid[:, horizon], horizon]
            if values.size == 0:
                raise ValueError(f"horizon {horizon} has no valid calibration scores")
            self.score_history_.append(values.astype(float).tolist())
        self.alpha_state_ = np.full(y.shape[1], self.alpha, dtype=float)
        return self

    def _correction(self) -> np.ndarray:
        if self.score_history_ is None or self.alpha_state_ is None:
            raise RuntimeError("adaptive conformalizer is not fitted")
        corrections = []
        for horizon, history in enumerate(self.score_history_):
            selected_history = (
                history if self.rolling_window is None else history[-self.rolling_window :]
            )
            values = np.asarray(selected_history, dtype=float)
            corrections.append(
                _finite_sample_quantile(values, float(self.alpha_state_[horizon]))
            )
        return np.asarray(corrections)

    def transform_stream(
        self,
        lower: np.ndarray,
        upper: np.ndarray,
        target: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> AdaptiveCalibrationResult:
        """Calibrate each origin, then reveal that origin's target and update."""

        if self.score_history_ is None or self.alpha_state_ is None:
            raise RuntimeError("adaptive conformalizer is not fitted")
        lo, hi, y = map(lambda value: np.asarray(value, dtype=float), (lower, upper, target))
        if lo.ndim != 2 or lo.shape != hi.shape or lo.shape != y.shape:
            raise ValueError("lower, upper and target must share [sample,horizon]")
        if y.shape[1] != len(self.score_history_):
            raise ValueError("horizon count differs from fitted calibration data")
        valid = np.isfinite(lo) & np.isfinite(hi) & np.isfinite(y)
        if mask is not None:
            if np.asarray(mask).shape != y.shape:
                raise ValueError("mask shape differs from target")
            valid &= np.asarray(mask, dtype=bool)

        calibrated_lo = np.empty_like(lo)
        calibrated_hi = np.empty_like(hi)
        corrections = np.empty_like(lo)
        alpha_trace = np.empty_like(lo)
        misses = np.zeros_like(valid)
        raw_score = _scores(lo, hi, y)
        for index in range(y.shape[0]):
            correction = self._correction()
            corrections[index] = correction
            alpha_trace[index] = self.alpha_state_
            calibrated_lo[index] = lo[index] - correction
            calibrated_hi[index] = hi[index] + correction
            misses[index] = valid[index] & (
                (y[index] < calibrated_lo[index]) | (y[index] > calibrated_hi[index])
            )
            update = self.gamma * (self.alpha - misses[index].astype(float))
            self.alpha_state_[valid[index]] = np.clip(
                self.alpha_state_[valid[index]] + update[valid[index]],
                self.min_alpha,
                self.max_alpha,
            )
            for horizon in np.flatnonzero(valid[index]):
                self.score_history_[horizon].append(float(raw_score[index, horizon]))
        return AdaptiveCalibrationResult(
            lower=calibrated_lo,
            upper=calibrated_hi,
            correction=corrections,
            alpha_before_update=alpha_trace,
            miss=misses,
            valid=valid,
        )


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, level: float) -> float:
    order = np.argsort(values, kind="stable")
    ordered_values = values[order]
    ordered_weights = weights[order]
    cumulative = np.cumsum(ordered_weights)
    if cumulative[-1] <= 0:
        raise ValueError("weighted quantile has no positive weight")
    index = int(np.searchsorted(cumulative, level * cumulative[-1], side="left"))
    return float(ordered_values[min(index, len(ordered_values) - 1)])


@dataclass(frozen=True)
class ContextCalibrationResult:
    lower: np.ndarray
    upper: np.ndarray
    correction: np.ndarray
    fallback_triggered: np.ndarray
    nearest_distance: np.ndarray


class ContextFallbackConformal:
    """Empirical context-kernel correction with conservative OOD fallback.

    This is an ablation mechanism, not a distribution-free conditional
    coverage guarantee. Contexts must be available at forecast issue time.
    """

    def __init__(
        self,
        *,
        alpha: float = 0.10,
        neighbors: int = 100,
        bandwidth: float | None = None,
        ood_quantile: float = 0.95,
    ) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        if neighbors <= 0:
            raise ValueError("neighbors must be positive")
        if bandwidth is not None and bandwidth <= 0:
            raise ValueError("bandwidth must be positive")
        if not 0.0 < ood_quantile <= 1.0:
            raise ValueError("ood_quantile must be in (0, 1]")
        self.alpha = float(alpha)
        self.neighbors = int(neighbors)
        self.bandwidth = bandwidth
        self.ood_quantile = float(ood_quantile)
        self.context_: np.ndarray | None = None
        self.context_mean_: np.ndarray | None = None
        self.context_scale_: np.ndarray | None = None
        self.scores_: np.ndarray | None = None
        self.valid_: np.ndarray | None = None
        self.fallback_: np.ndarray | None = None
        self.ood_threshold_: float | None = None

    def fit(
        self,
        lower: np.ndarray,
        upper: np.ndarray,
        target: np.ndarray,
        context: np.ndarray,
        mask: np.ndarray | None = None,
        subgroup: np.ndarray | None = None,
    ) -> "ContextFallbackConformal":
        lo, hi, y = map(lambda value: np.asarray(value, dtype=float), (lower, upper, target))
        ctx = np.asarray(context, dtype=float)
        if lo.ndim != 2 or lo.shape != hi.shape or lo.shape != y.shape:
            raise ValueError("lower, upper and target must share [sample,horizon]")
        if ctx.ndim != 2 or ctx.shape[0] != y.shape[0] or not np.isfinite(ctx).all():
            raise ValueError("finite context must have shape [sample,context_feature]")
        valid = np.isfinite(lo) & np.isfinite(hi) & np.isfinite(y)
        if mask is not None:
            if np.asarray(mask).shape != y.shape:
                raise ValueError("mask shape differs from target")
            valid &= np.asarray(mask, dtype=bool)
        self.context_mean_ = ctx.mean(axis=0)
        self.context_scale_ = np.maximum(ctx.std(axis=0), 1e-8)
        self.context_ = (ctx - self.context_mean_) / self.context_scale_
        self.scores_ = _scores(lo, hi, y)
        self.valid_ = valid

        groups = np.zeros(y.shape[0], dtype=int) if subgroup is None else np.asarray(subgroup)
        if groups.ndim != 1 or groups.shape[0] != y.shape[0]:
            raise ValueError("subgroup must have shape [sample]")
        group_corrections = []
        for group in np.unique(groups):
            selected_group = groups == group
            per_horizon = []
            for horizon in range(y.shape[1]):
                selected = selected_group & valid[:, horizon]
                if selected.any():
                    per_horizon.append(
                        _finite_sample_quantile(
                            self.scores_[selected, horizon], self.alpha
                        )
                    )
                else:
                    per_horizon.append(np.nan)
            group_corrections.append(per_horizon)
        correction_array = np.asarray(group_corrections, dtype=float)
        if np.any(np.all(~np.isfinite(correction_array), axis=0)):
            raise ValueError("a horizon has no valid calibration scores")
        self.fallback_ = np.nanmax(correction_array, axis=0)

        if len(ctx) < 2:
            self.ood_threshold_ = float("inf")
        else:
            difference = self.context_[:, None, :] - self.context_[None, :, :]
            distances = np.linalg.norm(difference, axis=-1)
            np.fill_diagonal(distances, np.inf)
            nearest = distances.min(axis=1)
            self.ood_threshold_ = float(np.quantile(nearest, self.ood_quantile))
        return self

    def transform(
        self,
        lower: np.ndarray,
        upper: np.ndarray,
        context: np.ndarray,
    ) -> ContextCalibrationResult:
        if any(
            value is None
            for value in (
                self.context_,
                self.context_mean_,
                self.context_scale_,
                self.scores_,
                self.valid_,
                self.fallback_,
                self.ood_threshold_,
            )
        ):
            raise RuntimeError("context conformalizer is not fitted")
        lo, hi = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
        ctx = np.asarray(context, dtype=float)
        if lo.ndim != 2 or lo.shape != hi.shape or ctx.ndim != 2:
            raise ValueError("intervals and context must be two-dimensional")
        if ctx.shape[0] != lo.shape[0] or ctx.shape[1] != self.context_.shape[1]:
            raise ValueError("context shape differs from fitted calibration context")
        if lo.shape[1] != self.scores_.shape[1] or not np.isfinite(ctx).all():
            raise ValueError("interval horizon mismatch or non-finite context")
        standardized = (ctx - self.context_mean_) / self.context_scale_
        distance = np.linalg.norm(
            standardized[:, None, :] - self.context_[None, :, :], axis=-1
        )
        nearest_distance = distance.min(axis=1)
        fallback = nearest_distance > self.ood_threshold_
        correction = np.empty_like(lo)
        for sample_index in range(lo.shape[0]):
            for horizon in range(lo.shape[1]):
                if fallback[sample_index]:
                    correction[sample_index, horizon] = self.fallback_[horizon]
                    continue
                valid_index = np.flatnonzero(self.valid_[:, horizon])
                ordered = valid_index[
                    np.argsort(distance[sample_index, valid_index], kind="stable")
                ][: self.neighbors]
                selected_distance = distance[sample_index, ordered]
                if self.bandwidth is None:
                    positive = selected_distance[selected_distance > 0]
                    bandwidth = float(np.median(positive)) if positive.size else 1.0
                else:
                    bandwidth = float(self.bandwidth)
                weights = np.exp(-0.5 * np.square(selected_distance / max(bandwidth, 1e-8)))
                correction[sample_index, horizon] = _weighted_quantile(
                    self.scores_[ordered, horizon], weights, 1.0 - self.alpha
                )
        return ContextCalibrationResult(
            lower=lo - correction,
            upper=hi + correction,
            correction=correction,
            fallback_triggered=fallback,
            nearest_distance=nearest_distance,
        )
