"""Dependency-light metrics used by the first reproducible pipeline."""

from __future__ import annotations

import numpy as np


def _valid_arrays(
    truth: np.ndarray, prediction: np.ndarray, mask: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    truth = np.asarray(truth, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if truth.shape != prediction.shape:
        raise ValueError("truth and prediction shapes differ")
    valid = np.isfinite(truth) & np.isfinite(prediction)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    if not valid.any():
        raise ValueError("no valid observations")
    return truth[valid], prediction[valid]


def masked_mae(truth: np.ndarray, prediction: np.ndarray, mask: np.ndarray | None = None) -> float:
    y, yhat = _valid_arrays(truth, prediction, mask)
    return float(np.mean(np.abs(y - yhat)))


def masked_rmse(truth: np.ndarray, prediction: np.ndarray, mask: np.ndarray | None = None) -> float:
    y, yhat = _valid_arrays(truth, prediction, mask)
    return float(np.sqrt(np.mean(np.square(y - yhat))))


def masked_r2(truth: np.ndarray, prediction: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Masked coefficient of determination (identical to NSE on one sample set)."""

    y, yhat = _valid_arrays(truth, prediction, mask)
    denominator = np.square(y - y.mean()).sum()
    if denominator == 0:
        return float("nan")
    return float(1.0 - np.square(y - yhat).sum() / denominator)


def pinball_loss(
    truth: np.ndarray,
    prediction: np.ndarray,
    quantile: float,
    mask: np.ndarray | None = None,
) -> float:
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be in (0,1)")
    y, q = _valid_arrays(truth, prediction, mask)
    difference = y - q
    return float(np.mean(np.maximum(quantile * difference, (quantile - 1.0) * difference)))


def crps_ensemble(
    truth: np.ndarray,
    scenarios: np.ndarray,
    *,
    scenario_axis: int = 0,
    mask: np.ndarray | None = None,
) -> float:
    """Empirical CRPS: E|X-y| - 0.5 E|X-X'|."""

    score = crps_ensemble_values(truth, scenarios, scenario_axis=scenario_axis)
    y = np.asarray(truth, dtype=float)
    valid = np.isfinite(score) & np.isfinite(y)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    if not valid.any():
        raise ValueError("no valid CRPS observations")
    return float(score[valid].mean())


def crps_ensemble_values(
    truth: np.ndarray,
    scenarios: np.ndarray,
    *,
    scenario_axis: int = 0,
) -> np.ndarray:
    """Return unreduced empirical CRPS with the same shape as ``truth``."""

    samples = np.moveaxis(np.asarray(scenarios, dtype=float), scenario_axis, 0)
    y = np.asarray(truth, dtype=float)
    if samples.shape[1:] != y.shape:
        raise ValueError("scenario and truth shapes differ")
    first = np.mean(np.abs(samples - y[None]), axis=0)
    # Exact O(M log M) form of 0.5 E|X-X'|; avoids an O(M^2) tensor.
    ordered = np.sort(samples, axis=0)
    m = samples.shape[0]
    coefficient = (2 * np.arange(1, m + 1) - m - 1).reshape(
        (m,) + (1,) * (samples.ndim - 1)
    )
    half_pairwise = np.sum(coefficient * ordered, axis=0) / (m * m)
    return first - half_pairwise


def energy_score(truth: np.ndarray, scenarios: np.ndarray) -> float:
    """Mean multivariate Energy Score for `[M,B,...]` scenarios."""

    return float(np.mean(energy_score_values(truth, scenarios)))


def energy_score_values(truth: np.ndarray, scenarios: np.ndarray) -> np.ndarray:
    """Return one multivariate Energy Score per forecast origin."""

    samples = np.asarray(scenarios, dtype=float)
    y = np.asarray(truth, dtype=float)
    if samples.ndim < 3 or samples.shape[1:] != y.shape:
        raise ValueError("expected scenarios [M,B,...] and truth [B,...]")
    flat_samples = samples.reshape(samples.shape[0], samples.shape[1], -1)
    flat_truth = y.reshape(y.shape[0], -1)
    first = np.linalg.norm(flat_samples - flat_truth[None], axis=-1).mean(axis=0)
    difference = flat_samples[:, None] - flat_samples[None, :]
    second = np.linalg.norm(difference, axis=-1).mean(axis=(0, 1))
    return first - 0.5 * second


def variogram_score(
    truth: np.ndarray,
    scenarios: np.ndarray,
    *,
    order: float = 0.5,
) -> float:
    """Unweighted Variogram Score over flattened turbine/horizon variables."""

    return float(np.mean(variogram_score_values(truth, scenarios, order=order)))


def variogram_score_values(
    truth: np.ndarray,
    scenarios: np.ndarray,
    *,
    order: float = 0.5,
) -> np.ndarray:
    """Return one unweighted Variogram Score per forecast origin."""

    samples = np.asarray(scenarios, dtype=float)
    y = np.asarray(truth, dtype=float)
    if samples.shape[1:] != y.shape or y.ndim < 2:
        raise ValueError("shape mismatch")
    sf = samples.reshape(samples.shape[0], samples.shape[1], -1)
    yf = y.reshape(y.shape[0], -1)
    truth_variogram = np.abs(yf[:, :, None] - yf[:, None, :]) ** order
    scenario_variogram = np.mean(
        np.abs(sf[:, :, :, None] - sf[:, :, None, :]) ** order, axis=0
    )
    return np.mean(
        np.square(truth_variogram - scenario_variogram), axis=(1, 2)
    )


def brier_score(event: np.ndarray, probability: np.ndarray) -> float:
    event = np.asarray(event, dtype=float)
    probability = np.asarray(probability, dtype=float)
    if event.shape != probability.shape:
        raise ValueError("event and probability shapes differ")
    valid = np.isfinite(event) & np.isfinite(probability)
    return float(np.mean(np.square(probability[valid] - event[valid])))


def brier_skill_score(
    event: np.ndarray, probability: np.ndarray, reference_probability: np.ndarray
) -> float:
    score = brier_score(event, probability)
    reference = brier_score(event, reference_probability)
    return float(1.0 - score / reference) if reference > 0 else float("nan")


def binary_event_metrics(
    event: np.ndarray, probability: np.ndarray, *, threshold: float = 0.5
) -> dict[str, float | int]:
    y = np.asarray(event).reshape(-1)
    p = np.asarray(probability, dtype=float).reshape(-1)
    valid = np.isfinite(y) & np.isfinite(p)
    y = y[valid].astype(bool)
    predicted = p[valid] >= threshold
    tp = int(np.sum(y & predicted))
    fp = int(np.sum(~y & predicted))
    fn = int(np.sum(y & ~predicted))
    tn = int(np.sum(~y & ~predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": float(precision), "recall": float(recall),
        "f1": float(f1), "fnr": float(1.0 - recall) if np.isfinite(recall) else float("nan"),
    }


def ks_distance(first: np.ndarray, second: np.ndarray) -> float:
    """Dependency-free two-sample Kolmogorov--Smirnov distance."""

    a = np.sort(np.asarray(first, dtype=float).reshape(-1))
    b = np.sort(np.asarray(second, dtype=float).reshape(-1))
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        raise ValueError("KS inputs need finite observations")
    values = np.sort(np.unique(np.concatenate([a, b])))
    cdf_a = np.searchsorted(a, values, side="right") / a.size
    cdf_b = np.searchsorted(b, values, side="right") / b.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


def dependence_diagnostics(truth: np.ndarray, scenarios: np.ndarray) -> dict[str, float]:
    """Compare flattened variable correlation matrices on complete origins."""

    y = np.asarray(truth, dtype=float)
    samples = np.asarray(scenarios, dtype=float)
    if samples.shape[1:] != y.shape or y.ndim < 2:
        raise ValueError("expected scenarios [M,B,...] matching truth [B,...]")
    truth_flat = y.reshape(y.shape[0], -1)
    scenario_flat = samples.transpose(1, 0, *range(2, samples.ndim)).reshape(
        y.shape[0] * samples.shape[0], -1
    )
    truth_corr = np.nan_to_num(np.corrcoef(truth_flat, rowvar=False))
    scenario_corr = np.nan_to_num(np.corrcoef(scenario_flat, rowvar=False))
    difference = scenario_corr - truth_corr
    return {
        "correlation_mae": float(np.mean(np.abs(difference))),
        "correlation_rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "correlation_frobenius_normalized": float(
            np.linalg.norm(difference) / max(1, difference.shape[0])
        ),
    }


def average_precision(event: np.ndarray, probability: np.ndarray) -> float:
    """Average precision without a scikit-learn dependency."""

    y = np.asarray(event).reshape(-1)
    p = np.asarray(probability, dtype=float).reshape(-1)
    valid = np.isfinite(y) & np.isfinite(p)
    y, p = y[valid].astype(bool), p[valid]
    positives = int(y.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-p, kind="stable")
    ranked = y[order]
    precision = np.cumsum(ranked) / np.arange(1, ranked.size + 1)
    return float(precision[ranked].sum() / positives)


def reliability_bins(
    event: np.ndarray, probability: np.ndarray, *, n_bins: int = 10
) -> dict[str, np.ndarray]:
    y = np.asarray(event, dtype=float).reshape(-1)
    p = np.asarray(probability, dtype=float).reshape(-1)
    valid = np.isfinite(y) & np.isfinite(p)
    y, p = y[valid], np.clip(p[valid], 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_id = np.minimum(np.digitize(p, edges[1:-1]), n_bins - 1)
    count = np.bincount(bin_id, minlength=n_bins)
    predicted = np.full(n_bins, np.nan)
    observed = np.full(n_bins, np.nan)
    for i in range(n_bins):
        selected = bin_id == i
        if selected.any():
            predicted[i] = p[selected].mean()
            observed[i] = y[selected].mean()
    return {"edges": edges, "count": count, "predicted": predicted, "observed": observed}


def interval_metrics(
    truth: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    alpha: float,
    mask: np.ndarray | None = None,
    normalization: float | None = None,
) -> dict[str, float]:
    """PICP, mean/PINAW width and Winkler interval score."""

    y = np.asarray(truth, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    if y.shape != lo.shape or y.shape != hi.shape:
        raise ValueError("truth/lower/upper shapes differ")
    if np.any(hi < lo) or not 0.0 < alpha < 1.0:
        raise ValueError("invalid interval or alpha")
    valid = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    if not valid.any():
        raise ValueError("no valid intervals")
    width = hi - lo
    covered = (y >= lo) & (y <= hi)
    winkler = width.copy()
    below = y < lo
    above = y > hi
    winkler[below] += (2.0 / alpha) * (lo[below] - y[below])
    winkler[above] += (2.0 / alpha) * (y[above] - hi[above])
    mean_width = float(width[valid].mean())
    result = {
        "picp": float(covered[valid].mean()),
        "mean_width": mean_width,
        "winkler": float(winkler[valid].mean()),
        "count": int(valid.sum()),
    }
    if normalization is not None:
        if normalization <= 0:
            raise ValueError("normalization must be positive")
        result["pinaw"] = mean_width / normalization
    return result
