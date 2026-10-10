"""Train-only residual scenario baselines for fair Direct-CFM comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _validate_error(
    error: np.ndarray, mask: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    values = np.asarray(error, dtype=float)
    if values.ndim < 2:
        raise ValueError("error must have [sample,...]")
    valid = np.isfinite(values)
    if mask is not None:
        if np.asarray(mask).shape != values.shape:
            raise ValueError("error and mask shapes differ")
        valid &= np.asarray(mask, dtype=bool)
    shape = values.shape[1:]
    return values.reshape(values.shape[0], -1), valid.reshape(valid.shape[0], -1), shape


def _pairwise_covariance(centered: np.ndarray, valid: np.ndarray) -> np.ndarray:
    zeroed = np.where(valid, centered, 0.0)
    cross = zeroed.T @ zeroed
    pair_count = valid.astype(np.float64).T @ valid.astype(np.float64)
    return np.divide(
        cross,
        np.maximum(pair_count - 1.0, 1.0),
        out=np.zeros_like(cross),
        where=pair_count > 1,
    )


def _regularize_covariance(
    covariance: np.ndarray, shrinkage: float, min_eigenvalue: float
) -> np.ndarray:
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("shrinkage must be in [0,1]")
    diagonal = np.diag(np.maximum(np.diag(covariance), min_eigenvalue))
    regularized = (1.0 - shrinkage) * covariance + shrinkage * diagonal
    regularized = 0.5 * (regularized + regularized.T)
    eigenvalue, eigenvector = np.linalg.eigh(regularized)
    eigenvalue = np.maximum(eigenvalue, min_eigenvalue)
    return (eigenvector * eigenvalue) @ eigenvector.T


def _normal_cdf(value: np.ndarray) -> np.ndarray:
    """Vectorized normal CDF using a stable erf approximation."""

    x = np.asarray(value, dtype=float) / np.sqrt(2.0)
    sign = np.sign(x)
    absolute = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * absolute)
    polynomial = (
        (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t)
        + 0.254829592
    ) * t
    erf = sign * (1.0 - polynomial * np.exp(-absolute * absolute))
    return np.clip(0.5 * (1.0 + erf), 1e-12, 1.0 - 1e-12)


def _normal_ppf(probability: np.ndarray) -> np.ndarray:
    """Acklam's vectorized inverse-standard-normal approximation."""

    p = np.asarray(probability, dtype=float)
    if np.any((p <= 0.0) | (p >= 1.0)):
        raise ValueError("normal PPF probabilities must be strictly between 0 and 1")
    a = np.array([-39.6968302866538, 220.946098424521, -275.928510446969,
                  138.357751867269, -30.6647980661472, 2.50662827745924])
    b = np.array([-54.4760987982241, 161.585836858041, -155.698979859887,
                  66.8013118877197, -13.2806815528857])
    c = np.array([-0.00778489400243029, -0.322396458041136, -2.40075827716184,
                  -2.54973253934373, 4.37466414146497, 2.93816398269878])
    d = np.array([0.00778469570904146, 0.32246712907004, 2.445134137143,
                  3.75440866190742])
    lower, upper = 0.02425, 1.0 - 0.02425
    result = np.empty_like(p)
    low = p < lower
    high = p > upper
    middle = ~(low | high)
    if np.any(low):
        q = np.sqrt(-2.0 * np.log(p[low]))
        result[low] = np.polyval(c, q) / np.polyval(np.r_[d, 1.0], q)
    if np.any(high):
        q = np.sqrt(-2.0 * np.log(1.0 - p[high]))
        result[high] = -np.polyval(c, q) / np.polyval(np.r_[d, 1.0], q)
    if np.any(middle):
        q = p[middle] - 0.5
        r = q * q
        result[middle] = np.polyval(a, r) * q / np.polyval(np.r_[b, 1.0], r)
    return result


@dataclass
class GaussianResidual:
    """Joint multivariate Gaussian fitted to final deterministic errors."""

    shrinkage: float = 0.10
    min_eigenvalue: float = 1e-6
    mean_: np.ndarray | None = None
    covariance_: np.ndarray | None = None
    cholesky_: np.ndarray | None = None
    event_shape_: tuple[int, ...] | None = None

    def fit(self, error: np.ndarray, mask: np.ndarray | None = None) -> "GaussianResidual":
        values, valid, shape = _validate_error(error, mask)
        count = valid.sum(axis=0)
        if np.any(count < 2):
            raise ValueError("every residual dimension needs at least two Train observations")
        self.mean_ = np.where(valid, values, 0.0).sum(axis=0) / count
        centered = values - self.mean_
        covariance = _pairwise_covariance(centered, valid)
        self.covariance_ = _regularize_covariance(
            covariance, self.shrinkage, self.min_eigenvalue
        )
        self.cholesky_ = np.linalg.cholesky(self.covariance_)
        self.event_shape_ = shape
        return self

    def sample_errors(
        self, n_scenarios: int, n_origins: int, *, seed: int | None = None
    ) -> np.ndarray:
        if self.mean_ is None or self.cholesky_ is None or self.event_shape_ is None:
            raise RuntimeError("Gaussian residual baseline is not fitted")
        rng = np.random.default_rng(seed)
        noise = rng.standard_normal((n_scenarios, n_origins, self.mean_.size))
        error = noise @ self.cholesky_.T + self.mean_
        return error.reshape(n_scenarios, n_origins, *self.event_shape_)

    def sample(
        self,
        centre: np.ndarray,
        n_scenarios: int,
        *,
        rated_power: float,
        seed: int | None = None,
    ) -> np.ndarray:
        centre = np.asarray(centre, dtype=float)
        error = self.sample_errors(n_scenarios, centre.shape[0], seed=seed)
        if error.shape[1:] != centre.shape:
            raise ValueError("centre shape differs from fitted residual event shape")
        return np.clip(centre[None] + error, 0.0, rated_power)


@dataclass
class ResidualBootstrap:
    """Resample whole historical error fields to preserve observed dependence."""

    minimum_valid_fraction: float = 0.95
    residuals_: np.ndarray | None = None
    event_shape_: tuple[int, ...] | None = None

    def fit(self, error: np.ndarray, mask: np.ndarray | None = None) -> "ResidualBootstrap":
        values, valid, shape = _validate_error(error, mask)
        if not 0.0 < self.minimum_valid_fraction <= 1.0:
            raise ValueError("minimum_valid_fraction must be in (0,1]")
        count = valid.sum(axis=0)
        if np.any(count == 0):
            raise ValueError("a residual dimension has no Train observations")
        mean = np.where(valid, values, 0.0).sum(axis=0) / count
        filled = np.where(valid, values, mean)
        keep = valid.mean(axis=1) >= self.minimum_valid_fraction
        if not np.any(keep):
            raise ValueError("no Train residual field passes minimum_valid_fraction")
        self.residuals_ = filled[keep]
        self.event_shape_ = shape
        return self

    def sample(
        self,
        centre: np.ndarray,
        n_scenarios: int,
        *,
        rated_power: float,
        seed: int | None = None,
    ) -> np.ndarray:
        if self.residuals_ is None or self.event_shape_ is None:
            raise RuntimeError("bootstrap baseline is not fitted")
        centre = np.asarray(centre, dtype=float)
        if centre.shape[1:] != self.event_shape_:
            raise ValueError("centre event shape differs from fitted residual shape")
        rng = np.random.default_rng(seed)
        index = rng.integers(0, len(self.residuals_), size=(n_scenarios, centre.shape[0]))
        error = self.residuals_[index].reshape(n_scenarios, centre.shape[0], *self.event_shape_)
        return np.clip(centre[None] + error, 0.0, rated_power)


@dataclass
class GaussianCopulaResidual:
    """Gaussian copula with empirical per-dimension residual marginals."""

    shrinkage: float = 0.10
    min_eigenvalue: float = 1e-6
    marginals_: list[np.ndarray] | None = None
    correlation_: np.ndarray | None = None
    cholesky_: np.ndarray | None = None
    event_shape_: tuple[int, ...] | None = None

    def fit(self, error: np.ndarray, mask: np.ndarray | None = None) -> "GaussianCopulaResidual":
        values, valid, shape = _validate_error(error, mask)
        latent = np.zeros_like(values)
        marginals: list[np.ndarray] = []
        for dimension in range(values.shape[1]):
            selected = values[valid[:, dimension], dimension]
            if selected.size < 3:
                raise ValueError("each copula marginal needs at least three Train observations")
            order = np.argsort(selected, kind="stable")
            rank = np.empty(selected.size, dtype=float)
            rank[order] = np.arange(1, selected.size + 1)
            probability = (rank - 0.5) / selected.size
            latent[valid[:, dimension], dimension] = _normal_ppf(probability)
            marginals.append(np.sort(selected))
        correlation = _pairwise_covariance(latent, valid)
        standard = np.sqrt(np.maximum(np.diag(correlation), self.min_eigenvalue))
        correlation = correlation / np.outer(standard, standard)
        np.fill_diagonal(correlation, 1.0)
        self.correlation_ = _regularize_covariance(
            correlation, self.shrinkage, self.min_eigenvalue
        )
        diagonal = np.sqrt(np.diag(self.correlation_))
        self.correlation_ /= np.outer(diagonal, diagonal)
        np.fill_diagonal(self.correlation_, 1.0)
        self.cholesky_ = np.linalg.cholesky(self.correlation_)
        self.marginals_ = marginals
        self.event_shape_ = shape
        return self

    def sample_errors(
        self, n_scenarios: int, n_origins: int, *, seed: int | None = None
    ) -> np.ndarray:
        if self.cholesky_ is None or self.marginals_ is None or self.event_shape_ is None:
            raise RuntimeError("Gaussian copula baseline is not fitted")
        rng = np.random.default_rng(seed)
        latent = rng.standard_normal((n_scenarios, n_origins, len(self.marginals_)))
        latent = latent @ self.cholesky_.T
        uniform = _normal_cdf(latent)
        result = np.empty_like(uniform)
        for dimension, marginal in enumerate(self.marginals_):
            result[..., dimension] = np.quantile(
                marginal, uniform[..., dimension], method="linear"
            )
        return result.reshape(n_scenarios, n_origins, *self.event_shape_)

    def sample(
        self,
        centre: np.ndarray,
        n_scenarios: int,
        *,
        rated_power: float,
        seed: int | None = None,
    ) -> np.ndarray:
        centre = np.asarray(centre, dtype=float)
        error = self.sample_errors(n_scenarios, centre.shape[0], seed=seed)
        if error.shape[1:] != centre.shape:
            raise ValueError("centre shape differs from fitted copula residual shape")
        return np.clip(centre[None] + error, 0.0, rated_power)


ProbabilisticResidualModel = GaussianResidual | ResidualBootstrap | GaussianCopulaResidual


def save_probabilistic_baseline(
    model: ProbabilisticResidualModel, path: str | Path
) -> None:
    """Save a fitted baseline as a pickle-free compressed NumPy archive."""

    destination = Path(path)
    if isinstance(model, GaussianResidual):
        if any(
            value is None
            for value in (
                model.mean_,
                model.covariance_,
                model.cholesky_,
                model.event_shape_,
            )
        ):
            raise RuntimeError("Gaussian residual baseline is not fitted")
        np.savez_compressed(
            destination,
            method=np.asarray("gaussian_residual"),
            event_shape=np.asarray(model.event_shape_, dtype=np.int64),
            shrinkage=np.asarray(model.shrinkage),
            min_eigenvalue=np.asarray(model.min_eigenvalue),
            mean=model.mean_,
            covariance=model.covariance_,
            cholesky=model.cholesky_,
        )
        return
    if isinstance(model, ResidualBootstrap):
        if model.residuals_ is None or model.event_shape_ is None:
            raise RuntimeError("bootstrap baseline is not fitted")
        np.savez_compressed(
            destination,
            method=np.asarray("residual_bootstrap"),
            event_shape=np.asarray(model.event_shape_, dtype=np.int64),
            minimum_valid_fraction=np.asarray(model.minimum_valid_fraction),
            residuals=model.residuals_,
        )
        return
    if isinstance(model, GaussianCopulaResidual):
        if any(
            value is None
            for value in (
                model.marginals_,
                model.correlation_,
                model.cholesky_,
                model.event_shape_,
            )
        ):
            raise RuntimeError("Gaussian copula baseline is not fitted")
        lengths = np.asarray([len(values) for values in model.marginals_], dtype=np.int64)
        offsets = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(lengths)])
        marginal_values = np.concatenate(model.marginals_)
        np.savez_compressed(
            destination,
            method=np.asarray("gaussian_copula"),
            event_shape=np.asarray(model.event_shape_, dtype=np.int64),
            shrinkage=np.asarray(model.shrinkage),
            min_eigenvalue=np.asarray(model.min_eigenvalue),
            correlation=model.correlation_,
            cholesky=model.cholesky_,
            marginal_offsets=offsets,
            marginal_values=marginal_values,
        )
        return
    raise TypeError(f"unsupported baseline type: {type(model)!r}")


def load_probabilistic_baseline(path: str | Path) -> ProbabilisticResidualModel:
    """Restore a baseline written by :func:`save_probabilistic_baseline`."""

    with np.load(Path(path), allow_pickle=False) as archive:
        method = str(archive["method"].item())
        event_shape = tuple(int(value) for value in archive["event_shape"])
        if method == "gaussian_residual":
            model = GaussianResidual(
                shrinkage=float(archive["shrinkage"]),
                min_eigenvalue=float(archive["min_eigenvalue"]),
            )
            model.mean_ = archive["mean"].copy()
            model.covariance_ = archive["covariance"].copy()
            model.cholesky_ = archive["cholesky"].copy()
        elif method == "residual_bootstrap":
            model = ResidualBootstrap(
                minimum_valid_fraction=float(archive["minimum_valid_fraction"])
            )
            model.residuals_ = archive["residuals"].copy()
        elif method == "gaussian_copula":
            model = GaussianCopulaResidual(
                shrinkage=float(archive["shrinkage"]),
                min_eigenvalue=float(archive["min_eigenvalue"]),
            )
            offsets = archive["marginal_offsets"]
            values = archive["marginal_values"]
            model.marginals_ = [
                values[offsets[index] : offsets[index + 1]].copy()
                for index in range(len(offsets) - 1)
            ]
            model.correlation_ = archive["correlation"].copy()
            model.cholesky_ = archive["cholesky"].copy()
        else:
            raise ValueError(f"unknown probabilistic baseline method: {method}")
    model.event_shape_ = event_shape
    return model
