"""Explicit graph builders with train-only/statistical boundaries."""

from __future__ import annotations

import numpy as np


def _row_normalize(adjacency: np.ndarray) -> np.ndarray:
    adjacency = np.asarray(adjacency, dtype=np.float32)
    denominator = adjacency.sum(axis=-1, keepdims=True)
    return np.divide(adjacency, denominator, out=np.zeros_like(adjacency), where=denominator > 0)


def _top_k(adjacency: np.ndarray, k: int | None) -> np.ndarray:
    if k is None or k >= adjacency.shape[-1]:
        return adjacency
    if k <= 0:
        raise ValueError("k must be positive")
    result = np.zeros_like(adjacency)
    indices = np.argpartition(adjacency, -k, axis=-1)[..., -k:]
    np.put_along_axis(result, indices, np.take_along_axis(adjacency, indices, axis=-1), axis=-1)
    return result


def geographic_graph(
    coordinates: np.ndarray,
    *,
    k: int = 8,
    distance_scale: float | None = None,
    include_self: bool = True,
    normalize: bool = True,
) -> np.ndarray:
    """Build a Gaussian-distance k-nearest-neighbour graph."""

    xy = np.asarray(coordinates, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("coordinates must have shape [node,2]")
    distance = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1)
    nonzero = distance[distance > 0]
    scale = float(np.median(nonzero)) if distance_scale is None else float(distance_scale)
    if scale <= 0:
        raise ValueError("distance_scale must be positive")
    adjacency = np.exp(-np.square(distance / scale))
    if not include_self:
        np.fill_diagonal(adjacency, 0.0)
    adjacency = _top_k(adjacency, min(k + int(include_self), len(xy)))
    return _row_normalize(adjacency) if normalize else adjacency.astype(np.float32)


def correlation_graph(
    train_power: np.ndarray,
    *,
    mode: str = "difference",
    k: int = 12,
    absolute: bool = True,
    include_self: bool = True,
    normalize: bool = True,
) -> np.ndarray:
    """Build a pairwise-complete correlation graph from Train data only.

    Input shape is `[time,node]`.  The function name deliberately says
    `train_power`: callers should not pass Val/Calib/Test observations.
    """

    values = np.asarray(train_power, dtype=float)
    if values.ndim != 2:
        raise ValueError("train_power must have shape [time,node]")
    if mode == "difference":
        values = np.diff(values, axis=0)
    elif mode != "raw":
        raise ValueError("mode must be 'raw' or 'difference'")
    n = values.shape[1]
    corr = np.eye(n, dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            valid = np.isfinite(values[:, i]) & np.isfinite(values[:, j])
            if valid.sum() < 3:
                value = 0.0
            else:
                xi, xj = values[valid, i], values[valid, j]
                if np.std(xi) == 0.0 or np.std(xj) == 0.0:
                    value = 0.0
                else:
                    value = float(np.corrcoef(xi, xj)[0, 1])
            corr[i, j] = corr[j, i] = value
    adjacency = np.abs(corr) if absolute else np.maximum(corr, 0.0)
    if not include_self:
        np.fill_diagonal(adjacency, 0.0)
    adjacency = _top_k(adjacency, min(k + int(include_self), n))
    return _row_normalize(adjacency) if normalize else adjacency.astype(np.float32)


def directional_graph(
    coordinates: np.ndarray,
    wind_from_degrees: np.ndarray,
    *,
    distance_scale: float,
    sigma_degrees: float = 30.0,
    sector_degrees: float = 90.0,
    include_self: bool = True,
    normalize: bool = True,
) -> np.ndarray:
    """Construct a batch of downwind directional/advection graphs.

    Coordinates are `(east, north)`.  Meteorological wind direction is assumed
    to denote where wind comes from, so propagation heads `direction + 180°`.
    `wind_from_degrees` can be `[batch]` or `[batch,node]`; node-wise directions
    define each source row independently.  The convention is
    `A[source,destination]`; neural layers aggregate incoming source messages.
    """

    xy = np.asarray(coordinates, dtype=float)
    direction = np.asarray(wind_from_degrees, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("coordinates must have shape [node,2]")
    if direction.ndim == 1:
        direction = np.repeat(direction[:, None], xy.shape[0], axis=1)
    if direction.ndim != 2 or direction.shape[1] != xy.shape[0]:
        raise ValueError("wind direction must have shape [batch] or [batch,node]")
    if distance_scale <= 0 or sigma_degrees <= 0 or not 0 < sector_degrees <= 360:
        raise ValueError("direction graph scales must be positive")

    delta = xy[None, :, :] - xy[:, None, :]  # source i -> destination j
    distance = np.linalg.norm(delta, axis=-1)
    bearing = np.degrees(np.arctan2(delta[..., 0], delta[..., 1])) % 360.0
    downwind = (direction + 180.0) % 360.0
    angle_error = (bearing[None, :, :] - downwind[:, :, None] + 180.0) % 360.0 - 180.0
    adjacency = np.exp(-distance[None] / distance_scale)
    adjacency = adjacency * np.exp(-0.5 * np.square(angle_error / sigma_degrees))
    adjacency *= np.abs(angle_error) <= sector_degrees / 2.0
    if include_self:
        diagonal = np.arange(xy.shape[0])
        adjacency[:, diagonal, diagonal] = 1.0
    else:
        diagonal = np.arange(xy.shape[0])
        adjacency[:, diagonal, diagonal] = 0.0
    return _row_normalize(adjacency) if normalize else adjacency.astype(np.float32)
