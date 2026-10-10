"""Leakage-safe sliding-window construction."""

from __future__ import annotations

import numpy as np


def make_windows(
    x: np.ndarray,
    mask: np.ndarray,
    delta_t: np.ndarray,
    y: np.ndarray,
    y_mask: np.ndarray,
    *,
    history_steps: int,
    forecast_steps: int,
    stride: int = 1,
) -> dict[str, np.ndarray]:
    """Window arrays aligned as `[time, node, feature]` and `[time, node]`."""

    if x.ndim != 3 or mask.shape != x.shape or delta_t.shape != x.shape:
        raise ValueError("x, mask and delta_t must share [time,node,feature]")
    if y.ndim != 2 or y_mask.shape != y.shape or y.shape[:2] != x.shape[:2]:
        raise ValueError("y and y_mask must share [time,node] with x")
    if history_steps <= 0 or forecast_steps <= 0 or stride <= 0:
        raise ValueError("history_steps, forecast_steps and stride must be positive")
    final_origin = x.shape[0] - forecast_steps
    origins = np.arange(history_steps - 1, final_origin, stride, dtype=np.int64)
    if origins.size == 0:
        raise ValueError("split is too short for the requested window")

    xw = np.stack([x[o - history_steps + 1 : o + 1] for o in origins])
    mw = np.stack([mask[o - history_steps + 1 : o + 1] for o in origins])
    dtw = np.stack([delta_t[o - history_steps + 1 : o + 1] for o in origins])
    yw = np.stack([y[o + 1 : o + 1 + forecast_steps] for o in origins])
    ymw = np.stack([y_mask[o + 1 : o + 1 + forecast_steps] for o in origins])
    # [B,L,N,D] -> [B,N,L,D]; [B,H,N] -> [B,N,H]
    return {
        "x": np.transpose(xw, (0, 2, 1, 3)),
        "mask": np.transpose(mw, (0, 2, 1, 3)),
        "delta_t": np.transpose(dtw, (0, 2, 1, 3)),
        "y": np.transpose(yw, (0, 2, 1)),
        "y_mask": np.transpose(ymw, (0, 2, 1)),
        "origin_index": origins,
    }
