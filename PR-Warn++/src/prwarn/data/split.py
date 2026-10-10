"""Chronological split helpers with explicit boundary metadata."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


DEFAULT_RATIOS = {"train": 0.60, "val": 0.15, "calib": 0.10, "test": 0.15}


def chronological_split(
    frame: pd.DataFrame,
    *,
    timestamp_col: str = "Tmstamp",
    ratios: Mapping[str, float] = DEFAULT_RATIOS,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, str | int]]]:
    """Split unique timestamps chronologically, never individual rows.

    Windows must subsequently be generated independently inside each returned
    frame.  This makes cross-boundary windows impossible by construction.
    """

    names = list(ratios)
    if names != ["train", "val", "calib", "test"]:
        raise ValueError("ratios must be ordered train, val, calib, test")
    if abs(sum(ratios.values()) - 1.0) > 1e-9 or any(v <= 0 for v in ratios.values()):
        raise ValueError("split ratios must be positive and sum to 1")
    timestamps = pd.Index(pd.to_datetime(frame[timestamp_col]).unique()).sort_values()
    if len(timestamps) < len(names):
        raise ValueError("not enough unique timestamps for four splits")

    raw_counts = np.asarray([ratios[name] * len(timestamps) for name in names])
    counts = np.floor(raw_counts).astype(int)
    # A four-way protocol is not meaningful with an empty split.  Reserve one
    # timestamp for each split, then assign rounding remainder deterministically.
    counts = np.maximum(counts, 1)
    while counts.sum() > len(timestamps):
        candidates = np.flatnonzero(counts > 1)
        counts[candidates[np.argmax(counts[candidates] - raw_counts[candidates])]] -= 1
    for index in np.argsort(-(raw_counts - np.floor(raw_counts)), kind="stable"):
        if counts.sum() == len(timestamps):
            break
        counts[index] += 1
    edges = np.concatenate([[0], np.cumsum(counts)]).tolist()

    splits: dict[str, pd.DataFrame] = {}
    manifest: dict[str, dict[str, str | int]] = {}
    ts_values = pd.to_datetime(frame[timestamp_col])
    for name, left, right in zip(names, edges[:-1], edges[1:]):
        selected = timestamps[left:right]
        if selected.empty:
            raise ValueError(f"split {name} is empty")
        part = frame.loc[ts_values.isin(selected)].copy()
        splits[name] = part
        manifest[name] = {
            "start": selected[0].isoformat(),
            "end": selected[-1].isoformat(),
            "n_timestamps": len(selected),
            "n_rows": len(part),
        }
    return splits, manifest
