"""Contracts for processed split archives and online-available wind direction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np


REQUIRED_ARRAYS = (
    "x",
    "mask",
    "delta_t",
    "y",
    "y_mask",
    "p_pc",
    "current_y",
    "current_y_mask",
    "wind_from",
    "origin_time",
    "turbines",
)


@dataclass
class ProcessedSplit:
    x: np.ndarray
    mask: np.ndarray
    delta_t: np.ndarray
    y: np.ndarray
    y_mask: np.ndarray
    p_pc: np.ndarray
    current_y: np.ndarray
    current_y_mask: np.ndarray
    wind_from: np.ndarray
    origin_time: np.ndarray
    turbines: np.ndarray
    future_weather: np.ndarray | None = None
    future_weather_mask: np.ndarray | None = None

    def __len__(self) -> int:
        return self.x.shape[0]

    def validate(self) -> "ProcessedSplit":
        if self.x.ndim != 4:
            raise ValueError("x must have [sample,node,history,feature]")
        if self.mask.shape != self.x.shape or self.delta_t.shape != self.x.shape:
            raise ValueError("x/mask/delta_t shapes differ")
        sample, node = self.x.shape[:2]
        if self.y.ndim != 3 or self.y.shape[:2] != (sample, node):
            raise ValueError("y must have [sample,node,horizon]")
        if self.y_mask.shape != self.y.shape or self.p_pc.shape != self.y.shape:
            raise ValueError("y/y_mask/p_pc shapes differ")
        if self.current_y.shape != (sample, node) or self.current_y_mask.shape != (sample, node):
            raise ValueError("current_y/current_y_mask must have [sample,node]")
        if self.wind_from.shape != (sample, node):
            raise ValueError("wind_from must have [sample,node]")
        if self.origin_time.shape != (sample,) or self.turbines.shape != (node,):
            raise ValueError("origin_time or turbines shape mismatch")
        if (self.future_weather is None) != (self.future_weather_mask is None):
            raise ValueError("future weather values and mask must be provided together")
        if self.future_weather is not None:
            expected = (sample, node, self.y.shape[-1])
            if self.future_weather.ndim != 4 or self.future_weather.shape[:3] != expected:
                raise ValueError("future_weather must have [sample,node,horizon,feature]")
            if self.future_weather_mask.shape != self.future_weather.shape:
                raise ValueError("future weather values/mask shapes differ")
            if not np.array_equal(
                self.future_weather_mask, self.future_weather_mask.astype(bool)
            ):
                raise ValueError("future_weather_mask is not binary")
        if not np.array_equal(self.mask, self.mask.astype(bool)):
            raise ValueError("mask is not binary")
        if not np.array_equal(self.y_mask, self.y_mask.astype(bool)):
            raise ValueError("y_mask is not binary")
        return self


def load_processed_split(path: str | Path) -> ProcessedSplit:
    """Load and validate a `preprocess_sdwpf` NPZ archive."""

    with np.load(path, allow_pickle=False) as archive:
        absent = set(REQUIRED_ARRAYS) - set(archive.files)
        if absent:
            raise KeyError(f"processed split is missing arrays: {sorted(absent)}")
        values = {name: archive[name] for name in REQUIRED_ARRAYS}
        for optional in ("future_weather", "future_weather_mask"):
            if optional in archive:
                values[optional] = archive[optional]
        split = ProcessedSplit(**values)
    return split.validate()


def origin_wind_from_degrees(
    x_fill: np.ndarray,
    feature_index: Mapping[str, int],
    *,
    wind_direction_feature: str,
    mode: str = "global",
    nacelle_direction_feature: str | None = None,
) -> np.ndarray:
    """Return origin-time meteorological wind-from direction `[sample,node]`.

    Forward-filled values are allowed as numerical carriers.  Their age and
    validity remain available to the neural model through `mask/delta_t`.
    """

    values = np.asarray(x_fill)
    if values.ndim != 4:
        raise ValueError("x_fill must have [sample,node,history,feature]")
    if wind_direction_feature not in feature_index:
        raise KeyError(f"wind direction feature {wind_direction_feature!r} not found")
    wind = values[:, :, -1, feature_index[wind_direction_feature]].astype(float)
    if mode == "global":
        result = wind
    elif mode == "relative_plus_nacelle":
        if nacelle_direction_feature is None or nacelle_direction_feature not in feature_index:
            raise KeyError("relative_plus_nacelle requires a nacelle direction feature")
        nacelle = values[:, :, -1, feature_index[nacelle_direction_feature]].astype(float)
        result = wind + nacelle
    else:
        raise ValueError("wind direction mode must be 'global' or 'relative_plus_nacelle'")
    return np.mod(result, 360.0).astype(np.float32)
