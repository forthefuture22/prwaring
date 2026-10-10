"""Extreme-event definition: train-frozen thresholds and episode linking.

Gap 2 (constraints C1-1/C1-2/C1-3). Two building blocks:

1. :func:`fit_extreme_thresholds` -- fit the high-wind / high-temperature
   thresholds as distribution quantiles on the **train split only** (C1-2).
   The inverse-standardization mirrors
   ``prwarn.cli.evaluate_missing_stress._train_extreme_threshold``; the scaler
   itself was already frozen on train by the preprocessing step.  Ramp-amplitude
   thresholds are *not* fitted here: they stay frozen from config
   (``risk.ramp_threshold_fractions * farm_capacity``).

2. :func:`link_event_episodes` -- build the per-timestep observed ramp truth
   with :func:`prwarn.risk.proxies.observed_ramp_event` for every configured
   (fraction, duration, direction), project it onto the union future timeline,
   and link contiguous positives into episodes with
   :func:`prwarn.eval.statistics.contiguous_event_ids`.

Note on the calibrated artifacts: ``calibrated_test.h5`` / ``scenarios_test.h5``
only carry farm-level power (no future weather series), so the episode truth is
the union of observed ramp events.  The wind/temperature thresholds are fitted
on train and reported in the artifact (C1-2); they are not AND-masked here
because the test export does not contain a future weather series.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from prwarn.data.processed import ProcessedSplit
from prwarn.eval.statistics import contiguous_event_ids
from prwarn.risk.proxies import observed_ramp_event

_NS_PER_MINUTE = 60 * 1_000_000_000


def _column_index(metadata: Mapping[str, object], column_key: str) -> int:
    feature_index = {
        str(name): int(index) for name, index in metadata["feature_index"].items()
    }
    name = str(metadata["physical_columns"][column_key])
    if name not in feature_index:
        raise KeyError(
            f"physical column {column_key!r} maps to feature {name!r} which is "
            f"not present in feature_index"
        )
    return feature_index[name]


def _inverse(
    split: ProcessedSplit, index: int, scaler: Mapping[str, object]
) -> tuple[np.ndarray, np.ndarray]:
    """Return raw (un-standardized) history values and their validity mask."""

    mean = float(np.asarray(scaler["mean"], dtype=float)[index])
    std = float(np.asarray(scaler["std"], dtype=float)[index])
    raw = split.x[..., index].astype(float) * std + mean
    valid = split.mask[..., index].astype(bool) & np.isfinite(raw)
    return raw, valid


def fit_extreme_thresholds(
    train: ProcessedSplit,
    metadata: Mapping[str, object],
    *,
    wind_quantile: float = 0.95,
    temperature_quantile: float = 0.95,
) -> dict[str, float]:
    """Fit high-wind / high-temperature thresholds on the train split only.

    Returns ``{"wind_speed_mps": float, "temperature_c": float}``.  The ramp
    amplitude threshold is deliberately *not* fitted here; it is frozen from
    ``config.risk.ramp_threshold_fractions * farm_capacity``.

    Raises:
        ValueError: if a quantile leaves (0, 1), the scaler was not frozen on
            train, no valid observations exist, or the passed split falls
            outside the declared train window (when the manifest is present).
    """

    for label, quantile in (
        ("wind_quantile", wind_quantile),
        ("temperature_quantile", temperature_quantile),
    ):
        if not 0.0 < quantile < 1.0:
            raise ValueError(f"{label} must be in (0, 1)")

    scaler = metadata["feature_scaler"]
    if scaler.get("fit_split", "train") != "train":
        raise ValueError(
            "feature_scaler is not frozen on the train split; refusing to fit "
            "extreme thresholds (C1-2: thresholds may only touch train data)"
        )

    # Reject val/calib/test splits when the train time window is declared.
    manifest = metadata.get("split_manifest") or {}
    train_window = manifest.get("train")
    if isinstance(train_window, Mapping) and "start" in train_window and "end" in train_window:
        lo = np.datetime64(str(train_window["start"])).astype("datetime64[ns]").astype("int64")
        hi = np.datetime64(str(train_window["end"])).astype("datetime64[ns]").astype("int64")
        origins = np.asarray(train.origin_time).astype("int64")
        if origins.size and (origins.min() < lo or origins.max() > hi):
            raise ValueError(
                "refusing to fit extreme thresholds: split origin_time falls "
                "outside the declared train window (val/calib/test are forbidden)"
            )

    speed_raw, speed_valid = _inverse(train, _column_index(metadata, "wind_speed"), scaler)
    if not speed_valid.any():
        raise ValueError("train split has no valid wind-speed observations")
    wind_threshold = float(np.quantile(speed_raw[speed_valid], wind_quantile))

    temp_raw, temp_valid = _inverse(
        train, _column_index(metadata, "temperature"), scaler
    )
    if not temp_valid.any():
        raise ValueError("train split has no valid temperature observations")
    units = metadata.get("physical_units") or {}
    temperature_input = str(units.get("temperature_input", "celsius"))
    if temperature_input == "celsius":
        temperature_c = temp_raw
    elif temperature_input == "kelvin":
        temperature_c = temp_raw - 273.15
    else:
        raise ValueError(f"unsupported temperature_input unit: {temperature_input!r}")
    if not temp_valid.any():
        raise ValueError("train split has no valid temperature observations after unit conversion")
    temperature_threshold = float(np.quantile(temperature_c[temp_valid], temperature_quantile))

    return {"wind_speed_mps": wind_threshold, "temperature_c": temperature_threshold}


def future_timestamp_grid(
    origin_time: np.ndarray, horizon: int, resolution_minutes: float
) -> np.ndarray:
    """Union of future timesteps for all origins, sorted, ``datetime64[ns]``.

    ``future[h]`` sits ``h + 1`` resolution steps after the origin (the origin
    itself carries ``current_y_farm``), matching
    :func:`prwarn.risk.proxies.observed_ramp_event`.
    """

    origins = np.asarray(origin_time).astype("int64").reshape(-1)
    offset = (np.arange(horizon, dtype="int64") + 1) * int(round(resolution_minutes * _NS_PER_MINUTE))
    grid = origins[:, None] + offset[None, :]
    return np.unique(grid).astype("datetime64[ns]")


def link_event_episodes(
    origin_time: np.ndarray,
    y_farm: np.ndarray,
    y_farm_mask: np.ndarray,
    current_y_farm: np.ndarray,
    current_y_farm_mask: np.ndarray,
    thresholds: Mapping[str, object],
    *,
    farm_capacity: float,
    resolution_minutes: float,
    link_max_gap_steps: int = 1,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Link observed ramp positives into chronological episodes.

    Args:
        origin_time: ``[sample]`` epoch-nanosecond origins (test split only).
        y_farm: ``[sample, horizon]`` observed farm power.
        y_farm_mask: ``[sample, horizon]`` validity mask.
        current_y_farm: ``[sample]`` farm power at the origin.
        current_y_farm_mask: ``[sample]`` origin validity.
        thresholds: dict containing ``ramp_threshold_fractions`` and
            ``ramp_durations_steps`` (plus the fitted ``wind_speed_mps`` /
            ``temperature_c`` values, which are reported but not used to mask
            because the test export has no future weather series).
        farm_capacity: rated farm power (``rated_power * n_nodes``).
        resolution_minutes: grid spacing.
        link_max_gap_steps: adjacent positives separated by at most this many
            steps belong to the same episode.

    Returns:
        ``(episode_ids, event_table)`` where ``episode_ids`` aligns with the
        sorted union future timeline (``future_timestamp_grid``), non-event
        positions are ``-1``, and ``event_table`` has one row per episode:
        ``event_id`` / ``start`` / ``end`` / ``direction`` /
        ``max_magnitude_pct`` / ``duration_steps``.
    """

    future = np.asarray(y_farm, dtype=float)
    if future.ndim != 2:
        raise ValueError("y_farm must be [sample, horizon]")
    n, horizon = future.shape
    current = np.asarray(current_y_farm, dtype=float).reshape(-1)
    if current.shape != (n,):
        raise ValueError("current_y_farm must be [sample]")
    future_mask = np.asarray(y_farm_mask).astype(bool)
    current_mask = np.asarray(current_y_farm_mask).astype(bool).reshape(-1)
    if future_mask.shape != future.shape or current_mask.shape != (n,):
        raise ValueError("mask shapes do not match y_farm/current_y_farm")
    if farm_capacity <= 0:
        raise ValueError("farm_capacity must be positive")
    if link_max_gap_steps < 0:
        raise ValueError("link_max_gap_steps must be non-negative")

    fractions = [float(f) for f in thresholds["ramp_threshold_fractions"]]
    durations = [int(d) for d in thresholds["ramp_durations_steps"]]

    timeline = future_timestamp_grid(origin_time, horizon, resolution_minutes)
    timeline_ns = timeline.astype("int64")
    origins = np.asarray(origin_time).astype("int64").reshape(-1)
    step_ns = int(round(resolution_minutes * _NS_PER_MINUTE))
    step_positions = origins[:, None] + (np.arange(horizon, dtype="int64") + 1) * step_ns
    positions = np.searchsorted(timeline_ns, step_positions.reshape(-1)).reshape(n, horizon)

    event_union = np.zeros(timeline.shape, dtype=bool)
    power_sum = np.zeros(timeline.shape, dtype=float)
    power_count = np.zeros(timeline.shape, dtype=np.float64)
    valid_read = future_mask & np.isfinite(future)
    np.add.at(power_sum, positions[valid_read], future[valid_read])
    np.add.at(power_count, positions[valid_read], 1.0)
    power_timeline = np.divide(
        power_sum, power_count, out=np.full_like(power_sum, np.nan), where=power_count > 0
    )

    for fraction in fractions:
        threshold = float(fraction) * float(farm_capacity)
        for duration in durations:
            for direction in ("up", "down"):
                event, valid = observed_ramp_event(
                    future,
                    threshold=threshold,
                    duration_steps=int(duration),
                    direction=direction,
                    current_power=current,
                    future_mask=future_mask,
                    current_mask=current_mask,
                )
                flagged = valid & event.astype(bool)
                np.logical_or.at(event_union, positions[flagged], True)

    episode_ids = contiguous_event_ids(
        event_union,
        timeline,
        maximum_gap=np.timedelta64(int(link_max_gap_steps * step_ns), "ns"),
    )

    event_table: list[dict[str, object]] = []
    for episode_id in sorted(set(int(v) for v in episode_ids) - {-1}):
        indices = np.flatnonzero(episode_ids == episode_id)
        first, last = int(indices[0]), int(indices[-1])
        # Include the step entering the episode so a single-timestep ramp still
        # has a well-defined signed magnitude (the trigger is p[k]-p[k-1]).
        segment = power_timeline[max(0, first - 1) : last + 1]
        net_change = power_timeline[last] - power_timeline[first]
        max_magnitude_pct = 0.0
        signed_change = 0.0
        for duration in durations:
            if duration >= segment.size:
                continue
            changes = segment[duration:] - segment[:-duration]
            if changes.size:
                best = changes[int(np.argmax(np.abs(changes)))]
                magnitude = float(abs(best) / float(farm_capacity) * 100.0)
                if magnitude > max_magnitude_pct:
                    max_magnitude_pct = magnitude
                    signed_change = float(best)
        direction = "down" if signed_change < 0 else ("up" if signed_change > 0 else ("down" if net_change < 0 else "up"))
        event_table.append(
            {
                "event_id": episode_id,
                "start": str(timeline[first]),
                "end": str(timeline[last]),
                "direction": direction,
                "max_magnitude_pct": max_magnitude_pct,
                "duration_steps": int(indices.size),
            }
        )
    return episode_ids, event_table
