"""Rigid-grid and mask construction for SDWPF-like SCADA tables.

The critical invariant is that invalid observations are represented, never
removed from the time axis.  Filled values are only numerical carriers; the
mask and elapsed time retain observation semantics.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Callable, Iterable, Mapping

import numpy as np
import pandas as pd


class ReasonCode(IntEnum):
    VALID = 0
    MISSING = 1
    UNKNOWN = 2
    ABNORMAL = 3


def build_rigid_grid(
    frame: pd.DataFrame,
    *,
    timestamp_col: str = "Tmstamp",
    turbine_col: str = "TurbID",
    frequency: str = "10min",
    turbines: Iterable[int | str] | None = None,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Reindex every turbine to the same complete time grid.

    Duplicate turbine/timestamp rows are rejected because silently choosing one
    would change the experimental truth.
    """

    required = {timestamp_col, turbine_col}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"missing required columns: {sorted(missing)}")

    work = frame.copy()
    work[timestamp_col] = pd.to_datetime(work[timestamp_col], errors="raise")
    if work.duplicated([turbine_col, timestamp_col]).any():
        examples = work.loc[
            work.duplicated([turbine_col, timestamp_col], keep=False),
            [turbine_col, timestamp_col],
        ].head()
        raise ValueError(f"duplicate turbine/timestamp rows found:\n{examples}")

    turbine_values = list(turbines) if turbines is not None else sorted(work[turbine_col].unique())
    if not turbine_values:
        raise ValueError("no turbines available")
    start_ts = pd.Timestamp(start) if start is not None else work[timestamp_col].min()
    end_ts = pd.Timestamp(end) if end is not None else work[timestamp_col].max()
    times = pd.date_range(start_ts, end_ts, freq=frequency)
    expected = pd.MultiIndex.from_product(
        [turbine_values, times], names=[turbine_col, timestamp_col]
    )
    result = work.set_index([turbine_col, timestamp_col]).reindex(expected).reset_index()
    return result.sort_values([timestamp_col, turbine_col], kind="stable").reset_index(drop=True)


def sdwpf_reason_codes(
    frame: pd.DataFrame,
    *,
    power_col: str = "Patv",
    wind_speed_col: str = "Wspd",
    pitch_cols: tuple[str, ...] = ("Pab1", "Pab2", "Pab3"),
    nacelle_direction_col: str = "Ndir",
    wind_direction_col: str = "Wdir",
) -> np.ndarray:
    """Return row-level official-style SDWPF reason codes.

    Priority is missing -> abnormal -> unknown.  The rules intentionally stay
    close to the published SDWPF usage notes.  Project-specific IQR/DBSCAN
    filters belong in sensitivity experiments, not in this ground-truth mask.
    """

    n = len(frame)
    codes = np.full(n, ReasonCode.VALID, dtype=np.int8)
    used = [power_col, wind_speed_col, nacelle_direction_col, wind_direction_col]
    used.extend(c for c in pitch_cols if c in frame.columns)
    existing = [c for c in used if c in frame.columns]
    if power_col not in frame or wind_speed_col not in frame:
        raise KeyError(f"{power_col!r} and {wind_speed_col!r} are required")

    missing = frame[existing].isna().any(axis=1).to_numpy()
    codes[missing] = ReasonCode.MISSING

    abnormal = np.zeros(n, dtype=bool)
    if nacelle_direction_col in frame:
        ndir = pd.to_numeric(frame[nacelle_direction_col], errors="coerce").to_numpy()
        abnormal |= (ndir < -720.0) | (ndir > 720.0)
    if wind_direction_col in frame:
        wdir = pd.to_numeric(frame[wind_direction_col], errors="coerce").to_numpy()
        abnormal |= (wdir < -180.0) | (wdir > 180.0)
    codes[(codes == ReasonCode.VALID) & abnormal] = ReasonCode.ABNORMAL

    power = pd.to_numeric(frame[power_col], errors="coerce").to_numpy()
    speed = pd.to_numeric(frame[wind_speed_col], errors="coerce").to_numpy()
    unknown = (power <= 0.0) & (speed > 2.5)
    for column in pitch_cols:
        if column in frame:
            pitch = pd.to_numeric(frame[column], errors="coerce").to_numpy()
            unknown |= pitch > 89.0
    codes[(codes == ReasonCode.VALID) & unknown] = ReasonCode.UNKNOWN
    return codes


def generic_scada_reason_codes(
    frame: pd.DataFrame,
    *,
    power_col: str,
    wind_speed_col: str,
    speed_max: float = 25.0,
) -> np.ndarray:
    """Dataset-agnostic row-level reason codes for SCADA tables without pitch/nacelle.

    This is the quality-code strategy for farms such as Kelmarsh, whose published
    table has no blade-pitch or nacelle-direction channels, so SDWPF's
    ``pitch > 89`` / ``Ndir in [-720, 720]`` / ``Wdir in [-180, 180]`` rules do
    not transfer.  Priority stays missing -> abnormal, matching
    :func:`sdwpf_reason_codes`:

    * ``MISSING`` -- power or wind speed is absent/unparseable;
    * ``ABNORMAL`` -- wind speed exceeds ``speed_max`` (anemometer over-range,
      beyond typical cut-out) or power is negative (non-physical generation);
    * ``VALID`` -- everything else.

    The ``UNKNOWN`` bucket deliberately collapses into ``ABNORMAL`` because the
    only distinction available without the SDWPF-specific channels is
    "physically usable" versus "not".
    """

    if power_col not in frame.columns or wind_speed_col not in frame.columns:
        raise KeyError(f"{power_col!r} and {wind_speed_col!r} are required")

    n = len(frame)
    codes = np.full(n, ReasonCode.VALID, dtype=np.int8)
    power = pd.to_numeric(frame[power_col], errors="coerce").to_numpy()
    speed = pd.to_numeric(frame[wind_speed_col], errors="coerce").to_numpy()

    missing = np.isnan(power) | np.isnan(speed)
    codes[missing] = ReasonCode.MISSING

    abnormal = (speed > speed_max) | (power < 0.0)
    codes[(codes == ReasonCode.VALID) & abnormal] = ReasonCode.ABNORMAL
    return codes


def _time_since_valid(valid: np.ndarray, step_minutes: float) -> np.ndarray:
    """Compute elapsed minutes since the last valid value for a 2-D array."""

    if valid.ndim != 2:
        raise ValueError("valid must have shape [time, feature]")
    elapsed = np.empty(valid.shape, dtype=np.float32)
    counters = np.zeros(valid.shape[1], dtype=np.float32)
    seen = np.zeros(valid.shape[1], dtype=bool)
    for t in range(valid.shape[0]):
        is_valid = valid[t]
        counters = np.where(is_valid, 0.0, counters + step_minutes)
        seen |= is_valid
        # Before the first observation, elapsed time still increases.  The mask
        # makes its "not observed yet" meaning explicit.
        elapsed[t] = np.where(seen, counters, (t + 1) * step_minutes)
    return elapsed


def build_sdwpf_bundle(
    rigid_frame: pd.DataFrame,
    feature_columns: list[str],
    *,
    timestamp_col: str = "Tmstamp",
    turbine_col: str = "TurbID",
    step_minutes: float = 10.0,
    train_means: Mapping[str, float] | None = None,
    reason_code_fn: Callable[[pd.DataFrame], np.ndarray] | None = None,
) -> dict[str, object]:
    """Create aligned `X_fill`, `M`, `delta_t`, and row reason codes.

    `train_means` must be fitted from the Train split by the caller.  If it is
    omitted, means are returned for inspection but no non-forward-fill fallback
    is allowed, which prevents accidental full-dataset leakage.

    `reason_code_fn` selects the row-validity policy per dataset.  When `None`
    the legacy SDWPF policy (`sdwpf_reason_codes`) is used, which keeps every
    existing caller backward compatible.  External farms without pitch/nacelle
    channels pass `generic_scada_reason_codes` instead.
    """

    absent = set(feature_columns) - set(rigid_frame.columns)
    if absent:
        raise KeyError(f"feature columns missing: {sorted(absent)}")
    work = rigid_frame.sort_values([turbine_col, timestamp_col], kind="stable").copy()
    if reason_code_fn is None:
        codes = sdwpf_reason_codes(work)
    else:
        codes = reason_code_fn(work)
        codes = np.asarray(codes)
        if codes.shape[0] != len(work):
            raise ValueError("reason_code_fn must return one code per frame row")
    row_valid = codes == ReasonCode.VALID

    values = work[feature_columns].apply(pd.to_numeric, errors="coerce")
    observed = values.notna().to_numpy() & row_valid[:, None]
    fill = values.where(observed)
    fill = fill.groupby(work[turbine_col], sort=False).ffill()

    fitted_means = {
        column: float(values.loc[observed[:, i], column].mean())
        for i, column in enumerate(feature_columns)
    }
    if train_means is not None:
        missing_means = set(feature_columns) - set(train_means)
        if missing_means:
            raise KeyError(f"train means missing: {sorted(missing_means)}")
        invalid_means = [c for c in feature_columns if not np.isfinite(float(train_means[c]))]
        if invalid_means:
            raise ValueError(f"non-finite Train means: {invalid_means}")
        fill = fill.fillna({column: float(train_means[column]) for column in feature_columns})

    delta = np.empty(observed.shape, dtype=np.float32)
    for _, indices in work.groupby(turbine_col, sort=False).indices.items():
        idx = np.asarray(indices)
        delta[idx] = _time_since_valid(observed[idx], step_minutes)

    return {
        "frame": work,
        "x_fill": fill.to_numpy(dtype=np.float32),
        "mask": observed.astype(np.float32),
        "delta_t": delta,
        "reason_code": codes,
        "fitted_means": fitted_means,
    }


def bundle_to_time_node(
    bundle: Mapping[str, object],
    *,
    timestamp_col: str = "Tmstamp",
    turbine_col: str = "TurbID",
) -> dict[str, np.ndarray]:
    """Reshape a bundle from rows to aligned `[time,node,...]` tensors."""

    frame = bundle["frame"]
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("bundle['frame'] must be a DataFrame")
    times = pd.Index(pd.to_datetime(frame[timestamp_col]).unique()).sort_values()
    turbines = pd.Index(frame[turbine_col].unique()).sort_values()
    row_key = pd.MultiIndex.from_frame(frame[[timestamp_col, turbine_col]])
    expected = pd.MultiIndex.from_product([times, turbines], names=[timestamp_col, turbine_col])
    indexer = row_key.get_indexer(expected)
    if np.any(indexer < 0):
        raise ValueError("bundle is not a complete rigid grid")

    result: dict[str, np.ndarray] = {
        "timestamps": times.to_numpy(),
        "turbines": turbines.to_numpy(),
    }
    for name in ("x_fill", "mask", "delta_t"):
        values = np.asarray(bundle[name])
        result[name] = values[indexer].reshape(len(times), len(turbines), values.shape[-1])
    for name in ("reason_code",):
        values = np.asarray(bundle[name])
        result[name] = values[indexer].reshape(len(times), len(turbines))
    return result
