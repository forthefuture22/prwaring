"""Raw SDWPF audit reports used as the Gate-1 training precondition."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from .grid import ReasonCode, sdwpf_reason_codes


def audit_sdwpf_frame(
    frame: pd.DataFrame,
    *,
    timestamp_col: str = "Tmstamp",
    turbine_col: str = "TurbID",
    expected_turbines: int | None = 134,
    frequency_minutes: int = 10,
) -> dict:
    required = {timestamp_col, turbine_col, "Patv", "Wspd"}
    absent = required - set(frame.columns)
    if absent:
        raise KeyError(f"raw audit missing required columns: {sorted(absent)}")
    work = frame.copy()
    work[timestamp_col] = pd.to_datetime(work[timestamp_col], errors="raise")
    turbines = pd.Index(work[turbine_col].dropna().unique()).sort_values()
    timestamps = pd.Index(work[timestamp_col].dropna().unique()).sort_values()
    duplicates = int(work.duplicated([turbine_col, timestamp_col]).sum())
    interval_counter: Counter[float] = Counter()
    for _, group in work.groupby(turbine_col, sort=False):
        unique_time = pd.Index(group[timestamp_col].dropna().unique()).sort_values()
        if len(unique_time) > 1:
            minutes = np.diff(unique_time.to_numpy()).astype("timedelta64[s]").astype(float) / 60.0
            interval_counter.update(float(value) for value in minutes)

    codes = sdwpf_reason_codes(work)
    reason_counts = {
        reason.name.lower(): int(np.sum(codes == reason)) for reason in ReasonCode
    }
    per_turbine_valid = {}
    for turbine, indices in work.groupby(turbine_col, sort=False).indices.items():
        selected = codes[np.asarray(indices)]
        per_turbine_valid[str(turbine)] = float(np.mean(selected == ReasonCode.VALID))

    columns = {}
    for name in work.columns:
        series = work[name]
        item = {
            "dtype": str(series.dtype),
            "missing_fraction": float(series.isna().mean()),
            "unique": int(series.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(series):
            numeric = pd.to_numeric(series, errors="coerce")
            item["min"] = float(numeric.min()) if numeric.notna().any() else None
            item["max"] = float(numeric.max()) if numeric.notna().any() else None
        columns[name] = item

    expected_grid_rows = 0
    if len(timestamps):
        full_time = pd.date_range(timestamps[0], timestamps[-1], freq=f"{frequency_minutes}min")
        expected_grid_rows = len(full_time) * len(turbines)
    issues = []
    blocking_issues = []
    if expected_turbines is not None and len(turbines) != expected_turbines:
        message = f"expected {expected_turbines} turbines, observed {len(turbines)}"
        issues.append(message)
        blocking_issues.append(message)
    if duplicates:
        message = f"found {duplicates} duplicate turbine/timestamp rows"
        issues.append(message)
        blocking_issues.append(message)
    if interval_counter and interval_counter.most_common(1)[0][0] != float(frequency_minutes):
        message = (
            f"dominant interval is {interval_counter.most_common(1)[0][0]} min, "
            f"expected {frequency_minutes}"
        )
        issues.append(message)
        blocking_issues.append(message)
    if expected_grid_rows and len(work) != expected_grid_rows:
        issues.append(
            f"raw rows {len(work)} differ from complete grid rows {expected_grid_rows}; reindex is required"
        )
    return {
        "rows": len(work),
        "columns": columns,
        "n_turbines": len(turbines),
        "turbines": turbines.tolist(),
        "n_unique_timestamps": len(timestamps),
        "start": timestamps[0].isoformat() if len(timestamps) else None,
        "end": timestamps[-1].isoformat() if len(timestamps) else None,
        "duplicate_turbine_timestamps": duplicates,
        "expected_grid_rows": expected_grid_rows,
        "grid_row_gap": expected_grid_rows - len(work),
        "interval_minutes_counts": {
            str(key): value for key, value in sorted(interval_counter.items())
        },
        "reason_counts": reason_counts,
        "per_turbine_valid_fraction": per_turbine_valid,
        "issues": issues,
        "blocking_issues": blocking_issues,
        "gate_ready": not blocking_issues,
    }
