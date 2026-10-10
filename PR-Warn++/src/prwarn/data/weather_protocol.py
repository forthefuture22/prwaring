"""Leakage-safe weather-availability contracts for A11."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


WEATHER_PROTOCOLS = ("history_only", "oracle_era5", "issue_time_forecast")


@dataclass(frozen=True)
class WeatherAvailabilityReport:
    protocol: str
    row_count: int
    future_valid_count: int
    deployable: bool
    issue_time_required: bool
    leakage_violations: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def audit_weather_availability(
    origin_time: np.ndarray,
    valid_time: np.ndarray,
    *,
    protocol: str,
    issue_time: np.ndarray | None = None,
    raise_on_violation: bool = True,
) -> WeatherAvailabilityReport:
    """Audit weather rows used by a model against their forecast origin.

    Times must use a common sortable representation (epoch integer or
    ``datetime64``). For deployable forecasts, every issue time must be no
    later than the corresponding forecast origin.
    """

    if protocol not in WEATHER_PROTOCOLS:
        raise ValueError(f"unknown weather protocol: {protocol}")
    origin, valid = np.broadcast_arrays(np.asarray(origin_time), np.asarray(valid_time))
    future = valid > origin
    violation = np.zeros(origin.shape, dtype=bool)
    issue_required = protocol == "issue_time_forecast"
    if protocol == "history_only":
        violation |= future
    elif protocol == "issue_time_forecast":
        if issue_time is None:
            violation |= True
        else:
            issue = np.broadcast_to(np.asarray(issue_time), origin.shape)
            violation |= issue > origin
            violation |= issue > valid
    # Oracle ERA5 intentionally permits future reanalysis fields but is marked
    # non-deployable in the returned provenance.
    count = int(violation.sum())
    report = WeatherAvailabilityReport(
        protocol=protocol,
        row_count=int(origin.size),
        future_valid_count=int(future.sum()),
        deployable=protocol != "oracle_era5",
        issue_time_required=issue_required,
        leakage_violations=count,
    )
    if count and raise_on_violation:
        raise ValueError(
            f"weather availability violation: protocol={protocol}, rows={count}"
        )
    return report


def select_latest_available_issue(
    origins: np.ndarray,
    requested_valid_times: np.ndarray,
    archive_issue_times: np.ndarray,
    archive_valid_times: np.ndarray,
) -> np.ndarray:
    """Return archive row indices using latest issue <= origin, or ``-1``.

    This simple deterministic join is intended for archive preprocessing and
    makes the no-future-issue rule independently testable before GPU training.
    """

    origins = np.asarray(origins)
    requested = np.asarray(requested_valid_times)
    issues = np.asarray(archive_issue_times)
    valid = np.asarray(archive_valid_times)
    if origins.ndim != 1 or requested.ndim != 2 or requested.shape[0] != origins.size:
        raise ValueError("origins must be [B] and requested_valid_times [B,H]")
    if issues.ndim != 1 or valid.shape != issues.shape:
        raise ValueError("archive issue/valid times must be equal-length vectors")
    selected = np.full(requested.shape, -1, dtype=np.int64)
    for batch_index, origin in enumerate(origins):
        available = issues <= origin
        for horizon_index, target_valid in enumerate(requested[batch_index]):
            candidates = np.flatnonzero(available & (valid == target_valid))
            if candidates.size:
                selected[batch_index, horizon_index] = candidates[
                    np.argmax(issues[candidates])
                ]
    return selected
