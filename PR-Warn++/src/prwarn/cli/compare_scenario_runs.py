"""Paired frozen-Test comparison with CRPS/ES/VS, HAC/DM and cluster bootstrap."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import h5py
import numpy as np

from prwarn.eval.metrics import (
    crps_ensemble_values,
    energy_score_values,
    variogram_score_values,
)
from prwarn.eval.statistics import (
    contiguous_event_ids,
    diebold_mariano,
    event_bootstrap_ci,
)
from prwarn.risk.proxies import observed_ramp_event, ramp_probability


def _validate_pair(first: h5py.File, second: h5py.File) -> None:
    required = (
        "farm_scenarios",
        "y_farm",
        "y_farm_mask",
        "current_y_farm",
        "current_y_farm_mask",
        "origin_time",
    )
    for name in required:
        if name not in first or name not in second:
            raise KeyError(f"paired artifact is missing {name}")
    for name in ("y_farm", "y_farm_mask", "current_y_farm", "current_y_farm_mask", "origin_time"):
        if not np.array_equal(first[name][:], second[name][:]):
            raise ValueError(f"paired artifacts differ in frozen field {name}")
    if first["farm_scenarios"].shape != second["farm_scenarios"].shape:
        raise ValueError("paired scenario tensors must use the same B/M/H shape")
    for attribute in ("rated_power", "n_nodes"):
        if first.attrs[attribute] != second.attrs[attribute]:
            raise ValueError(f"paired artifacts differ in {attribute}")


def _origin_scores(source: h5py.File, *, chunk_size: int) -> dict[str, np.ndarray]:
    n = len(source["y_farm"])
    result = {
        "crps": np.full(n, np.nan),
        "energy_score": np.full(n, np.nan),
        "variogram_score": np.full(n, np.nan),
    }
    for left in range(0, n, chunk_size):
        right = min(left + chunk_size, n)
        truth = source["y_farm"][left:right].astype(float)
        mask = source["y_farm_mask"][left:right].astype(bool)
        scenarios = source["farm_scenarios"][left:right].astype(float).transpose(1, 0, 2)
        element_crps = crps_ensemble_values(truth, scenarios)
        valid_count = mask.sum(axis=1)
        valid_origin = valid_count > 0
        crps = np.full(right - left, np.nan)
        crps[valid_origin] = (
            (element_crps * mask).sum(axis=1)[valid_origin] / valid_count[valid_origin]
        )
        result["crps"][left:right] = crps
        complete = mask.all(axis=1)
        if complete.any():
            energy = np.full(right - left, np.nan)
            variogram = np.full(right - left, np.nan)
            energy[complete] = energy_score_values(
                truth[complete], scenarios[:, complete]
            )
            variogram[complete] = variogram_score_values(
                truth[complete], scenarios[:, complete]
            )
            result["energy_score"][left:right] = energy
            result["variogram_score"][left:right] = variogram
    return result


def _finite(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _paired_loss_report(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    cluster_ids: np.ndarray,
    *,
    hac_lags: int,
    n_boot: int,
    confidence: float,
    seed: int,
) -> dict[str, object]:
    valid = np.isfinite(loss_a) & np.isfinite(loss_b)
    a, b, groups = loss_a[valid], loss_b[valid], cluster_ids[valid]
    if a.size == 0:
        raise ValueError("paired metric has no finite origins")
    lags = min(hac_lags, len(a) - 1)
    dm = diebold_mariano(a, b, hac_lags=lags)
    bootstrap = event_bootstrap_ci(
        a - b,
        groups,
        n_boot=n_boot,
        confidence=confidence,
        seed=seed,
    )
    dm_payload = asdict(dm)
    dm_payload["statistic"] = _finite(dm.statistic)
    return {
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "mean_difference_a_minus_b": float(np.mean(a - b)),
        "count": int(a.size),
        "dm_hac": dm_payload,
        "cluster_bootstrap_difference": asdict(bootstrap),
    }


def _ramp_report(
    first: h5py.File,
    second: h5py.File,
    timestamps: np.ndarray,
    *,
    threshold: float,
    duration: int,
    direction: str,
    hac_lags: int,
    n_boot: int,
    confidence: float,
    event_gap_minutes: int,
    seed: int,
) -> dict[str, object]:
    scenarios_a = first["farm_scenarios"][:].transpose(1, 0, 2)
    scenarios_b = second["farm_scenarios"][:].transpose(1, 0, 2)
    current = first["current_y_farm"][:]
    probability_a = ramp_probability(
        scenarios_a,
        threshold=threshold,
        duration_steps=duration,
        direction=direction,
        current_power=current,
    )
    probability_b = ramp_probability(
        scenarios_b,
        threshold=threshold,
        duration_steps=duration,
        direction=direction,
        current_power=current,
    )
    event, valid = observed_ramp_event(
        first["y_farm"][:],
        threshold=threshold,
        duration_steps=duration,
        direction=direction,
        current_power=current,
        future_mask=first["y_farm_mask"][:],
        current_mask=first["current_y_farm_mask"][:],
    )
    valid &= np.isfinite(probability_a) & np.isfinite(probability_b)
    event_valid = event[valid].astype(float)
    loss_a = np.square(probability_a[valid] - event_valid)
    loss_b = np.square(probability_b[valid] - event_valid)
    time_valid = timestamps[valid]
    day_groups = time_valid.astype("datetime64[D]").astype(np.int64)
    overall = _paired_loss_report(
        loss_a,
        loss_b,
        day_groups,
        hac_lags=hac_lags,
        n_boot=n_boot,
        confidence=confidence,
        seed=seed,
    )
    positive = event_valid.astype(bool)
    positive_report = None
    if positive.any():
        episode_ids = contiguous_event_ids(
            positive,
            time_valid,
            maximum_gap=np.timedelta64(event_gap_minutes, "m"),
        )
        bootstrap = event_bootstrap_ci(
            loss_a[positive] - loss_b[positive],
            episode_ids[positive],
            n_boot=n_boot,
            confidence=confidence,
            seed=seed + 1,
        )
        positive_report = {
            "positive_origin_count": int(positive.sum()),
            "event_episode_count": int(bootstrap.n_events),
            "mean_brier_a": float(np.mean(loss_a[positive])),
            "mean_brier_b": float(np.mean(loss_b[positive])),
            "event_bootstrap_difference": asdict(bootstrap),
        }
    return {
        "threshold": threshold,
        "duration_steps": duration,
        "direction": direction,
        "valid_count": int(valid.sum()),
        "positive_count": int(event_valid.sum()),
        "overall_brier": overall,
        "positive_event_episodes": positive_report,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-a", type=Path, required=True)
    parser.add_argument("--run-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--label-b", default="B")
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--hac-lags", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--ramp-threshold-fraction", type=float, default=0.10)
    parser.add_argument("--ramp-duration", type=int, default=3)
    parser.add_argument("--event-gap-minutes", type=int, default=10)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.chunk_size <= 0 or args.hac_lags < 0 or args.event_gap_minutes <= 0:
        raise ValueError("chunk size/gap must be positive and HAC lags non-negative")

    path_a = args.run_a / "scenarios_test.h5"
    path_b = args.run_b / "scenarios_test.h5"
    with h5py.File(path_a, "r") as first, h5py.File(path_b, "r") as second:
        _validate_pair(first, second)
        score_a = _origin_scores(first, chunk_size=args.chunk_size)
        score_b = _origin_scores(second, chunk_size=args.chunk_size)
        origin_ns = first["origin_time"][:]
        timestamps = origin_ns.astype("datetime64[ns]")
        day_groups = timestamps.astype("datetime64[D]").astype(np.int64)
        metrics = {}
        for index, name in enumerate(("crps", "energy_score", "variogram_score")):
            metrics[name] = _paired_loss_report(
                score_a[name],
                score_b[name],
                day_groups,
                hac_lags=args.hac_lags,
                n_boot=args.bootstrap,
                confidence=args.confidence,
                seed=args.seed + index,
            )
        farm_capacity = float(first.attrs["rated_power"]) * int(first.attrs["n_nodes"])
        threshold = args.ramp_threshold_fraction * farm_capacity
        ramp = {
            direction: _ramp_report(
                first,
                second,
                timestamps,
                threshold=threshold,
                duration=args.ramp_duration,
                direction=direction,
                hac_lags=args.hac_lags,
                n_boot=args.bootstrap,
                confidence=args.confidence,
                event_gap_minutes=args.event_gap_minutes,
                seed=args.seed + 100 + direction_index * 10,
            )
            for direction_index, direction in enumerate(("up", "down"))
        }
        payload = {
            "schema_version": 1,
            "label_a": args.label_a,
            "label_b": args.label_b,
            "run_a": str(args.run_a.resolve()),
            "run_b": str(args.run_b.resolve()),
            "scenario_method_a": str(first.attrs.get("scenario_method", "unknown")),
            "scenario_method_b": str(second.attrs.get("scenario_method", "unknown")),
            "sign_convention": "positive A-minus-B loss means B is better",
            "paired_origin_count": int(len(timestamps)),
            "scenario_count": int(first["farm_scenarios"].shape[1]),
            "metrics": metrics,
            "ramp": ramp,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "metrics": list(metrics)}))


if __name__ == "__main__":
    main()
