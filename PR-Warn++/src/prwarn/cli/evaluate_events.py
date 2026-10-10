"""Event-level extreme-weather evaluation (gap 2: extreme_rmse/mae/picp).

Reads a finished flow run and, on the subset of forecast horizons that fall
inside observed ramp episodes, reports conditional error/interval quality
separated from the overall average (C1-1).  Thresholds are fitted on the train
split only (C1-2) and the event table plus sample counts are emitted (C1-3).

Usage:
    python -m prwarn.cli.evaluate_events \\
        --flow-run outputs/<run> --data-dir data/<processed> \\
        --config configs/sdwpf_v3_2.yaml \\
        --output outputs/<run>/evaluation/events_metrics.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import yaml

from prwarn.data.processed import load_processed_split
from prwarn.eval.event_definition import (
    fit_extreme_thresholds,
    future_timestamp_grid,
    link_event_episodes,
)
from prwarn.eval.statistics import event_bootstrap_ci

_NS_PER_MINUTE = 60 * 1_000_000_000


def _resolve_calibrated(run_dir: Path) -> Path:
    candidates = [
        run_dir / "evaluation" / "calibrated_test.h5",
        run_dir / "calibrated_test.h5",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"calibrated_test.h5 not found under {run_dir} (touched evaluation/ and root)"
    )


def _ci_section(result) -> dict[str, float | int] | None:
    if result is None:
        return None
    return {
        "estimate": result.estimate,
        "lower": result.lower,
        "upper": result.upper,
        "confidence": result.confidence,
        "n_boot": result.n_boot,
        "n_events": result.n_events,
    }


def _conditional_summary(
    squared_error: np.ndarray,
    abs_error: np.ndarray,
    hit: np.ndarray,
    step_episode: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> dict[str, object]:
    """Compute extreme_* point estimates and cluster-bootstrap CIs."""

    event_step = step_episode >= 0
    n_event_steps = int(event_step.sum())
    episodes = sorted(set(int(v) for v in step_episode[event_step].ravel()) - {-1})
    enough = n_event_steps >= 2 and len(episodes) >= 2

    if not enough:
        note = (
            "event subset too small (need >=2 event steps and >=2 episodes); "
            "extreme_* reported as null"
        )
        return {
            "extreme_rmse": None,
            "extreme_mae": None,
            "extreme_picp": None,
            "extreme_rmse_ci": None,
            "extreme_mae_ci": None,
            "extreme_picp_ci": None,
            "event_steps": n_event_steps,
            "note": note,
        }

    ids = step_episode[event_step].ravel()
    sse = squared_error[event_step].ravel()
    ae = abs_error[event_step].ravel()
    hits = hit[event_step].ravel().astype(float)

    rmse_ci = event_bootstrap_ci(
        sse, ids, statistic=lambda v: float(np.sqrt(np.mean(v))),
        n_boot=n_boot, seed=seed,
    )
    mae_ci = event_bootstrap_ci(ae, ids, statistic=np.mean, n_boot=n_boot, seed=seed + 1)
    picp_ci = event_bootstrap_ci(hits, ids, statistic=np.mean, n_boot=n_boot, seed=seed + 2)

    return {
        "extreme_rmse": float(np.sqrt(np.mean(sse))),
        "extreme_mae": float(np.mean(ae)),
        "extreme_picp": float(np.mean(hits)),
        "extreme_rmse_ci": _ci_section(rmse_ci),
        "extreme_mae_ci": _ci_section(mae_ci),
        "extreme_picp_ci": _ci_section(picp_ci),
        "event_steps": n_event_steps,
        "note": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flow-run", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="processed split directory with train.npz + metadata.json")
    parser.add_argument("--config", type=Path, default=Path("configs/sdwpf_v3_2.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wind-quantile", type=float, default=0.95)
    parser.add_argument("--temperature-quantile", type=float, default=0.95)
    parser.add_argument("--link-max-gap-steps", type=int, default=1)
    parser.add_argument("--bootstrap-n", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    risk = config["risk"]

    metadata = json.loads((args.data_dir / "metadata.json").read_text(encoding="utf-8"))
    train = load_processed_split(args.data_dir / "train.npz")
    thresholds = fit_extreme_thresholds(
        train, metadata,
        wind_quantile=args.wind_quantile,
        temperature_quantile=args.temperature_quantile,
    )
    resolution_minutes = float(metadata["resolution_minutes"])

    calibrated_path = _resolve_calibrated(args.flow_run)
    with h5py.File(calibrated_path, "r") as calib:
        lower = calib["lower"][:].astype(float)
        upper = calib["upper"][:].astype(float)
        y_farm = calib["y_farm"][:].astype(float)
        y_farm_mask = calib["y_farm_mask"][:].astype(bool)
        origin_time = calib["origin_time"][:].astype("int64")

    with h5py.File(args.flow_run / "scenarios_test.h5", "r") as scenarios:
        y_det = scenarios["y_det_farm"][:].astype(float)
        current_y_farm = scenarios["current_y_farm"][:].astype(float)
        current_mask = scenarios["current_y_farm_mask"][:].astype(bool)
        farm_capacity = (
            float(scenarios.attrs["rated_power"]) * int(scenarios.attrs["n_nodes"])
        )

    if not (y_farm.shape == y_det.shape == lower.shape == upper.shape):
        raise ValueError("calibrated_test.h5 and scenarios_test.h5 shapes disagree")

    thresholds_for_link = {
        **thresholds,
        "ramp_threshold_fractions": risk["ramp_threshold_fractions"],
        "ramp_durations_steps": risk["ramp_durations_steps"],
    }
    episode_ids, event_table = link_event_episodes(
        origin_time, y_farm, y_farm_mask, current_y_farm, current_mask,
        thresholds_for_link,
        farm_capacity=farm_capacity,
        resolution_minutes=resolution_minutes,
        link_max_gap_steps=args.link_max_gap_steps,
    )

    # Map each (origin, horizon) step onto the union timeline -> episode id.
    timeline = future_timestamp_grid(origin_time, y_farm.shape[1], resolution_minutes)
    timeline_ns = timeline.astype("int64")
    step_ns = int(round(resolution_minutes * _NS_PER_MINUTE))
    step_positions = (
        origin_time.reshape(-1, 1)
        + (np.arange(y_farm.shape[1], dtype="int64") + 1) * step_ns
    )
    step_episode = episode_ids[np.searchsorted(timeline_ns, step_positions)]

    valid = y_farm_mask & np.isfinite(y_farm)
    error = y_det - y_farm
    squared_error = np.square(error)
    abs_error = np.abs(error)
    hit = (y_farm >= lower) & (y_farm <= upper)

    conditional = _conditional_summary(
        squared_error, abs_error, hit, step_episode,
        n_boot=args.bootstrap_n, seed=args.bootstrap_seed,
    )

    overall = {
        "overall_rmse": float(np.sqrt(squared_error[valid].mean())),
        "overall_mae": float(abs_error[valid].mean()),
        "overall_picp": float(hit[valid].mean()),
    }

    event_step = step_episode >= 0
    in_event_origins = int((event_step & valid).any(axis=1).sum())

    payload: dict[str, object] = {
        "schema_version": 1,
        "flow_run": str(args.flow_run.resolve()),
        "thresholds": {
            "wind_speed_mps": thresholds["wind_speed_mps"],
            "wind_quantile": args.wind_quantile,
            "temperature_c": thresholds["temperature_c"],
            "temperature_quantile": args.temperature_quantile,
            "ramp_threshold_fractions": list(risk["ramp_threshold_fractions"]),
            "ramp_durations_steps": list(risk["ramp_durations_steps"]),
            "farm_capacity": farm_capacity,
            "fitted_on": "train_only",
        },
        "event_counts": {
            "episodes": len(event_table),
            "linked_event_timesteps": int((episode_ids >= 0).sum()),
            "event_steps": conditional["event_steps"],
            "in_event_origins": in_event_origins,
            "overall_origins": int(y_farm.shape[0]),
        },
        "event_table": event_table,
        **overall,
        "extreme_rmse": conditional["extreme_rmse"],
        "extreme_mae": conditional["extreme_mae"],
        "extreme_picp": conditional["extreme_picp"],
        "extreme_rmse_ci": conditional["extreme_rmse_ci"],
        "extreme_mae_ci": conditional["extreme_mae_ci"],
        "extreme_picp_ci": conditional["extreme_picp_ci"],
    }
    if conditional["note"]:
        payload["subset_note"] = conditional["note"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "episodes": len(event_table),
        "in_event_origins": in_event_origins,
        "extreme_rmse": payload["extreme_rmse"],
        "extreme_picp": payload["extreme_picp"],
    }, indent=2))


if __name__ == "__main__":
    main()
