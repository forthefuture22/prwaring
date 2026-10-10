"""Build leakage-safe A10 Calib/Test arrays from frozen run artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from prwarn.risk.proxies import MahalanobisOOD, observed_ramp_event, ramp_probability


def _hidden(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as source:
        # Mean pooling is node-count invariant and keeps covariance estimation stable.
        return source["hidden"][:].mean(axis=1), source["origin_time"][:]


def _processed_quality(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        mask = source["mask"][:].astype(bool)
        origin = source["origin_time"][:].astype("datetime64[ns]").astype(np.int64)
    return 1.0 - mask.mean(axis=(1, 2, 3)), origin


def _split_arrays(
    scenario_path: Path,
    hidden_path: Path,
    processed_path: Path,
    detector: MahalanobisOOD,
    *,
    threshold: float,
    duration_steps: int,
) -> dict[str, np.ndarray]:
    with h5py.File(scenario_path, "r") as source:
        scenarios = source["farm_scenarios"][:].transpose(1, 0, 2)
        truth = source["y_farm"][:]
        truth_mask = source["y_farm_mask"][:].astype(bool)
        current = source["current_y_farm"][:]
        current_mask = source["current_y_farm_mask"][:].astype(bool)
        origin = source["origin_time"][:]
    hidden, hidden_origin = _hidden(hidden_path)
    data_quality, processed_origin = _processed_quality(processed_path)
    if not np.array_equal(origin, hidden_origin) or not np.array_equal(origin, processed_origin):
        raise ValueError("scenario, deterministic and processed origins differ")
    probability = ramp_probability(
        scenarios, threshold=threshold, duration_steps=duration_steps,
        direction="down", current_power=current,
    )
    event, valid = observed_ramp_event(
        truth, threshold=threshold, duration_steps=duration_steps,
        direction="down", current_power=current, future_mask=truth_mask,
        current_mask=current_mask,
    )
    probability = np.where(valid, probability, np.nan)
    event = np.where(valid, event, np.nan)
    horizon = truth.shape[-1]
    return {
        "event": event.astype(np.float32),
        "core_probability": probability.astype(np.float32),
        "ood": np.broadcast_to(detector.score(hidden)[:, None], (len(hidden), horizon)).astype(np.float32),
        "data_quality": np.broadcast_to(data_quality[:, None], (len(hidden), horizon)).astype(np.float32),
        "origin_time": origin,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-run", type=Path, required=True)
    parser.add_argument("--deterministic-run", type=Path, required=True)
    parser.add_argument("--processed-data", type=Path, required=True)
    parser.add_argument("--threshold-fraction", type=float, default=0.10)
    parser.add_argument("--duration-steps", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 < args.threshold_fraction < 1 or args.duration_steps <= 0:
        raise ValueError("invalid ramp definition")
    with h5py.File(args.scenario_run / "scenarios_calib.h5", "r") as source:
        rated_power = float(source.attrs["rated_power"])
        nodes = int(source.attrs["n_nodes"])
    threshold = args.threshold_fraction * rated_power * nodes
    train_hidden, _ = _hidden(args.deterministic_run / "deterministic_train.h5")
    detector = MahalanobisOOD().fit(train_hidden)
    payload = {}
    for split in ("calib", "test"):
        arrays = _split_arrays(
            args.scenario_run / f"scenarios_{split}.h5",
            args.deterministic_run / f"deterministic_{split}.h5",
            args.processed_data / f"{split}.npz", detector,
            threshold=threshold, duration_steps=args.duration_steps,
        )
        payload.update({f"{split}_{name}": value for name, value in arrays.items()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    metadata = {
        "protocol": "train_ood_fit_calib_test_frozen",
        "ramp_threshold": threshold,
        "threshold_fraction": args.threshold_fraction,
        "duration_steps": args.duration_steps,
        "ood_representation": "node_mean_deterministic_hidden",
        "data_quality": "one_minus_observed_history_fraction",
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), **metadata}, indent=2))


if __name__ == "__main__":
    main()
