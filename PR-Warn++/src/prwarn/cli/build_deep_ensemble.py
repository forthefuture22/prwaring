"""Combine independently trained deterministic caches into a scenario ensemble."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def _validate_members(paths: list[Path], split: str) -> tuple[int, int, int, float]:
    shape = None
    rated_power = None
    reference = None
    for path in paths:
        with h5py.File(path / f"deterministic_{split}.h5", "r") as source:
            current_shape = tuple(source["y"].shape)
            current_power = float(source.attrs["rated_power"])
            fields = (source["y"][:], source["y_mask"][:], source["origin_time"][:])
        if shape is None:
            shape, rated_power, reference = current_shape, current_power, fields
        elif current_shape != shape or current_power != rated_power:
            raise ValueError("ensemble member shapes or rated powers differ")
        elif any(not np.array_equal(a, b) for a, b in zip(reference, fields)):
            raise ValueError("ensemble members do not share frozen targets/origins")
    return (*shape, float(rated_power))


def _write_split(paths: list[Path], split: str, output_path: Path, alpha: float) -> None:
    samples, nodes, horizon, rated_power = _validate_members(paths, split)
    members = len(paths)
    with h5py.File(paths[0] / f"deterministic_{split}.h5", "r") as reference:
        y = reference["y"][:]
        mask = reference["y_mask"][:]
        current = reference["current_y"][:]
        current_mask = reference["current_y_mask"][:]
        origin = reference["origin_time"][:]
    predictions = []
    for path in paths:
        with h5py.File(path / f"deterministic_{split}.h5", "r") as source:
            predictions.append(source["y_det"][:])
    turbine = np.stack(predictions, axis=1)
    farm = turbine.sum(axis=2)
    with h5py.File(output_path, "w") as output:
        output.attrs["n_scenarios"] = members
        output.attrs["rated_power"] = rated_power
        output.attrs["n_nodes"] = nodes
        output.attrs["interval_alpha"] = alpha
        output.attrs["scenario_method"] = "deep_ensemble"
        output.create_dataset("farm_scenarios", data=farm, compression="lzf")
        output.create_dataset("lower", data=np.quantile(farm, alpha / 2, axis=1), compression="lzf")
        output.create_dataset("upper", data=np.quantile(farm, 1 - alpha / 2, axis=1), compression="lzf")
        output.create_dataset("y_farm", data=y.sum(axis=1), compression="lzf")
        output.create_dataset("y_farm_mask", data=mask.astype(bool).all(axis=1), compression="lzf")
        output.create_dataset("y_det_farm", data=turbine.mean(axis=1).sum(axis=1), compression="lzf")
        output.create_dataset("current_y_farm", data=current.sum(axis=1), compression="lzf")
        output.create_dataset("current_y_farm_mask", data=current_mask.astype(bool).all(axis=1), compression="lzf")
        output.create_dataset("origin_time", data=origin, compression="lzf")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--member", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.10)
    args = parser.parse_args()
    if len(args.member) < 2 or len({path.resolve() for path in args.member}) != len(args.member):
        raise ValueError("deep ensemble needs at least two unique member runs")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    config = {
        "method": "deep_ensemble", "members": [str(path.resolve()) for path in args.member],
        "member_count": len(args.member), "alpha": args.alpha,
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    for split in ("calib", "test"):
        _write_split(args.member, split, args.output_dir / f"scenarios_{split}.h5", args.alpha)
    print(json.dumps({"output_dir": str(args.output_dir), "members": len(args.member)}))


if __name__ == "__main__":
    main()
