"""Evaluate persistence and physical-curve point references from a frozen cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from prwarn.eval.metrics import masked_mae, masked_r2, masked_rmse


def _score(truth, prediction, mask) -> dict[str, float]:
    return {
        "mae": masked_mae(truth, prediction, mask),
        "rmse": masked_rmse(truth, prediction, mask),
        "r2_nse": masked_r2(truth, prediction, mask),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deterministic-run", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "calib", "test"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with h5py.File(
        args.deterministic_run / f"deterministic_{args.split}.h5", "r"
    ) as source:
        truth = source["y"][:]
        mask = source["y_mask"][:].astype(bool)
        physical = source["p_pc"][:]
        current = source["current_y"][:]
        current_mask = source["current_y_mask"][:].astype(bool)
    persistence = np.broadcast_to(current[..., None], truth.shape)
    persistence_mask = mask & current_mask[..., None]
    report: dict[str, object] = {
        "split": args.split,
        "physical_curve": {"overall": _score(truth, physical, mask)},
        "persistence": {"overall": _score(truth, persistence, persistence_mask)},
    }
    for horizon_index in range(truth.shape[-1]):
        name = f"h{horizon_index + 1}"
        report["physical_curve"][name] = _score(
            truth[..., horizon_index], physical[..., horizon_index],
            mask[..., horizon_index],
        )
        report["persistence"][name] = _score(
            truth[..., horizon_index], persistence[..., horizon_index],
            persistence_mask[..., horizon_index],
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
