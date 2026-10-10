"""Fit pooled per-horizon CQR on Calib and evaluate frozen marginal Test forecasts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from prwarn.calibration.split import PerHorizonSplitConformal
from prwarn.eval.metrics import interval_metrics, masked_mae, masked_rmse, pinball_loss


def _read(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as source:
        return {
            "levels": source["quantile_levels"][:],
            "quantiles": source["quantiles"][:],
            "y": source["y"][:],
            "mask": source["y_mask"][:],
            "origin_time": source["origin_time"][:],
        }


def _level_index(levels: np.ndarray, target: float) -> int:
    matches = np.flatnonzero(np.isclose(levels, target, atol=1e-7))
    if matches.size != 1:
        raise ValueError(f"required quantile level {target:g} is absent or duplicated")
    return int(matches[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--alpha", type=float, default=0.10)
    args = parser.parse_args()
    output_dir = args.output_dir or args.run / "evaluation_cqr"
    output_dir.mkdir(parents=True, exist_ok=False)
    calib = _read(args.run / "marginal_calib.h5")
    test = _read(args.run / "marginal_test.h5")
    if not np.array_equal(calib["levels"], test["levels"]):
        raise ValueError("Calib/Test quantile levels differ")
    levels = test["levels"]
    lower_index = _level_index(levels, args.alpha / 2.0)
    upper_index = _level_index(levels, 1.0 - args.alpha / 2.0)
    median_index = _level_index(levels, 0.5)
    horizon = calib["y"].shape[-1]
    conformal = PerHorizonSplitConformal(alpha=args.alpha).fit(
        calib["quantiles"][..., lower_index].reshape(-1, horizon),
        calib["quantiles"][..., upper_index].reshape(-1, horizon),
        calib["y"].reshape(-1, horizon),
        calib["mask"].reshape(-1, horizon),
    )
    raw_lower = test["quantiles"][..., lower_index]
    raw_upper = test["quantiles"][..., upper_index]
    lower, upper = conformal.transform(raw_lower, raw_upper)
    truth, mask = test["y"], test["mask"]
    median = test["quantiles"][..., median_index]
    metrics = {
        "alpha": args.alpha,
        "correction_per_horizon": conformal.correction_.tolist(),
        "point": {
            "mae": masked_mae(truth, median, mask),
            "rmse": masked_rmse(truth, median, mask),
        },
        "interval": {
            "overall": interval_metrics(
                truth, lower, upper, alpha=args.alpha, mask=mask
            )
        },
        "pinball": {},
    }
    for forecast_index in range(horizon):
        metrics["interval"][f"h{forecast_index + 1}"] = interval_metrics(
            truth[..., forecast_index], lower[..., forecast_index],
            upper[..., forecast_index], alpha=args.alpha,
            mask=mask[..., forecast_index],
        )
    for level_index, level in enumerate(levels):
        metrics["pinball"][f"q{float(level):g}"] = pinball_loss(
            truth, test["quantiles"][..., level_index], float(level), mask
        )
    with h5py.File(output_dir / "calibrated_test.h5", "w") as output:
        output.create_dataset("lower", data=lower.astype(np.float32), compression="lzf")
        output.create_dataset("upper", data=upper.astype(np.float32), compression="lzf")
        output.create_dataset("y", data=truth, compression="lzf")
        output.create_dataset("y_mask", data=mask, compression="lzf")
        output.create_dataset("origin_time", data=test["origin_time"], compression="lzf")
        output.attrs["alpha"] = args.alpha
        output.attrs["calibration"] = "pooled_turbines_per_horizon_cqr"
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(output_dir), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
