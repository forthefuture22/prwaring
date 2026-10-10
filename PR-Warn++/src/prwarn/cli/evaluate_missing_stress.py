"""Evaluate A9 deterministic variants under frozen synthetic missingness stresses."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import yaml

from prwarn.cli.train_deterministic import _loader
from prwarn.data.processed import ProcessedSplit, load_processed_split
from prwarn.data.stress import (
    apply_missingness_stress,
    block_missingness,
    extreme_conditioned_missingness,
    mcar_missingness,
    rebuild_issue_time_derived,
    spatial_outage_missingness,
)
from prwarn.models.scenario_checkpoint import load_deterministic_checkpoint


def _parse_runs(values: list[str]) -> dict[str, Path]:
    runs: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("each --run must be LABEL=RUN_DIRECTORY")
        label, raw_path = value.split("=", 1)
        if not label or label in runs:
            raise ValueError("run labels must be non-empty and unique")
        runs[label] = Path(raw_path)
    return runs


def _train_extreme_threshold(
    train: ProcessedSplit,
    metadata: dict[str, object],
    *,
    quantile: float,
) -> float:
    if not 0.0 < quantile < 1.0:
        raise ValueError("extreme quantile must be in (0, 1)")
    feature_index = {name: int(index) for name, index in metadata["feature_index"].items()}
    speed_name = str(metadata["physical_columns"]["wind_speed"])
    speed_index = feature_index[speed_name]
    scaler = metadata["feature_scaler"]
    raw_speed = (
        train.x[..., speed_index] * float(scaler["std"][speed_index])
        + float(scaler["mean"][speed_index])
    )
    valid = train.mask[..., speed_index].astype(bool) & np.isfinite(raw_speed)
    if not valid.any():
        raise ValueError("Train has no valid wind-speed values for extreme threshold")
    return float(np.quantile(raw_speed[valid], quantile))


def _stress_selections(
    test: ProcessedSplit,
    train: ProcessedSplit,
    metadata: dict[str, object],
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    selections: dict[str, np.ndarray] = {
        "clean": np.zeros_like(test.mask, dtype=bool)
    }
    sequence = 0
    for rate in args.mcar_rates:
        selections[f"mcar_{rate:g}"] = mcar_missingness(
            test.mask, rate=rate, seed=args.seed + sequence
        )
        sequence += 1
    for length in args.block_lengths:
        selections[f"block_{length}"] = block_missingness(
            test.mask,
            block_length=length,
            blocks_per_sample=args.blocks_per_sample,
            seed=args.seed + sequence,
        )
        sequence += 1
    for fraction in args.spatial_fractions:
        selections[f"spatial_{fraction:g}"] = spatial_outage_missingness(
            test.mask,
            node_fraction=fraction,
            duration=args.spatial_duration,
            seed=args.seed + sequence,
        )
        sequence += 1
    threshold = _train_extreme_threshold(
        train, metadata, quantile=args.extreme_quantile
    )
    feature_index = {name: int(index) for name, index in metadata["feature_index"].items()}
    speed_index = feature_index[str(metadata["physical_columns"]["wind_speed"])]
    scaler = metadata["feature_scaler"]
    test_speed = (
        test.x[..., speed_index] * float(scaler["std"][speed_index])
        + float(scaler["mean"][speed_index])
    )
    extreme = (test_speed >= threshold)[..., None]
    selections["extreme_conditioned"] = extreme_conditioned_missingness(
        test.mask,
        extreme,
        base_rate=args.extreme_base_rate,
        extreme_multiplier=args.extreme_multiplier,
        seed=args.seed + sequence,
    )
    args.extreme_threshold = threshold
    return selections


def _build_stressed_split(
    source: ProcessedSplit,
    selected: np.ndarray,
    metadata: dict[str, object],
) -> tuple[ProcessedSplit, dict[str, object]]:
    if not selected.any():
        return source, {"selected_count": 0, "selected_fraction": 0.0}
    result = apply_missingness_stress(
        source.x,
        source.mask,
        source.delta_t,
        selected,
        step_minutes=float(metadata["resolution_minutes"]),
        fill_value=0.0,
        fill_strategy="forward_fill",
    )
    derived = rebuild_issue_time_derived(result.x, result.mask, metadata)
    stressed = replace(
        source,
        x=result.x.astype(np.float32),
        mask=result.mask.astype(np.float32),
        delta_t=result.delta_t.astype(np.float32),
        p_pc=derived.p_pc,
        current_y=derived.current_y,
        current_y_mask=derived.current_y_mask,
        wind_from=derived.wind_from,
    ).validate()
    return stressed, result.metadata


@torch.no_grad()
def _evaluate(
    run_dir: Path,
    split: ProcessedSplit,
    config: dict,
    coordinates: np.ndarray,
    static_graphs: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, object]:
    device = torch.device(args.device)
    model = load_deterministic_checkpoint(run_dir, device)
    loader = _loader(split, split.wind_from, coordinates, config, args, shuffle=False)
    absolute = 0.0
    squared = 0.0
    count = 0.0
    per_origin_squared = []
    for batch in loader:
        values = {key: value.to(device) for key, value in batch.items()}
        output = model(
            values["x"],
            values["mask"],
            values["delta_t"],
            static_graphs,
            values["directional_graph"],
            values["p_pc"],
            values.get("future_weather"),
            values.get("future_weather_mask"),
        )
        error = output["y_det"] - values["y"]
        weights = values["y_mask"]
        absolute += float((error.abs() * weights).sum())
        squared += float((error.square() * weights).sum())
        count += float(weights.sum())
        raw_origin_count = weights.sum(dim=(1, 2))
        valid_origin = raw_origin_count > 0
        origin_mse = (error.square() * weights).sum(dim=(1, 2))
        origin_mse = origin_mse[valid_origin] / raw_origin_count[valid_origin]
        per_origin_squared.append(origin_mse.cpu().numpy())
    if count == 0:
        raise ValueError("Test split has no valid labels")
    nonempty = [values for values in per_origin_squared if values.size]
    if not nonempty:
        raise ValueError("Test split has no origin with a valid label")
    per_origin_rmse = np.sqrt(np.concatenate(nonempty))
    return {
        "mae": absolute / count,
        "rmse": float(np.sqrt(squared / count)),
        "origin_rmse_mean": float(np.mean(per_origin_rmse)),
        "origin_rmse_p95": float(np.quantile(per_origin_rmse, 0.95)),
        "valid_count": int(count),
        "missing_inputs": list(model.missing_inputs),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Repeat LABEL=DETERMINISTIC_RUN for X-only/X+M/X+M+DeltaT.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/sdwpf_v3_2.yaml"))
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--mcar-rates", type=float, nargs="+", default=[0.1, 0.3, 0.5])
    parser.add_argument("--block-lengths", type=int, nargs="+", default=[3, 6, 12])
    parser.add_argument("--blocks-per-sample", type=int, default=16)
    parser.add_argument("--spatial-fractions", type=float, nargs="+", default=[0.1, 0.3])
    parser.add_argument("--spatial-duration", type=int, default=6)
    parser.add_argument("--extreme-quantile", type=float, default=0.95)
    parser.add_argument("--extreme-base-rate", type=float, default=0.05)
    parser.add_argument("--extreme-multiplier", type=float, default=5.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    runs = _parse_runs(args.run)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    metadata = json.loads((args.data_dir / "metadata.json").read_text(encoding="utf-8"))
    if "physical_columns" not in metadata or "resolution_minutes" not in metadata:
        raise ValueError("processed metadata predates stress-safe derived reconstruction")
    train = load_processed_split(args.data_dir / "train.npz")
    test = load_processed_split(args.data_dir / "test.npz")
    selections = _stress_selections(test, train, metadata, args)
    coordinates = np.load(args.data_dir / "coordinates.npy")
    static_np = np.stack(
        [np.load(args.data_dir / "a_geo.npy"), np.load(args.data_dir / "a_corr.npy")]
    ).astype(np.float32)
    static_graphs = torch.from_numpy(static_np).to(torch.device(args.device))
    payload: dict[str, object] = {
        "schema_version": 1,
        "seed": args.seed,
        "data_dir": str(args.data_dir.resolve()),
        "extreme_wind_speed_threshold_train_only": args.extreme_threshold,
        "stress": {},
        "runs": {label: str(path.resolve()) for label, path in runs.items()},
    }
    clean_by_run: dict[str, dict[str, object]] = {}
    for stress_name, selected in selections.items():
        stressed, stress_metadata = _build_stressed_split(test, selected, metadata)
        stress_result: dict[str, object] = {
            "selection": stress_metadata,
            "models": {},
        }
        for label, run_dir in runs.items():
            result = _evaluate(
                run_dir, stressed, config, coordinates, static_graphs, args
            )
            if stress_name == "clean":
                clean_by_run[label] = result
                result["rmse_degradation"] = 0.0
                result["mae_degradation"] = 0.0
            else:
                result["rmse_degradation"] = result["rmse"] - clean_by_run[label]["rmse"]
                result["mae_degradation"] = result["mae"] - clean_by_run[label]["mae"]
            stress_result["models"][label] = result
        payload["stress"][stress_name] = stress_result
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), "stress_count": len(selections)}))


if __name__ == "__main__":
    main()
