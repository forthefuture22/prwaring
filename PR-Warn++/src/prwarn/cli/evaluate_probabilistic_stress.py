"""Regenerate scenarios under A9 stresses and compare probabilistic degradation."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

from prwarn.cli.evaluate_missing_stress import (
    _build_stressed_split,
    _parse_runs,
    _stress_selections,
)
from prwarn.cli.train_deterministic import _loader
from prwarn.data.processed import load_processed_split
from prwarn.eval.metrics import (
    crps_ensemble_values,
    energy_score_values,
    variogram_score_values,
)
from prwarn.eval.statistics import diebold_mariano, event_bootstrap_ci
from prwarn.models.scenario_checkpoint import (
    LoadedScenarioCheckpoint,
    load_deterministic_checkpoint,
    load_scenario_checkpoint,
)


@torch.no_grad()
def _evaluate_run(
    scenario: LoadedScenarioCheckpoint,
    deterministic,
    deterministic_config: dict,
    split,
    coordinates: np.ndarray,
    static_graphs: torch.Tensor,
    args: argparse.Namespace,
    *,
    random_seed: int,
) -> tuple[dict[str, object], np.ndarray]:
    loader = _loader(
        split,
        split.wind_from,
        coordinates,
        deterministic_config,
        args,
        shuffle=False,
    )
    point_squared = 0.0
    point_count = 0
    turbine_crps_sum = 0.0
    turbine_count = 0
    farm_crps_sum = 0.0
    farm_count = 0
    energy_sum = 0.0
    variogram_sum = 0.0
    joint_count = 0
    origin_farm_crps: list[np.ndarray] = []
    for batch_index, batch in enumerate(loader):
        values = {key: value.to(scenario.device) for key, value in batch.items()}
        centre = deterministic(
            values["x"],
            values["mask"],
            values["delta_t"],
            static_graphs,
            values["directional_graph"],
            values["p_pc"],
            values.get("future_weather"),
            values.get("future_weather_mask"),
        )
        power = scenario.sample_power(
            centre["y_det"],
            centre["hidden"],
            centre["adjacency"],
            n_scenarios=args.scenarios,
            seed=random_seed + batch_index,
            sampling_steps=args.sampling_steps,
        )
        truth = values["y"].cpu().numpy().astype(float)
        mask = values["y_mask"].cpu().numpy().astype(bool)
        y_det = centre["y_det"].cpu().numpy().astype(float)
        scenario_np = power.cpu().numpy().astype(float)
        point_squared += float((np.square(y_det - truth) * mask).sum())
        point_count += int(mask.sum())

        turbine_score = crps_ensemble_values(truth, scenario_np)
        turbine_crps_sum += float((turbine_score * mask).sum())
        turbine_count += int(mask.sum())

        farm_truth = truth.sum(axis=1)
        farm_scenarios = scenario_np.sum(axis=2)
        farm_mask = mask.all(axis=1)
        farm_score = crps_ensemble_values(farm_truth, farm_scenarios)
        farm_crps_sum += float((farm_score * farm_mask).sum())
        farm_count += int(farm_mask.sum())
        per_origin_count = farm_mask.sum(axis=1)
        per_origin = np.full(len(truth), np.nan)
        valid_origin = per_origin_count > 0
        per_origin[valid_origin] = (
            (farm_score * farm_mask).sum(axis=1)[valid_origin]
            / per_origin_count[valid_origin]
        )
        origin_farm_crps.append(per_origin)

        complete = farm_mask.all(axis=1)
        if complete.any():
            energy = energy_score_values(
                farm_truth[complete], farm_scenarios[:, complete]
            )
            variogram = variogram_score_values(
                farm_truth[complete], farm_scenarios[:, complete]
            )
            energy_sum += float(energy.sum())
            variogram_sum += float(variogram.sum())
            joint_count += int(complete.sum())
    if point_count == 0 or turbine_count == 0 or farm_count == 0:
        raise ValueError("probabilistic stress evaluation has no valid targets")
    origin_values = np.concatenate(origin_farm_crps)
    finite_origin = origin_values[np.isfinite(origin_values)]
    metrics: dict[str, object] = {
        "method": scenario.method,
        "scenarios": args.scenarios,
        "sampling_steps": (
            args.sampling_steps or scenario.default_sampling_steps
            if scenario.method in {"direct_cfm", "ddim"}
            else 1
        ),
        "point_rmse": float(np.sqrt(point_squared / point_count)),
        "turbine_crps": turbine_crps_sum / turbine_count,
        "farm_crps": farm_crps_sum / farm_count,
        "farm_origin_crps_p95": float(np.quantile(finite_origin, 0.95)),
        "farm_energy_score": energy_sum / joint_count if joint_count else None,
        "farm_variogram_score": variogram_sum / joint_count if joint_count else None,
        "valid_turbine_count": turbine_count,
        "valid_farm_count": farm_count,
        "complete_joint_origins": joint_count,
    }
    return metrics, origin_values


def _degradation_metrics(
    stressed: dict[str, object], clean: dict[str, object]
) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for name in (
        "point_rmse",
        "turbine_crps",
        "farm_crps",
        "farm_energy_score",
        "farm_variogram_score",
    ):
        first, second = stressed[name], clean[name]
        result[name] = (
            float(first) - float(second)
            if first is not None and second is not None
            else None
        )
    return result


def _paired_degradation_report(
    degradation_a: np.ndarray,
    degradation_b: np.ndarray,
    timestamps: np.ndarray,
    args: argparse.Namespace,
    *,
    seed: int,
) -> dict[str, object]:
    valid = np.isfinite(degradation_a) & np.isfinite(degradation_b)
    a, b = degradation_a[valid], degradation_b[valid]
    if a.size == 0:
        raise ValueError("paired stress degradation has no finite origin")
    lags = min(args.hac_lags, len(a) - 1)
    dm = diebold_mariano(a, b, hac_lags=lags)
    groups = timestamps[valid].astype("datetime64[D]").astype(np.int64)
    bootstrap = event_bootstrap_ci(
        a - b,
        groups,
        confidence=args.confidence,
        n_boot=args.bootstrap,
        seed=seed,
    )
    dm_payload = asdict(dm)
    if not np.isfinite(dm.statistic):
        dm_payload["statistic"] = None
    return {
        "mean_degradation_a": float(np.mean(a)),
        "mean_degradation_b": float(np.mean(b)),
        "mean_difference_a_minus_b": float(np.mean(a - b)),
        "count": int(len(a)),
        "dm_hac": dm_payload,
        "day_cluster_bootstrap": asdict(bootstrap),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Repeat LABEL=SCENARIO_RUN for G1--G6 methods.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--scenarios", type=int, default=100)
    parser.add_argument("--sampling-steps", type=int)
    parser.add_argument("--mcar-rates", type=float, nargs="+", default=[0.1, 0.3, 0.5])
    parser.add_argument("--block-lengths", type=int, nargs="+", default=[3, 6, 12])
    parser.add_argument("--blocks-per-sample", type=int, default=16)
    parser.add_argument("--spatial-fractions", type=float, nargs="+", default=[0.1, 0.3])
    parser.add_argument("--spatial-duration", type=int, default=6)
    parser.add_argument("--extreme-quantile", type=float, default=0.95)
    parser.add_argument("--extreme-base-rate", type=float, default=0.05)
    parser.add_argument("--extreme-multiplier", type=float, default=5.0)
    parser.add_argument("--hac-lags", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.scenarios <= 1 or args.hac_lags < 0:
        raise ValueError("scenarios must exceed one and HAC lags must be non-negative")
    if args.sampling_steps is not None and args.sampling_steps <= 0:
        raise ValueError("sampling-steps must be positive when specified")

    run_paths = _parse_runs(args.run)
    metadata = json.loads((args.data_dir / "metadata.json").read_text(encoding="utf-8"))
    if "physical_columns" not in metadata or "resolution_minutes" not in metadata:
        raise ValueError("processed metadata predates stress-safe reconstruction")
    train = load_processed_split(args.data_dir / "train.npz")
    test = load_processed_split(args.data_dir / "test.npz")
    selections = _stress_selections(test, train, metadata, args)
    coordinates = np.load(args.data_dir / "coordinates.npy")
    static_np = np.stack(
        [np.load(args.data_dir / "a_geo.npy"), np.load(args.data_dir / "a_corr.npy")]
    ).astype(np.float32)
    device = torch.device(args.device)
    static_graphs = torch.from_numpy(static_np).to(device)

    loaded = {}
    for label, run_path in run_paths.items():
        scenario = load_scenario_checkpoint(run_path, device)
        deterministic = load_deterministic_checkpoint(
            scenario.deterministic_run, device
        )
        deterministic_config = json.loads(
            (scenario.deterministic_run / "config.json").read_text(encoding="utf-8")
        )
        loaded[label] = (scenario, deterministic, deterministic_config)

    timestamps = test.origin_time.astype("datetime64[ns]")
    clean_metrics: dict[str, dict[str, object]] = {}
    clean_origin: dict[str, np.ndarray] = {}
    payload: dict[str, object] = {
        "schema_version": 1,
        "data_dir": str(args.data_dir.resolve()),
        "seed": args.seed,
        "scenarios": args.scenarios,
        "extreme_wind_speed_threshold_train_only": args.extreme_threshold,
        "runs": {
            label: {
                "scenario_run": str(run_paths[label].resolve()),
                "method": loaded[label][0].method,
                "deterministic_run": str(loaded[label][0].deterministic_run.resolve()),
            }
            for label in loaded
        },
        "stress": {},
    }
    for stress_index, (stress_name, selected) in enumerate(selections.items()):
        stressed_split, selection_metadata = _build_stressed_split(
            test, selected, metadata
        )
        stress_models = {}
        origin_by_run = {}
        for label, (scenario, deterministic, deterministic_config) in loaded.items():
            metrics, origin_crps = _evaluate_run(
                scenario,
                deterministic,
                deterministic_config,
                stressed_split,
                coordinates,
                static_graphs,
                args,
                # Common random numbers isolate the effect of the stressed
                # condition/centre from Monte Carlo resampling noise.
                random_seed=args.seed,
            )
            origin_by_run[label] = origin_crps
            if stress_name == "clean":
                clean_metrics[label] = metrics
                clean_origin[label] = origin_crps
                metrics["degradation_from_clean"] = {
                    name: 0.0
                    for name in (
                        "point_rmse",
                        "turbine_crps",
                        "farm_crps",
                        "farm_energy_score",
                        "farm_variogram_score",
                    )
                }
            else:
                metrics["degradation_from_clean"] = _degradation_metrics(
                    metrics, clean_metrics[label]
                )
            stress_models[label] = metrics

        paired = {}
        if stress_name != "clean":
            for pair_index, (label_a, label_b) in enumerate(combinations(loaded, 2)):
                degradation_a = origin_by_run[label_a] - clean_origin[label_a]
                degradation_b = origin_by_run[label_b] - clean_origin[label_b]
                paired[f"{label_a}_vs_{label_b}"] = _paired_degradation_report(
                    degradation_a,
                    degradation_b,
                    timestamps,
                    args,
                    seed=args.seed + stress_index * 100 + pair_index,
                )
        payload["stress"][stress_name] = {
            "selection": selection_metadata,
            "models": stress_models,
            "paired_farm_crps_degradation": paired,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "stress_count": len(selections),
                "run_count": len(loaded),
            }
        )
    )


if __name__ == "__main__":
    main()
