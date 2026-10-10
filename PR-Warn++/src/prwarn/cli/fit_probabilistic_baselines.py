"""Fit Train-only residual baselines and export Calib/Test scenario artifacts."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from time import perf_counter

import h5py
import numpy as np
import yaml

from prwarn.baselines.probabilistic import (
    GaussianCopulaResidual,
    GaussianResidual,
    ResidualBootstrap,
    save_probabilistic_baseline,
)
from prwarn.eval.metrics import crps_ensemble


METHODS = {
    "gaussian_residual": GaussianResidual,
    "residual_bootstrap": ResidualBootstrap,
    "gaussian_copula": GaussianCopulaResidual,
}


def _training_errors(
    path: Path, *, maximum_origins: int | None, seed: int
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    with h5py.File(path, "r") as source:
        count = len(source["y"])
        if maximum_origins is not None and maximum_origins <= 0:
            raise ValueError("fit-max-origins must be positive when specified")
        if maximum_origins is not None and maximum_origins < count:
            rng = np.random.default_rng(seed)
            index = np.sort(rng.choice(count, size=maximum_origins, replace=False))
            truth = source["y"][index]
            centre = source["y_det"][index]
            mask = source["y_mask"][index]
            sampled = True
        else:
            truth = source["y"][:]
            centre = source["y_det"][:]
            mask = source["y_mask"][:]
            sampled = False
    return truth - centre, mask, {
        "available_train_origins": count,
        "used_train_origins": int(len(truth)),
        "subsampled": sampled,
        "subsample_seed": int(seed) if sampled else None,
    }


def _create_scenario_artifact(
    model,
    source_path: Path,
    output_path: Path,
    *,
    method: str,
    n_scenarios: int,
    rated_power: float,
    alpha: float,
    chunk_size: int,
    seed: int,
    save_full: bool,
) -> dict[str, float | int]:
    started = perf_counter()
    crps_total = 0.0
    valid_total = 0
    with h5py.File(source_path, "r") as source, h5py.File(output_path, "w") as output:
        n, nodes, horizon = source["y"].shape
        output.attrs["n_scenarios"] = n_scenarios
        output.attrs["rated_power"] = rated_power
        output.attrs["n_nodes"] = nodes
        output.attrs["interval_alpha"] = alpha
        output.attrs["scenario_method"] = method
        output.attrs["random_seed"] = seed
        farm_ds = output.create_dataset(
            "farm_scenarios",
            (n, n_scenarios, horizon),
            dtype="f4",
            chunks=(min(chunk_size, n), n_scenarios, horizon),
            compression="lzf",
        )
        datasets = {
            "lower": output.create_dataset("lower", (n, horizon), dtype="f4", compression="lzf"),
            "upper": output.create_dataset("upper", (n, horizon), dtype="f4", compression="lzf"),
            "y_farm": output.create_dataset("y_farm", (n, horizon), dtype="f4", compression="lzf"),
            "y_farm_mask": output.create_dataset(
                "y_farm_mask", (n, horizon), dtype="f4", compression="lzf"
            ),
            "y_det_farm": output.create_dataset(
                "y_det_farm", (n, horizon), dtype="f4", compression="lzf"
            ),
            "current_y_farm": output.create_dataset(
                "current_y_farm", (n,), dtype="f4", compression="lzf"
            ),
            "current_y_farm_mask": output.create_dataset(
                "current_y_farm_mask", (n,), dtype="f4", compression="lzf"
            ),
            "origin_time": output.create_dataset(
                "origin_time", (n,), dtype="i8", compression="lzf"
            ),
        }
        full_ds = None
        if save_full:
            full_ds = output.create_dataset(
                "turbine_scenarios",
                (n, n_scenarios, nodes, horizon),
                dtype="f4",
                chunks=(1, n_scenarios, nodes, horizon),
                compression="lzf",
            )
        for chunk_index, left in enumerate(range(0, n, chunk_size)):
            right = min(left + chunk_size, n)
            centre = source["y_det"][left:right]
            scenarios = model.sample(
                centre,
                n_scenarios,
                rated_power=rated_power,
                seed=seed + chunk_index,
            ).astype(np.float32)
            farm = scenarios.sum(axis=2).transpose(1, 0, 2)
            truth = source["y"][left:right]
            mask = source["y_mask"][left:right]
            farm_mask = mask.astype(bool).all(axis=1).astype(np.float32)
            lower = np.quantile(farm, alpha / 2.0, axis=1)
            upper = np.quantile(farm, 1.0 - alpha / 2.0, axis=1)
            farm_ds[left:right] = farm
            values = {
                "lower": lower,
                "upper": upper,
                "y_farm": truth.sum(axis=1),
                "y_farm_mask": farm_mask,
                "y_det_farm": centre.sum(axis=1),
                "current_y_farm": source["current_y"][left:right].sum(axis=1),
                "current_y_farm_mask": source["current_y_mask"][left:right]
                .astype(bool)
                .all(axis=1),
                "origin_time": source["origin_time"][left:right],
            }
            for name, value in values.items():
                datasets[name][left:right] = value
            if full_ds is not None:
                full_ds[left:right] = scenarios.transpose(1, 0, 2, 3)
            valid = mask.astype(bool)
            if valid.any():
                valid_count = int(valid.sum())
                crps_total += crps_ensemble(truth, scenarios, mask=valid) * valid_count
                valid_total += valid_count
    if valid_total == 0:
        raise ValueError(f"{source_path} contains no valid targets")
    return {
        "crps": float(crps_total / valid_total),
        "scenario_export_seconds": perf_counter() - started,
        "samples": n,
        "scenarios": n_scenarios,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deterministic-run", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/sdwpf_v3_2.yaml"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--experiment-id")
    parser.add_argument("--methods", nargs="+", choices=sorted(METHODS), default=sorted(METHODS))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--scenarios", type=int)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument(
        "--fit-max-origins",
        type=int,
        help="Optional Train-only subsample cap; the chosen indices are recorded.",
    )
    parser.add_argument("--save-full-scenarios", action="store_true")
    args = parser.parse_args()
    if args.chunk_size <= 0:
        raise ValueError("chunk-size must be positive")

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed = int(args.seed if args.seed is not None else config["experiment"]["seed"])
    n_scenarios = int(args.scenarios or config["flow"]["scenarios_eval"])
    if n_scenarios <= 0:
        raise ValueError("scenarios must be positive")
    alpha = float(config["calibration"]["alpha"])
    train_path = args.deterministic_run / "deterministic_train.h5"
    with h5py.File(train_path, "r") as source:
        if "rated_power" not in source.attrs:
            raise ValueError(
                "deterministic cache lacks rated_power; regenerate it with train_deterministic"
            )
        rated_power = float(source.attrs["rated_power"])
    errors, mask, fit_data = _training_errors(
        train_path, maximum_origins=args.fit_max_origins, seed=seed
    )

    base_id = args.experiment_id or f"{config['experiment']['name']}_baseline_seed{seed}"
    for method_index, method in enumerate(args.methods):
        run_dir = args.output_root / f"{base_id}_{method}"
        run_dir.mkdir(parents=True, exist_ok=False)
        fit_started = perf_counter()
        model = METHODS[method]().fit(errors, mask)
        fit_seconds = perf_counter() - fit_started
        save_probabilistic_baseline(model, run_dir / "baseline_state.npz")
        run_config = {
            "schema_version": 1,
            "method": method,
            "seed": seed,
            "n_scenarios": n_scenarios,
            "interval_alpha": alpha,
            "deterministic_run": str(args.deterministic_run.resolve()),
            "fit_data": fit_data,
            "save_full_scenarios": args.save_full_scenarios,
            "evidence_status": "completed_artifact_requires_evaluation",
        }
        (run_dir / "config.json").write_text(
            json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        runtime = {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "h5py": h5py.__version__,
        }
        (run_dir / "runtime.json").write_text(
            json.dumps(runtime, indent=2), encoding="utf-8"
        )
        metrics: dict[str, object] = {"fit_seconds": fit_seconds, "fit_data": fit_data}
        for split_index, split in enumerate(("calib", "test")):
            metrics[split] = _create_scenario_artifact(
                model,
                args.deterministic_run / f"deterministic_{split}.h5",
                run_dir / f"scenarios_{split}.h5",
                method=method,
                n_scenarios=n_scenarios,
                rated_power=rated_power,
                alpha=alpha,
                chunk_size=args.chunk_size,
                seed=seed + 10_000 * (method_index + 1) + split_index,
                save_full=args.save_full_scenarios,
            )
        (run_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
        print(json.dumps({"run_dir": str(run_dir), "metrics": metrics}, ensure_ascii=False))


if __name__ == "__main__":
    main()
