"""Train Gate-4 Direct CFM and export Calib/Test scenario artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from prwarn.data.torch_dataset import DeterministicCacheDataset, FlowGraphCollator
from prwarn.eval.metrics import crps_ensemble
from prwarn.models.flow import DirectCFMVelocity, sample_reverse_ode
from prwarn.models.scenarios import reconstruct_power_scenarios
from prwarn.training import (
    evaluate_flow_epoch,
    initialize_run_directory,
    set_reproducible_seed,
    train_flow_epoch,
)


def _fit_error_standardizer(path: Path, chunk_size: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as source:
        horizon = source["y"].shape[-1]
        total = np.zeros(horizon, dtype=np.float64)
        squared = np.zeros(horizon, dtype=np.float64)
        count = np.zeros(horizon, dtype=np.float64)
        for left in range(0, len(source["y"]), chunk_size):
            right = min(left + chunk_size, len(source["y"]))
            error = source["y"][left:right] - source["y_det"][left:right]
            mask = source["y_mask"][left:right]
            total += (error * mask).sum(axis=(0, 1))
            squared += (np.square(error) * mask).sum(axis=(0, 1))
            count += mask.sum(axis=(0, 1))
    if np.any(count == 0):
        raise ValueError("a forecast horizon has no valid Train errors")
    mean = total / count
    variance = np.maximum(squared / count - np.square(mean), 1e-8)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def _graph_state(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as source:
        return source["coordinates"][:], source["static_graphs"][:], source["adaptive_graph"][:]


def _loader(path, mean, scale, mode, config, args, graph_state, *, shuffle):
    coordinates, static, adaptive = graph_state
    graph = config["graphs"]
    collator = FlowGraphCollator(
        coordinates, static, adaptive,
        float(graph["direction_distance_scale"]),
        float(graph["direction_sigma_degrees"]),
        float(graph["direction_sector_degrees"]),
    )
    return DataLoader(
        DeterministicCacheDataset(str(path), error_mean=mean, error_scale=scale, mode=mode),
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        collate_fn=collator,
    )


@torch.no_grad()
def _export_scenarios(
    velocity,
    loader,
    path: Path,
    mean: np.ndarray,
    scale: np.ndarray,
    rated_power: float,
    n_scenarios: int,
    steps: int,
    device: torch.device,
    save_full: bool,
    interval_alpha: float,
    export_seed: int,
) -> dict[str, float]:
    velocity.eval()
    n = len(loader.dataset)
    with h5py.File(loader.dataset.path, "r") as source:
        nodes, horizon = source["y"].shape[1:]
    offset = 0
    crps_weighted = valid_count = 0.0
    started = perf_counter()
    mean_t = torch.from_numpy(mean).to(device)
    scale_t = torch.from_numpy(scale).to(device)
    generator = torch.Generator(device=device)
    generator.manual_seed(export_seed)
    with h5py.File(path, "w") as output:
        output.attrs["n_scenarios"] = n_scenarios
        output.attrs["ode_steps"] = steps
        output.attrs["rated_power"] = rated_power
        output.attrs["n_nodes"] = nodes
        output.attrs["interval_alpha"] = interval_alpha
        output.attrs["scenario_method"] = "direct_cfm"
        farm_ds = output.create_dataset(
            "farm_scenarios", (n, n_scenarios, horizon), dtype="f4",
            chunks=(min(loader.batch_size or 1, n), n_scenarios, horizon), compression="lzf"
        )
        datasets = {
            "lower": output.create_dataset("lower", (n, horizon), dtype="f4", compression="lzf"),
            "upper": output.create_dataset("upper", (n, horizon), dtype="f4", compression="lzf"),
            "y_farm": output.create_dataset("y_farm", (n, horizon), dtype="f4", compression="lzf"),
            "y_farm_mask": output.create_dataset("y_farm_mask", (n, horizon), dtype="f4", compression="lzf"),
            "y_det_farm": output.create_dataset("y_det_farm", (n, horizon), dtype="f4", compression="lzf"),
            "current_y_farm": output.create_dataset("current_y_farm", (n,), dtype="f4", compression="lzf"),
            "current_y_farm_mask": output.create_dataset("current_y_farm_mask", (n,), dtype="f4", compression="lzf"),
            "origin_time": output.create_dataset("origin_time", (n,), dtype="i8", compression="lzf"),
        }
        full_ds = None
        if save_full:
            full_ds = output.create_dataset(
                "turbine_scenarios", (n, n_scenarios, nodes, horizon), dtype="f4",
                chunks=(1, n_scenarios, nodes, horizon), compression="lzf"
            )
        for batch in loader:
            size = batch["condition"].shape[0]
            values = {key: value.to(device) for key, value in batch.items()}
            normalized = sample_reverse_ode(
                velocity, values["condition"], values["adjacency"],
                n_scenarios=n_scenarios, steps=steps, generator=generator
            )
            scenarios = reconstruct_power_scenarios(
                normalized, values["y_det"], mean_t, scale_t, rated_power=rated_power
            )
            scenario_np = scenarios.cpu().numpy()  # [M,B,N,H]
            farm = scenario_np.sum(axis=2).transpose(1, 0, 2)
            y = batch["y"].numpy()
            mask = batch["y_mask"].numpy()
            y_farm = y.sum(axis=1)
            farm_mask = mask.astype(bool).all(axis=1).astype(np.float32)
            y_det_farm = batch["y_det"].numpy().sum(axis=1)
            current = batch["current_y"].numpy().sum(axis=1)
            current_mask = batch["current_y_mask"].numpy().astype(bool).all(axis=1).astype(np.float32)
            lower = np.quantile(farm, interval_alpha / 2.0, axis=1)
            upper = np.quantile(farm, 1.0 - interval_alpha / 2.0, axis=1)
            end = offset + size
            farm_ds[offset:end] = farm
            values_np = {
                "lower": lower, "upper": upper, "y_farm": y_farm,
                "y_farm_mask": farm_mask, "y_det_farm": y_det_farm,
                "current_y_farm": current, "current_y_farm_mask": current_mask,
                "origin_time": batch["origin_time"].numpy(),
            }
            for name, array in values_np.items():
                datasets[name][offset:end] = array
            if full_ds is not None:
                full_ds[offset:end] = scenario_np.transpose(1, 0, 2, 3)
            valid = mask.astype(bool)
            if valid.any():
                score = crps_ensemble(y, scenario_np, mask=valid)
                crps_weighted += score * valid.sum()
                valid_count += valid.sum()
            offset = end
    return {
        "crps": float(crps_weighted / valid_count),
        "scenario_export_seconds": perf_counter() - started,
        "samples": n,
        "scenarios": n_scenarios,
        "ode_steps": steps,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deterministic-run", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/sdwpf_v3_2.yaml"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--experiment-id")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--scenarios", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--coupling", choices=["independent", "optimal_transport"])
    parser.add_argument("--save-full-scenarios", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.coupling is not None:
        config["flow"]["coupling"] = args.coupling
    seed = int(args.seed if args.seed is not None else config["experiment"]["seed"])
    config["experiment"]["seed"] = seed
    config["execution"] = {
        "stage": "flow",
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "device": args.device,
        "deterministic_run": str(args.deterministic_run.resolve()),
        "scenarios_override": args.scenarios,
        "steps_override": args.steps,
        "save_full_scenarios": args.save_full_scenarios,
        "coupling": config["flow"].get("coupling", "independent"),
        "seed": seed,
    }
    set_reproducible_seed(seed)
    experiment_id = args.experiment_id or f"{config['experiment']['name']}_cfm_seed{seed}"
    run_dir = initialize_run_directory(args.output_root, experiment_id, config)
    device = torch.device(args.device)
    train_path = args.deterministic_run / "deterministic_train.h5"
    val_path = args.deterministic_run / "deterministic_val.h5"
    mean, scale = _fit_error_standardizer(train_path)
    graph_state = _graph_state(train_path)
    train_loader = _loader(train_path, mean, scale, "train", config, args, graph_state, shuffle=True)
    val_loader = _loader(val_path, mean, scale, "train", config, args, graph_state, shuffle=False)
    with h5py.File(train_path, "r") as source:
        condition_dim = source["hidden"].shape[-1]
        horizon = source["y"].shape[-1]
        rated_power = float(torch.load(
            args.deterministic_run / "best.pt", map_location="cpu", weights_only=False
        )["model_config"]["rated_power"])
    model_config = {
        "forecast_steps": horizon,
        "condition_dim": condition_dim,
        "hidden_dim": int(config["flow"]["hidden_dim"]),
        "dropout": float(config["model"]["dropout"]),
        "coupling": str(config["flow"].get("coupling", "independent")),
    }
    velocity = DirectCFMVelocity(**model_config).to(device)
    optimizer = torch.optim.AdamW(
        velocity.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    history, best, stale = [], float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_flow_epoch(velocity, train_loader, optimizer, device=device)
        # Fix evaluation randomness so early stopping compares like with like.
        fork_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=fork_devices):
            torch.manual_seed(seed + 10_000)
            val_loss = evaluate_flow_epoch(velocity, val_loader, device=device)
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        history.append(row)
        print(json.dumps(row))
        if val_loss < best:
            best, stale = val_loss, 0
            torch.save(
                {"model_state": velocity.state_dict(), "model_config": model_config,
                 "error_mean": mean, "error_scale": scale, "epoch": epoch, "val_loss": val_loss},
                run_dir / "best.pt",
            )
        else:
            stale += 1
            if stale >= args.patience:
                break
    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    velocity.load_state_dict(checkpoint["model_state"])
    scenarios = args.scenarios or int(config["flow"]["scenarios_eval"])
    steps = args.steps or int(config["flow"]["steps"])
    interval_alpha = float(config["calibration"]["alpha"])
    metrics = {}
    for export_index, name in enumerate(("calib", "test")):
        cache_path = args.deterministic_run / f"deterministic_{name}.h5"
        loader = _loader(cache_path, mean, scale, "export", config, args, graph_state, shuffle=False)
        metrics[name] = _export_scenarios(
            velocity, loader, run_dir / f"scenarios_{name}.h5", mean, scale,
            rated_power, scenarios, steps, device, args.save_full_scenarios, interval_alpha,
            seed + 20_000 + export_index
        )
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
