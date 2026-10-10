"""Train Gate-3 dynamic multi-graph residual centre and cache its outputs."""

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

from prwarn.data.processed import load_processed_split
from prwarn.data.torch_dataset import DirectionalGraphCollator, WindowTensorDataset
from prwarn.models.backbone import DynamicMultiGraphResidualForecaster
from prwarn.models.deterministic_baselines import TemporalDeterministicBaseline
from prwarn.training import (
    evaluate_point_epoch,
    initialize_run_directory,
    set_reproducible_seed,
    train_point_epoch,
)


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _metadata(data_dir: Path) -> dict:
    return json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))


def _feature_index(metadata: dict) -> dict[str, int]:
    return {name: int(index) for name, index in metadata["feature_index"].items()}


def _loader(split, wind, coordinates, config, args, *, shuffle: bool) -> DataLoader:
    graph = config["graphs"]
    collator = DirectionalGraphCollator(
        coordinates=coordinates,
        distance_scale=float(graph["direction_distance_scale"]),
        sigma_degrees=float(graph["direction_sigma_degrees"]),
        sector_degrees=float(graph["direction_sector_degrees"]),
    )
    return DataLoader(
        WindowTensorDataset(split, wind),
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        collate_fn=collator,
    )


@torch.no_grad()
def _cache_split(
    model,
    loader,
    path: Path,
    static_graphs: torch.Tensor,
    coordinates: np.ndarray,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    n = len(loader.dataset)
    nodes = model.n_nodes
    hidden_dim = model.forecast_head[0].in_features
    horizon = model.forecast_head[-1].out_features
    graph_count = model.n_static_graphs + 2
    chunks = (min(loader.batch_size or 1, n), nodes, horizon)
    absolute_sum = squared_sum = baseline_absolute = baseline_squared = valid_count = 0.0
    persistence_absolute = persistence_squared = persistence_count = 0.0
    target_sum = target_squared_sum = 0.0
    offset = 0
    started = perf_counter()
    with h5py.File(path, "w") as output:
        output.attrs["rated_power"] = float(model.rated_power)
        output.attrs["n_nodes"] = nodes
        output.attrs["forecast_steps"] = horizon
        output.create_dataset("static_graphs", data=static_graphs.cpu().numpy())
        output.create_dataset("adaptive_graph", data=model.adaptive().cpu().numpy())
        output.create_dataset("coordinates", data=coordinates)
        datasets = {
            "y": output.create_dataset("y", (n, nodes, horizon), dtype="f4", chunks=chunks, compression="lzf"),
            "y_mask": output.create_dataset("y_mask", (n, nodes, horizon), dtype="f4", chunks=chunks, compression="lzf"),
            "p_pc": output.create_dataset("p_pc", (n, nodes, horizon), dtype="f4", chunks=chunks, compression="lzf"),
            "y_det": output.create_dataset("y_det", (n, nodes, horizon), dtype="f4", chunks=chunks, compression="lzf"),
            "hidden": output.create_dataset(
                "hidden", (n, nodes, hidden_dim), dtype="f4",
                chunks=(min(loader.batch_size or 1, n), nodes, hidden_dim), compression="lzf"
            ),
            "graph_weights": output.create_dataset(
                "graph_weights", (n, graph_count), dtype="f4", compression="lzf"
            ),
            "wind_from": output.create_dataset("wind_from", (n, nodes), dtype="f4", compression="lzf"),
            "current_y": output.create_dataset("current_y", (n, nodes), dtype="f4", compression="lzf"),
            "current_y_mask": output.create_dataset("current_y_mask", (n, nodes), dtype="f4", compression="lzf"),
            "origin_time": output.create_dataset("origin_time", (n,), dtype="i8", compression="lzf"),
        }
        for batch in loader:
            size = batch["x"].shape[0]
            values = {key: value.to(device) for key, value in batch.items()}
            prediction = model(
                values["x"], values["mask"], values["delta_t"], static_graphs,
                values["directional_graph"], values["p_pc"],
                values.get("future_weather"), values.get("future_weather_mask")
            )
            end = offset + size
            to_write = {
                "y": batch["y"].numpy(),
                "y_mask": batch["y_mask"].numpy(),
                "p_pc": batch["p_pc"].numpy(),
                "y_det": prediction["y_det"].cpu().numpy(),
                "hidden": prediction["hidden"].cpu().numpy(),
                "graph_weights": prediction["graph_weights"].cpu().numpy(),
                "wind_from": batch["wind_from"].numpy(),
                "current_y": batch["current_y"].numpy(),
                "current_y_mask": batch["current_y_mask"].numpy(),
                "origin_time": batch["origin_time"].numpy(),
            }
            for name, values_np in to_write.items():
                datasets[name][offset:end] = values_np
            target = to_write["y"]
            mask = to_write["y_mask"]
            error = (to_write["y_det"] - target) * mask
            baseline_error = (to_write["p_pc"] - target) * mask
            persistence_mask = mask * to_write["current_y_mask"][:, :, None]
            persistence_error = (
                to_write["current_y"][:, :, None] - target
            ) * persistence_mask
            absolute_sum += np.abs(error).sum()
            squared_sum += np.square(error).sum()
            baseline_absolute += np.abs(baseline_error).sum()
            baseline_squared += np.square(baseline_error).sum()
            persistence_absolute += np.abs(persistence_error).sum()
            persistence_squared += np.square(persistence_error).sum()
            persistence_count += persistence_mask.sum()
            valid_count += mask.sum()
            target_sum += (target * mask).sum()
            target_squared_sum += (np.square(target) * mask).sum()
            offset = end
    if valid_count == 0:
        raise ValueError("cache split has no valid labels")
    result = {
        "mae": float(absolute_sum / valid_count),
        "rmse": float(np.sqrt(squared_sum / valid_count)),
        "p_pc_mae": float(baseline_absolute / valid_count),
        "p_pc_rmse": float(np.sqrt(baseline_squared / valid_count)),
        "cache_seconds": perf_counter() - started,
    }
    total_variation = target_squared_sum - target_sum * target_sum / valid_count
    result["r2_nse"] = (
        float(1.0 - squared_sum / total_variation)
        if total_variation > 0
        else None
    )
    if persistence_count > 0:
        result.update(
            {
                "persistence_mae": float(persistence_absolute / persistence_count),
                "persistence_rmse": float(np.sqrt(persistence_squared / persistence_count)),
                "persistence_valid_count": int(persistence_count),
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/sdwpf_v3_2.yaml"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--experiment-id")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--missing-inputs",
        nargs="*",
        choices=["mask", "delta_t"],
        help="Override model.missing_inputs; pass no values for X-only.",
    )
    parser.add_argument(
        "--architecture",
        choices=["multigraph", "gru", "tcn", "graphwavenet", "agcrn"],
    )
    parser.add_argument("--target-mode", choices=["physics_residual", "direct_power"])
    parser.add_argument(
        "--graph-components",
        nargs="+",
        choices=["geo", "corr", "directional", "adaptive"],
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = _load_config(args.config)
    if args.missing_inputs is not None:
        config["model"]["missing_inputs"] = args.missing_inputs
    if args.architecture is not None:
        config["model"]["architecture"] = args.architecture
    if args.target_mode is not None:
        config["model"]["target_mode"] = args.target_mode
    if args.graph_components is not None:
        config["graphs"]["enabled"] = args.graph_components
    seed = int(args.seed if args.seed is not None else config["experiment"]["seed"])
    config["experiment"]["seed"] = seed
    config["execution"] = {
        "stage": "deterministic",
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "device": args.device,
        "data_dir": str(args.data_dir.resolve()),
        "seed": seed,
        "architecture": config["model"].get("architecture", "multigraph"),
        "target_mode": config["model"].get("target_mode", "physics_residual"),
        "graph_components": list(config["graphs"].get("enabled", [])),
        "missing_inputs": list(config["model"].get("missing_inputs", [])),
    }
    set_reproducible_seed(seed)
    experiment_id = args.experiment_id or f"{config['experiment']['name']}_det_seed{seed}"
    run_dir = initialize_run_directory(args.output_root, experiment_id, config)
    device = torch.device(args.device)

    coordinates = np.load(args.data_dir / "coordinates.npy")
    static_np = np.stack(
        [np.load(args.data_dir / "a_geo.npy"), np.load(args.data_dir / "a_corr.npy")]
    ).astype(np.float32)
    static_graphs = torch.from_numpy(static_np).to(device)
    metadata = _metadata(args.data_dir)
    feature_index = _feature_index(metadata)
    train_split = load_processed_split(args.data_dir / "train.npz")
    val_split = load_processed_split(args.data_dir / "val.npz")
    if not np.array_equal(train_split.turbines, val_split.turbines):
        raise ValueError("node order differs across processed splits")
    train_loader = _loader(
        train_split, train_split.wind_from, coordinates, config, args, shuffle=True
    )
    val_loader = _loader(
        val_split, val_split.wind_from, coordinates, config, args, shuffle=False
    )

    model_config = {
        "n_nodes": train_split.x.shape[1],
        "n_features": train_split.x.shape[-1],
        "forecast_steps": train_split.y.shape[-1],
        "n_static_graphs": static_np.shape[0],
        "hidden_dim": int(config["model"]["hidden_dim"]),
        "temporal_kernel": int(config["model"]["temporal_kernel"]),
        "dropout": float(config["model"]["dropout"]),
        "rated_power": float(metadata["power_curve"]["rated_power"]),
        "missing_inputs": list(config["model"].get("missing_inputs", ["mask", "delta_t"])),
        "target_mode": str(config["model"].get("target_mode", "physics_residual")),
        "future_weather_features": (
            0 if train_split.future_weather is None
            else int(train_split.future_weather.shape[-1])
        ),
    }
    architecture = str(config["model"].get("architecture", "multigraph"))
    if architecture == "multigraph":
        model_config["graph_components"] = list(
            config["graphs"].get(
                "enabled", ["geo", "corr", "directional", "adaptive"]
            )
        )
        model = DynamicMultiGraphResidualForecaster(**model_config).to(device)
    else:
        model = TemporalDeterministicBaseline(
            architecture=architecture, **model_config
        ).to(device)
    huber_delta = model_config["rated_power"] * float(
        config["model"].get("huber_delta_fraction", 0.05)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    history = []
    best, stale = float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_point_epoch(
            model, train_loader, optimizer, static_graphs=static_graphs, device=device,
            loss_name=config["model"]["point_loss"], huber_delta=huber_delta
        )
        val_loss = evaluate_point_epoch(
            model, val_loader, static_graphs=static_graphs, device=device,
            loss_name=config["model"]["point_loss"], huber_delta=huber_delta
        )
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        history.append(row)
        print(json.dumps(row))
        if val_loss < best:
            best, stale = val_loss, 0
            torch.save(
                {"model_state": model.state_dict(), "model_config": model_config,
                 "architecture": architecture, "epoch": epoch,
                 "val_loss": val_loss, "feature_index": feature_index},
                run_dir / "best.pt",
            )
        else:
            stale += 1
            if stale >= args.patience:
                break
    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])

    metrics = {}
    for name in ("train", "val", "calib", "test"):
        split = load_processed_split(args.data_dir / f"{name}.npz")
        loader = _loader(split, split.wind_from, coordinates, config, args, shuffle=False)
        metrics[name] = _cache_split(
            model, loader, run_dir / f"deterministic_{name}.h5", static_graphs, coordinates, device
        )
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
