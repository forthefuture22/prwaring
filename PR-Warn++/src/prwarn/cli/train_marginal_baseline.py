"""Train Gaussian, Student-t or quantile marginal residual baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

from prwarn.cli.train_flow import _fit_error_standardizer, _graph_state, _loader
from prwarn.models.marginal_baselines import (
    GraphMarginalForecaster,
    marginal_loss,
    sample_marginal_errors,
)
from prwarn.training import initialize_run_directory, set_reproducible_seed


def _epoch(model, loader, device, optimizer=None) -> float:
    training = optimizer is not None
    model.train(training)
    total = 0.0
    count = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            values = {key: value.to(device) for key, value in batch.items()}
            loss = marginal_loss(
                model,
                values["normalized_error"], values["condition"],
                values["adjacency"], values["y_mask"],
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total += float(loss.detach())
            count += 1
    if count == 0:
        raise ValueError("empty marginal loader")
    return total / count


@torch.no_grad()
def _export(model, loader, path, mean, scale, rated_power, levels, mc_samples, device, seed):
    model.eval()
    n = len(loader.dataset)
    with h5py.File(loader.dataset.path, "r") as source:
        nodes, horizon = source["y"].shape[1:]
    offset = 0
    with h5py.File(path, "w") as output:
        output.attrs["method"] = model.method
        output.create_dataset("quantile_levels", data=np.asarray(levels, dtype=np.float32))
        quantile_ds = output.create_dataset(
            "quantiles", (n, nodes, horizon, len(levels)), dtype="f4", compression="lzf"
        )
        y_ds = output.create_dataset("y", (n, nodes, horizon), dtype="f4", compression="lzf")
        mask_ds = output.create_dataset("y_mask", (n, nodes, horizon), dtype="f4", compression="lzf")
        time_ds = output.create_dataset("origin_time", (n,), dtype="i8", compression="lzf")
        mean_t = torch.from_numpy(mean).to(device)
        scale_t = torch.from_numpy(scale).to(device)
        for batch_index, batch in enumerate(loader):
            size = batch["condition"].shape[0]
            values = {key: value.to(device) for key, value in batch.items()}
            prediction = model(values["condition"], values["adjacency"])
            if model.method == "quantile":
                normalized_quantiles = prediction["quantiles"]
            elif model.method == "gaussian":
                probability = torch.tensor(levels, device=device, dtype=mean_t.dtype)
                z = torch.distributions.Normal(0.0, 1.0).icdf(probability)
                normalized_quantiles = (
                    prediction["location"][..., None]
                    + prediction["scale"][..., None] * z
                )
            else:
                samples = sample_marginal_errors(
                    model, values["condition"], values["adjacency"],
                    n_scenarios=mc_samples, seed=seed + batch_index,
                )
                normalized_quantiles = torch.quantile(
                    samples, torch.tensor(levels, device=device), dim=0
                ).permute(1, 2, 3, 0)
            power_quantiles = (
                values["y_det"][..., None]
                + normalized_quantiles * scale_t[None, None, :, None]
                + mean_t[None, None, :, None]
            ).clamp(0.0, rated_power)
            end = offset + size
            quantile_ds[offset:end] = power_quantiles.cpu().numpy()
            y_ds[offset:end] = batch["y"].numpy()
            mask_ds[offset:end] = batch["y_mask"].numpy()
            time_ds[offset:end] = batch["origin_time"].numpy()
            offset = end


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["gaussian", "student_t", "quantile"], required=True)
    parser.add_argument("--deterministic-run", type=Path, required=True)
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
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--quantiles", type=float, nargs="+", default=[0.05, 0.1, 0.5, 0.9, 0.95])
    parser.add_argument("--student-mc-samples", type=int, default=1000)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed = int(args.seed if args.seed is not None else config["experiment"]["seed"])
    set_reproducible_seed(seed)
    config["execution"] = {
        "stage": "marginal_baseline", "method": args.method, "seed": seed,
        "deterministic_run": str(args.deterministic_run.resolve()),
        "quantiles": args.quantiles,
    }
    run_id = args.experiment_id or f"{config['experiment']['name']}_{args.method}_seed{seed}"
    run_dir = initialize_run_directory(args.output_root, run_id, config)
    device = torch.device(args.device)
    train_path = args.deterministic_run / "deterministic_train.h5"
    mean, scale = _fit_error_standardizer(train_path)
    graph_state = _graph_state(train_path)
    train_loader = _loader(train_path, mean, scale, "train", config, args, graph_state, shuffle=True)
    val_loader = _loader(
        args.deterministic_run / "deterministic_val.h5", mean, scale, "train",
        config, args, graph_state, shuffle=False,
    )
    with h5py.File(train_path, "r") as source:
        condition_dim = int(source["hidden"].shape[-1])
        horizon = int(source["y"].shape[-1])
        rated_power = float(source.attrs["rated_power"])
    model_config = {
        "method": args.method, "forecast_steps": horizon,
        "condition_dim": condition_dim, "hidden_dim": int(config["flow"]["hidden_dim"]),
        "dropout": float(config["model"]["dropout"]), "quantiles": args.quantiles,
    }
    model = GraphMarginalForecaster(**model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    history, best, stale = [], float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        train_loss = _epoch(model, train_loader, device, optimizer)
        with torch.random.fork_rng(devices=[] if device.type == "cpu" else [torch.cuda.current_device()]):
            torch.manual_seed(seed + 10_000)
            val_loss = _epoch(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        history.append(row)
        print(json.dumps(row))
        if val_loss < best:
            best, stale = val_loss, 0
            torch.save(
                {"model_state": model.state_dict(), "model_config": model_config,
                 "error_mean": mean, "error_scale": scale, "val_loss": best},
                run_dir / "best.pt",
            )
        else:
            stale += 1
            if stale >= args.patience:
                break
    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    state = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    for split_index, split in enumerate(("calib", "test")):
        loader = _loader(
            args.deterministic_run / f"deterministic_{split}.h5", mean, scale,
            "export", config, args, graph_state, shuffle=False,
        )
        _export(
            model, loader, run_dir / f"marginal_{split}.h5", mean, scale,
            rated_power, args.quantiles, args.student_mc_samples, device,
            seed + 20_000 + split_index,
        )
    print(json.dumps({"run_dir": str(run_dir), "best_val_loss": best}))


if __name__ == "__main__":
    main()
