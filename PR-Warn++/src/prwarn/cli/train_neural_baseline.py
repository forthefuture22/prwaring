"""Train G4 CVAE or G5 DDIM and export compatible Calib/Test scenarios."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import h5py
import numpy as np
import torch
import yaml

from prwarn.cli.train_flow import _fit_error_standardizer, _graph_state, _loader
from prwarn.eval.metrics import crps_ensemble
from prwarn.models.generative_baselines import (
    GraphConditionalDiffusion,
    GraphConditionalVAE,
    GraphVaeCFM,
    cvae_loss,
    sample_cvae,
    sample_vae_cfm,
    vae_cfm_loss,
)
from prwarn.models.scenarios import reconstruct_power_scenarios
from prwarn.training import initialize_run_directory, set_reproducible_seed


def _batch_loss(
    method: str,
    model: torch.nn.Module,
    values: dict[str, torch.Tensor],
    *,
    latent_beta: float,
    flow_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    if method == "cvae":
        loss, parts = cvae_loss(
            model,
            values["normalized_error"],
            values["condition"],
            values["adjacency"],
            values["y_mask"],
            beta=latent_beta,
        )
        return loss, {name: float(value) for name, value in parts.items()}
    if method == "vae_cfm":
        loss, parts = vae_cfm_loss(
            model, values["normalized_error"], values["condition"],
            values["adjacency"], values["y_mask"], beta=latent_beta,
            flow_weight=flow_weight,
        )
        return loss, {name: float(value) for name, value in parts.items()}
    result = model.training_loss(
        values["normalized_error"],
        values["condition"],
        values["adjacency"],
        values["y_mask"],
    )
    return result.loss, {"timestep_mean": float(result.timestep_mean)}


def _run_epoch(
    method: str,
    model: torch.nn.Module,
    loader,
    device: torch.device,
    *,
    latent_beta: float,
    flow_weight: float,
    optimizer: torch.optim.Optimizer | None,
    grad_clip: float = 1.0,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    loss_total = 0.0
    component_total: dict[str, float] = {}
    batch_count = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            values = {key: value.to(device) for key, value in batch.items()}
            loss, components = _batch_loss(
                method, model, values, latent_beta=latent_beta,
                flow_weight=flow_weight,
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            loss_total += float(loss.detach())
            for name, value in components.items():
                component_total[name] = component_total.get(name, 0.0) + value
            batch_count += 1
    if batch_count == 0:
        raise ValueError("empty neural-baseline loader")
    result = {"loss": loss_total / batch_count}
    result.update(
        {name: value / batch_count for name, value in component_total.items()}
    )
    return result


@torch.no_grad()
def _sample_normalized(
    method: str,
    model: torch.nn.Module,
    condition: torch.Tensor,
    adjacency: torch.Tensor,
    *,
    n_scenarios: int,
    sampling_steps: int,
    ddim_eta: float,
    generator: torch.Generator,
) -> torch.Tensor:
    if method == "cvae":
        return sample_cvae(
            model,
            condition,
            adjacency,
            n_scenarios=n_scenarios,
            generator=generator,
        )
    if method == "vae_cfm":
        return sample_vae_cfm(
            model, condition, adjacency, n_scenarios=n_scenarios,
            steps=sampling_steps, generator=generator,
        )
    return model.sample_ddim(
        condition,
        adjacency,
        n_scenarios=n_scenarios,
        sampling_steps=sampling_steps,
        eta=ddim_eta,
        generator=generator,
    )


@torch.no_grad()
def _export_scenarios(
    method: str,
    model: torch.nn.Module,
    loader,
    path: Path,
    mean: np.ndarray,
    scale: np.ndarray,
    *,
    rated_power: float,
    n_scenarios: int,
    sampling_steps: int,
    ddim_eta: float,
    device: torch.device,
    save_full: bool,
    interval_alpha: float,
    export_seed: int,
) -> dict[str, float | int]:
    model.eval()
    n = len(loader.dataset)
    with h5py.File(loader.dataset.path, "r") as source:
        nodes, horizon = source["y"].shape[1:]
    offset = 0
    crps_weighted = 0.0
    valid_count = 0
    started = perf_counter()
    mean_t = torch.from_numpy(mean).to(device)
    scale_t = torch.from_numpy(scale).to(device)
    generator = torch.Generator(device=device)
    generator.manual_seed(export_seed)
    with h5py.File(path, "w") as output:
        output.attrs["n_scenarios"] = n_scenarios
        output.attrs["rated_power"] = rated_power
        output.attrs["n_nodes"] = nodes
        output.attrs["interval_alpha"] = interval_alpha
        output.attrs["scenario_method"] = method
        if method in {"ddim", "vae_cfm"}:
            output.attrs["sampling_steps"] = sampling_steps
        if method == "ddim":
            output.attrs["ddim_eta"] = ddim_eta
        farm_ds = output.create_dataset(
            "farm_scenarios",
            (n, n_scenarios, horizon),
            dtype="f4",
            chunks=(min(loader.batch_size or 1, n), n_scenarios, horizon),
            compression="lzf",
        )
        datasets = {
            "lower": output.create_dataset(
                "lower", (n, horizon), dtype="f4", compression="lzf"
            ),
            "upper": output.create_dataset(
                "upper", (n, horizon), dtype="f4", compression="lzf"
            ),
            "y_farm": output.create_dataset(
                "y_farm", (n, horizon), dtype="f4", compression="lzf"
            ),
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
        for batch in loader:
            size = batch["condition"].shape[0]
            values = {key: value.to(device) for key, value in batch.items()}
            normalized = _sample_normalized(
                method,
                model,
                values["condition"],
                values["adjacency"],
                n_scenarios=n_scenarios,
                sampling_steps=sampling_steps,
                ddim_eta=ddim_eta,
                generator=generator,
            )
            scenarios = reconstruct_power_scenarios(
                normalized,
                values["y_det"],
                mean_t,
                scale_t,
                rated_power=rated_power,
            )
            scenario_np = scenarios.cpu().numpy()
            farm = scenario_np.sum(axis=2).transpose(1, 0, 2)
            truth = batch["y"].numpy()
            mask = batch["y_mask"].numpy()
            farm_mask = mask.astype(bool).all(axis=1).astype(np.float32)
            end = offset + size
            farm_ds[offset:end] = farm
            to_write = {
                "lower": np.quantile(farm, interval_alpha / 2.0, axis=1),
                "upper": np.quantile(farm, 1.0 - interval_alpha / 2.0, axis=1),
                "y_farm": truth.sum(axis=1),
                "y_farm_mask": farm_mask,
                "y_det_farm": batch["y_det"].numpy().sum(axis=1),
                "current_y_farm": batch["current_y"].numpy().sum(axis=1),
                "current_y_farm_mask": batch["current_y_mask"]
                .numpy()
                .astype(bool)
                .all(axis=1),
                "origin_time": batch["origin_time"].numpy(),
            }
            for name, array in to_write.items():
                datasets[name][offset:end] = array
            if full_ds is not None:
                full_ds[offset:end] = scenario_np.transpose(1, 0, 2, 3)
            valid = mask.astype(bool)
            if valid.any():
                count = int(valid.sum())
                crps_weighted += crps_ensemble(truth, scenario_np, mask=valid) * count
                valid_count += count
            offset = end
    if valid_count == 0:
        raise ValueError("scenario export has no valid labels")
    return {
        "crps": float(crps_weighted / valid_count),
        "scenario_export_seconds": perf_counter() - started,
        "samples": n,
        "scenarios": n_scenarios,
        "sampling_steps": sampling_steps if method in {"ddim", "vae_cfm"} else 1,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["cvae", "ddim", "vae_cfm"], required=True)
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
    parser.add_argument("--sampling-steps", type=int)
    parser.add_argument("--save-full-scenarios", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed = int(args.seed if args.seed is not None else config["experiment"]["seed"])
    set_reproducible_seed(seed)
    baseline_config = config["generative_baselines"]
    method_config = baseline_config[args.method]
    scenarios = int(args.scenarios or config["flow"]["scenarios_eval"])
    sampling_steps = int(
        args.sampling_steps or method_config.get("sampling_steps", 1)
    )
    config["execution"] = {
        "stage": "neural_scenario_baseline",
        "method": args.method,
        "seed": seed,
        "device": args.device,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "deterministic_run": str(args.deterministic_run.resolve()),
        "scenarios": scenarios,
        "sampling_steps": sampling_steps,
    }
    experiment_id = args.experiment_id or (
        f"{config['experiment']['name']}_{args.method}_seed{seed}"
    )
    run_dir = initialize_run_directory(args.output_root, experiment_id, config)
    device = torch.device(args.device)
    train_path = args.deterministic_run / "deterministic_train.h5"
    val_path = args.deterministic_run / "deterministic_val.h5"
    mean, scale = _fit_error_standardizer(train_path)
    graph_state = _graph_state(train_path)
    train_loader = _loader(
        train_path, mean, scale, "train", config, args, graph_state, shuffle=True
    )
    val_loader = _loader(
        val_path, mean, scale, "train", config, args, graph_state, shuffle=False
    )
    with h5py.File(train_path, "r") as source:
        condition_dim = int(source["hidden"].shape[-1])
        forecast_steps = int(source["y"].shape[-1])
        if "rated_power" not in source.attrs:
            raise ValueError("deterministic cache lacks rated_power; regenerate it")
        rated_power = float(source.attrs["rated_power"])
    common_model = {
        "forecast_steps": forecast_steps,
        "condition_dim": condition_dim,
        "hidden_dim": int(baseline_config["hidden_dim"]),
        "graph_layers": int(baseline_config["graph_layers"]),
        "dropout": float(config["model"]["dropout"]),
    }
    if args.method == "cvae":
        model_config = {**common_model, "latent_dim": int(method_config["latent_dim"])}
        model = GraphConditionalVAE(**model_config).to(device)
    elif args.method == "vae_cfm":
        model_config = {**common_model, "latent_dim": int(method_config["latent_dim"])}
        model = GraphVaeCFM(**model_config).to(device)
    else:
        model_config = {
            **common_model,
            "diffusion_steps": int(method_config["diffusion_steps"]),
        }
        model = GraphConditionalDiffusion(**model_config).to(device)
    latent_beta = float(method_config.get("beta", 0.0))
    flow_weight = float(method_config.get("flow_weight", 1.0))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    history = []
    best = float("inf")
    stale = 0
    for epoch in range(1, args.epochs + 1):
        train_metrics = _run_epoch(
            args.method,
            model,
            train_loader,
            device,
            latent_beta=latent_beta,
            flow_weight=flow_weight,
            optimizer=optimizer,
        )
        fork_devices = (
            [device.index if device.index is not None else torch.cuda.current_device()]
            if device.type == "cuda"
            else []
        )
        with torch.random.fork_rng(devices=fork_devices):
            torch.manual_seed(seed + 10_000)
            val_metrics = _run_epoch(
                args.method,
                model,
                val_loader,
                device,
                latent_beta=latent_beta,
                flow_weight=flow_weight,
                optimizer=None,
            )
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        print(json.dumps(row))
        if val_metrics["loss"] < best:
            best = val_metrics["loss"]
            stale = 0
            torch.save(
                {
                    "method": args.method,
                    "model_state": model.state_dict(),
                    "model_config": model_config,
                    "error_mean": mean,
                    "error_scale": scale,
                    "epoch": epoch,
                    "val_loss": best,
                },
                run_dir / "best.pt",
            )
        else:
            stale += 1
            if stale >= args.patience:
                break
    (run_dir / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    alpha = float(config["calibration"]["alpha"])
    ddim_eta = float(config["generative_baselines"]["ddim"]["eta"])
    metrics = {}
    for split_index, split in enumerate(("calib", "test")):
        cache_path = args.deterministic_run / f"deterministic_{split}.h5"
        loader = _loader(
            cache_path, mean, scale, "export", config, args, graph_state, shuffle=False
        )
        metrics[split] = _export_scenarios(
            args.method,
            model,
            loader,
            run_dir / f"scenarios_{split}.h5",
            mean,
            scale,
            rated_power=rated_power,
            n_scenarios=scenarios,
            sampling_steps=sampling_steps,
            ddim_eta=ddim_eta,
            device=device,
            save_full=args.save_full_scenarios,
            interval_alpha=alpha,
            export_seed=seed + 20_000 + split_index,
        )
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps({"run_dir": str(run_dir), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
