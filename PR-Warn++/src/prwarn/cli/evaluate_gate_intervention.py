"""Post-hoc gate-intervention evaluation on a trained learned-graph checkpoint.

Gap-3 (C2-6 / C2-1) post-hoc protocol, grounded in Jain & Wallace (2019),
Wiegreffe & Pinter (2019) and Michel et al. (2019).

The trainable variants ``learned`` / ``uniform`` / ``parameter_matched`` are fit by
``train_deterministic`` and registered as experiment A12.  The two remaining
controls are *post-hoc* interventions that act on an already-trained ``learned``
checkpoint and never refit anything:

* ``frozen_mean``: the gate output no longer depends on the input.  We first read
  the learned gate weights across the validation split, average them into a single
  constant weight vector, then re-evaluate with that vector broadcast to every
  sample.  The constant is fitted ONLY on the validation split.
* ``shuffled``: we keep the gate weight marginal distribution exactly but break the
  sample-to-weight correspondence by a fixed-seed permutation of the batch axis.

The intervention is injected through the model's ``_posthoc_weights`` hook on
``DynamicMultiGraphResidualForecaster._mix_graphs``; nothing is retrained and the
validation split is the only split read.

The deterministic gate-3 model emits point forecasts only, so the report contains
RMSE/MAE per mode; CRPS is reported as ``null`` (it belongs to the gate-4
probabilistic layer, which is out of scope for this post-hoc gate diagnostic).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from prwarn.data.processed import load_processed_split
from prwarn.data.torch_dataset import DirectionalGraphCollator, WindowTensorDataset
from prwarn.models.backbone import DynamicMultiGraphResidualForecaster

GATE_MODES = ("frozen_mean", "shuffled")


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _build_loader(split, coordinates, config, batch_size) -> DataLoader:
    graph = config["graphs"]
    collator = DirectionalGraphCollator(
        coordinates=coordinates,
        distance_scale=float(graph["direction_distance_scale"]),
        sigma_degrees=float(graph["direction_sigma_degrees"]),
        sector_degrees=float(graph["direction_sector_degrees"]),
    )
    return DataLoader(
        WindowTensorDataset(split, split.wind_from),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collator,
    )


def _gate_names(n_static_graphs: int) -> list[str]:
    return list(("geo", "corr")[:n_static_graphs]) + ["directional", "adaptive"]


@torch.no_grad()
def _run_loader(
    model,
    loader: DataLoader,
    static_graphs: torch.Tensor,
    device: torch.device,
    override_weights: torch.Tensor | None = None,
) -> dict:
    """Run one forward pass over the (unshuffled) loader.

    When ``override_weights`` is an ``(N, K)`` tensor aligned to the loader order,
    it is sliced per batch and injected through the model's post-hoc hook.
    Returns RMSE / MAE accumulators plus the collected per-sample graph weights.
    """

    model.eval()
    abs_sum = sq_sum = valid_count = 0.0
    weight_chunks: list[torch.Tensor] = []
    offset = 0
    for batch in loader:
        size = batch["x"].shape[0]
        values = {key: value.to(device) for key, value in batch.items()}
        if override_weights is not None:
            model._posthoc_weights = override_weights[offset:offset + size].to(device)
        prediction = model(
            values["x"], values["mask"], values["delta_t"], static_graphs,
            values["directional_graph"], values["p_pc"],
            values.get("future_weather"), values.get("future_weather_mask"),
        )
        # Clear the hook so it never leaks into an unrelated forward pass.
        if override_weights is not None:
            model._posthoc_weights = None
        y_det = prediction["y_det"]
        target = values["y"]
        mask = values["y_mask"]
        error = (y_det - target) * mask
        abs_sum += float(error.abs().sum())
        sq_sum += float(error.square().sum())
        valid_count += float(mask.sum())
        weight_chunks.append(prediction["graph_weights"].detach().cpu())
        offset += size
    if valid_count == 0:
        raise ValueError("split has no valid labels")
    graph_weights = torch.cat(weight_chunks, dim=0)
    return {
        "rmse": float(np.sqrt(sq_sum / valid_count)),
        "mae": float(abs_sum / valid_count),
        "graph_weights": graph_weights,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Path to a learned-gate best.pt checkpoint.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/sdwpf_v3_2.yaml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=GATE_MODES, required=True)
    parser.add_argument("--split", choices=["val"], default="val",
                        help="Only the validation split may be read (C3-5).")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--shuffle-seed", type=int, default=2025)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = _load_config(args.config)
    device = torch.device(args.device)

    coordinates = np.load(args.data_dir / "coordinates.npy")
    static_np = np.stack(
        [np.load(args.data_dir / "a_geo.npy"), np.load(args.data_dir / "a_corr.npy")]
    ).astype(np.float32)
    static_graphs = torch.from_numpy(static_np).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_config = dict(checkpoint["model_config"])
    model = DynamicMultiGraphResidualForecaster(**model_config).to(device)
    model.load_state_dict(checkpoint["model_state"])

    split = load_processed_split(args.data_dir / f"{args.split}.npz")
    loader = _build_loader(split, coordinates, config, args.batch_size)
    gate_names = _gate_names(int(model_config["n_static_graphs"]))

    # Pass 1: learned baseline (hook absent => original behaviour).
    learned = _run_loader(model, loader, static_graphs, device)
    learned_weights = learned["graph_weights"]

    # Build the intervention weight matrix aligned to loader (sample) order.
    if args.mode == "frozen_mean":
        constant = learned_weights.mean(dim=0)
        intervened_weights = constant.unsqueeze(0).expand(
            learned_weights.shape[0], -1
        ).clone()
    else:  # shuffled
        generator = torch.Generator().manual_seed(int(args.shuffle_seed))
        perm = torch.randperm(learned_weights.shape[0], generator=generator)
        intervened_weights = learned_weights[perm]

    # Pass 2: intervened re-evaluation.
    intervened = _run_loader(
        model, loader, static_graphs, device, override_weights=intervened_weights
    )

    def _stats(result: dict) -> dict:
        w = result["graph_weights"]
        return {
            "rmse": result["rmse"],
            "mae": result["mae"],
            "crps": None,
            "graph_weight_mean": [float(v) for v in w.mean(dim=0)],
            "graph_weight_std": [float(v) for v in w.std(dim=0, unbiased=False)],
        }

    report = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "mode": args.mode,
        "shuffle_seed": args.shuffle_seed if args.mode == "shuffled" else None,
        "gate_names": gate_names,
        "learned": _stats(learned),
        "intervention": _stats(intervened),
        "paired_delta_rmse": intervened["rmse"] - learned["rmse"],
        "paired_delta_mae": intervened["mae"] - learned["mae"],
        "notes": (
            "Post-hoc intervention on a learned checkpoint; nothing was retrained. "
            "frozen_mean constant is fitted on the validation split only. "
            "crps is null because the deterministic gate emits point forecasts; "
            "probabilistic CRPS is produced by the downstream gate-4 layer."
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"gate_intervention_{args.mode}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out_path),
                      "learned_rmse": report["learned"]["rmse"],
                      "intervened_rmse": report["intervention"]["rmse"],
                      "delta_rmse": report["paired_delta_rmse"]}, indent=2))


if __name__ == "__main__":
    main()
