"""Reusable training steps and reproducibility metadata.

The functions intentionally keep deterministic and probabilistic optimisation
separate, matching the paper's Gate 3 -> Gate 4 sequence.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch import Tensor, nn

from prwarn.models.backbone import masked_huber_loss
from prwarn.models.flow import conditional_flow_matching_loss


def set_reproducible_seed(seed: int, *, deterministic_algorithms: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_algorithms:
        torch.use_deterministic_algorithms(True)


def masked_mae_loss(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    weights = mask.to(prediction.dtype)
    return (torch.abs(prediction - target) * weights).sum() / weights.sum().clamp_min(1.0)


def train_point_epoch(
    model: nn.Module,
    batches: Iterable[Mapping[str, Tensor]],
    optimizer: torch.optim.Optimizer,
    *,
    static_graphs: Tensor,
    device: torch.device,
    loss_name: str = "masked_huber",
    grad_clip: float | None = 1.0,
    huber_delta: float = 1.0,
) -> float:
    """Run one point-forecast epoch on dictionary-like batches."""

    model.train()
    total, count = 0.0, 0
    for batch in batches:
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
        residual_target = (
            values["y"]
            if getattr(model, "target_mode", "physics_residual") == "direct_power"
            else values["y"] - values["p_pc"]
        )
        if loss_name == "masked_huber":
            loss = masked_huber_loss(
                output["residual"], residual_target, values["y_mask"], delta=huber_delta
            )
        elif loss_name == "masked_mae":
            loss = masked_mae_loss(output["residual"], residual_target, values["y_mask"])
        else:
            raise ValueError(f"unknown point loss: {loss_name}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total += float(loss.detach())
        count += 1
    if count == 0:
        raise ValueError("empty point-training loader")
    return total / count


@torch.no_grad()
def evaluate_point_epoch(
    model: nn.Module,
    batches: Iterable[Mapping[str, Tensor]],
    *,
    static_graphs: Tensor,
    device: torch.device,
    loss_name: str = "masked_huber",
    huber_delta: float = 1.0,
) -> float:
    model.eval()
    total, count = 0.0, 0
    for batch in batches:
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
        residual_target = (
            values["y"]
            if getattr(model, "target_mode", "physics_residual") == "direct_power"
            else values["y"] - values["p_pc"]
        )
        if loss_name == "masked_huber":
            loss = masked_huber_loss(
                output["residual"], residual_target, values["y_mask"], delta=huber_delta
            )
        elif loss_name == "masked_mae":
            loss = masked_mae_loss(output["residual"], residual_target, values["y_mask"])
        else:
            raise ValueError(f"unknown point loss: {loss_name}")
        total += float(loss)
        count += 1
    if count == 0:
        raise ValueError("empty point-evaluation loader")
    return total / count


def train_flow_epoch(
    velocity: nn.Module,
    batches: Iterable[Mapping[str, Tensor]],
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    grad_clip: float | None = 1.0,
) -> float:
    """Run one CFM epoch on cached deterministic errors and conditions."""

    velocity.train()
    total, count = 0.0, 0
    for batch in batches:
        values = {key: value.to(device) for key, value in batch.items()}
        loss = conditional_flow_matching_loss(
            velocity,
            values["normalized_error"],
            values["condition"],
            values["adjacency"],
            values["y_mask"],
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(velocity.parameters(), grad_clip)
        optimizer.step()
        total += float(loss.detach())
        count += 1
    if count == 0:
        raise ValueError("empty flow-training loader")
    return total / count


@torch.no_grad()
def evaluate_flow_epoch(
    velocity: nn.Module,
    batches: Iterable[Mapping[str, Tensor]],
    *,
    device: torch.device,
) -> float:
    velocity.eval()
    total, count = 0.0, 0
    for batch in batches:
        values = {key: value.to(device) for key, value in batch.items()}
        loss = conditional_flow_matching_loss(
            velocity,
            values["normalized_error"],
            values["condition"],
            values["adjacency"],
            values["y_mask"],
        )
        total += float(loss)
        count += 1
    if count == 0:
        raise ValueError("empty flow-evaluation loader")
    return total / count


def config_hash(config: Mapping[str, Any]) -> str:
    encoded = json.dumps(config, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def capture_runtime_metadata(config: Mapping[str, Any]) -> dict[str, Any]:
    """Capture the minimum metadata required for every paper run."""

    cuda_name = None
    if torch.cuda.is_available():
        cuda_name = torch.cuda.get_device_name(torch.cuda.current_device())
    return {
        "config_hash": config_hash(config),
        "git_commit": _git_commit(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cuda_device": cuda_name,
        "pid": os.getpid(),
    }


def initialize_run_directory(
    output_root: str | Path,
    experiment_id: str,
    config: Mapping[str, Any],
) -> Path:
    """Create a non-overwriting run directory and write immutable metadata."""

    output = Path(output_root) / experiment_id
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output / "runtime.json").write_text(
        json.dumps(capture_runtime_metadata(config), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return output
