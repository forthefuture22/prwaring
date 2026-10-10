"""Unified loading and sampling for residual and neural scenario runs."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from prwarn.baselines.probabilistic import load_probabilistic_baseline

from .backbone import DynamicMultiGraphResidualForecaster
from .flow import DirectCFMVelocity, sample_reverse_ode
from .deterministic_baselines import TemporalDeterministicBaseline
from .generative_baselines import (
    GraphConditionalDiffusion,
    GraphConditionalVAE,
    GraphVaeCFM,
    sample_cvae,
    sample_vae_cfm,
)
from .scenarios import reconstruct_power_scenarios


@dataclass
class LoadedScenarioCheckpoint:
    method: str
    run_dir: Path
    deterministic_run: Path
    rated_power: float
    model: object
    device: torch.device
    error_mean: np.ndarray | None = None
    error_scale: np.ndarray | None = None
    default_sampling_steps: int = 1
    ddim_eta: float = 0.0

    @torch.no_grad()
    def sample_power(
        self,
        y_det: Tensor,
        condition: Tensor,
        adjacency: Tensor,
        *,
        n_scenarios: int,
        seed: int,
        sampling_steps: int | None = None,
    ) -> Tensor:
        """Generate bounded power scenarios with shape ``[M,B,N,H]``."""

        if n_scenarios <= 0:
            raise ValueError("n_scenarios must be positive")
        if self.method in {
            "gaussian_residual",
            "residual_bootstrap",
            "gaussian_copula",
        }:
            scenario = self.model.sample(
                y_det.detach().cpu().numpy(),
                n_scenarios,
                rated_power=self.rated_power,
                seed=seed,
            )
            return torch.from_numpy(scenario.astype(np.float32)).to(self.device)

        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)
        steps = int(sampling_steps or self.default_sampling_steps)
        if self.method == "direct_cfm":
            normalized = sample_reverse_ode(
                self.model,
                condition,
                adjacency,
                n_scenarios=n_scenarios,
                steps=steps,
                generator=generator,
            )
        elif self.method == "cvae":
            normalized = sample_cvae(
                self.model,
                condition,
                adjacency,
                n_scenarios=n_scenarios,
                generator=generator,
            )
        elif self.method == "vae_cfm":
            normalized = sample_vae_cfm(
                self.model,
                condition,
                adjacency,
                n_scenarios=n_scenarios,
                steps=steps,
                generator=generator,
            )
        elif self.method == "ddim":
            normalized = self.model.sample_ddim(
                condition,
                adjacency,
                n_scenarios=n_scenarios,
                sampling_steps=steps,
                eta=self.ddim_eta,
                generator=generator,
            )
        else:
            raise ValueError(f"unsupported scenario method: {self.method}")
        mean = torch.from_numpy(self.error_mean).to(self.device)
        scale = torch.from_numpy(self.error_scale).to(self.device)
        return reconstruct_power_scenarios(
            normalized,
            y_det,
            mean,
            scale,
            rated_power=self.rated_power,
        )


def load_deterministic_checkpoint(
    run_dir: str | Path, device: torch.device
) -> DynamicMultiGraphResidualForecaster:
    path = Path(run_dir)
    checkpoint = torch.load(path / "best.pt", map_location=device, weights_only=False)
    architecture = str(checkpoint.get("architecture", "multigraph"))
    if architecture == "multigraph":
        model = DynamicMultiGraphResidualForecaster(**checkpoint["model_config"]).to(device)
    else:
        model = TemporalDeterministicBaseline(
            architecture=architecture, **checkpoint["model_config"]
        ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def load_scenario_checkpoint(
    run_dir: str | Path, device: torch.device
) -> LoadedScenarioCheckpoint:
    """Load any implemented G1--G6 scenario run and its immutable metadata."""

    path = Path(run_dir)
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    state_path = path / "baseline_state.npz"
    if state_path.exists():
        method = str(config["method"])
        deterministic_run = Path(config["deterministic_run"])
        model = load_probabilistic_baseline(state_path)
        deterministic_state = torch.load(
            deterministic_run / "best.pt", map_location="cpu", weights_only=False
        )
        rated_power = float(deterministic_state["model_config"]["rated_power"])
        return LoadedScenarioCheckpoint(
            method=method,
            run_dir=path,
            deterministic_run=deterministic_run,
            rated_power=rated_power,
            model=model,
            device=device,
        )

    checkpoint = torch.load(path / "best.pt", map_location=device, weights_only=False)
    execution = dict(config["execution"])
    deterministic_run = Path(execution["deterministic_run"])
    deterministic_state = torch.load(
        deterministic_run / "best.pt", map_location="cpu", weights_only=False
    )
    rated_power = float(deterministic_state["model_config"]["rated_power"])
    method = str(checkpoint.get("method", "direct_cfm"))
    model_config = dict(checkpoint["model_config"])
    if method == "direct_cfm":
        model = DirectCFMVelocity(**model_config).to(device)
        default_steps = int(execution.get("steps_override") or config["flow"]["steps"])
        ddim_eta = 0.0
    elif method == "cvae":
        model = GraphConditionalVAE(**model_config).to(device)
        default_steps = 1
        ddim_eta = 0.0
    elif method == "vae_cfm":
        model = GraphVaeCFM(**model_config).to(device)
        default_steps = int(execution["sampling_steps"])
        ddim_eta = 0.0
    elif method == "ddim":
        model = GraphConditionalDiffusion(**model_config).to(device)
        default_steps = int(execution["sampling_steps"])
        ddim_eta = float(config["generative_baselines"]["ddim"]["eta"])
    else:
        raise ValueError(f"unknown checkpoint method: {method}")
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return LoadedScenarioCheckpoint(
        method=method,
        run_dir=path,
        deterministic_run=deterministic_run,
        rated_power=rated_power,
        model=model,
        device=device,
        error_mean=np.asarray(checkpoint["error_mean"], dtype=np.float32),
        error_scale=np.asarray(checkpoint["error_scale"], dtype=np.float32),
        default_sampling_steps=default_steps,
        ddim_eta=ddim_eta,
    )
