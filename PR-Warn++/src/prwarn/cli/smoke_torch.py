"""Synthetic server-side tensor smoke test for the neural pipeline."""

from __future__ import annotations

import argparse

import torch

from prwarn.models.backbone import DynamicMultiGraphResidualForecaster, masked_huber_loss
from prwarn.models.flow import DirectCFMVelocity, conditional_flow_matching_loss, sample_reverse_ode
from prwarn.models.generative_baselines import (
    GraphConditionalDiffusion,
    GraphConditionalVAE,
    GraphVaeCFM,
    cvae_loss,
    sample_cvae,
    sample_vae_cfm,
    vae_cfm_loss,
)
from prwarn.models.deterministic_baselines import TemporalDeterministicBaseline
from prwarn.models.marginal_baselines import GraphMarginalForecaster, marginal_loss
from prwarn.models.scenarios import reconstruct_power_scenarios


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(2025)

    batch, nodes, history, features, horizon = 2, 5, 24, 8, 6
    x = torch.randn(batch, nodes, history, features, device=device)
    mask = (torch.rand_like(x) > 0.1).float()
    delta = torch.where(mask.bool(), torch.zeros_like(x), torch.full_like(x, 10.0))
    static = torch.softmax(torch.randn(2, nodes, nodes, device=device), dim=-1)
    directional = torch.softmax(torch.randn(batch, nodes, nodes, device=device), dim=-1)
    p_pc = torch.rand(batch, nodes, horizon, device=device) * 1500.0
    target = torch.rand(batch, nodes, horizon, device=device) * 2000.0
    target_mask = torch.ones_like(target)
    future_weather = torch.randn(batch, nodes, horizon, 3, device=device)
    future_weather_mask = (torch.rand_like(future_weather) > 0.1).float()

    backbone = DynamicMultiGraphResidualForecaster(
        n_nodes=nodes,
        n_features=features,
        forecast_steps=horizon,
        hidden_dim=32,
        rated_power=2000.0,
        future_weather_features=3,
    ).to(device)
    output = backbone(
        x, mask, delta, static, directional, p_pc,
        future_weather, future_weather_mask,
    )
    residual_target = target - p_pc
    point_loss = masked_huber_loss(output["residual"], residual_target, target_mask)
    point_loss.backward()

    flow = DirectCFMVelocity(
        forecast_steps=horizon,
        condition_dim=output["hidden"].shape[-1],
        hidden_dim=64,
    ).to(device)
    error = (target - output["y_det"].detach()) / 200.0
    flow_loss = conditional_flow_matching_loss(
        flow, error, output["hidden"].detach(), output["adjacency"].detach(), target_mask
    )
    flow_loss.backward()
    normalized = sample_reverse_ode(
        flow,
        output["hidden"].detach(),
        output["adjacency"].detach(),
        n_scenarios=4,
        steps=4,
    )
    scenarios = reconstruct_power_scenarios(
        normalized,
        output["y_det"].detach(),
        torch.zeros(horizon, device=device),
        torch.full((horizon,), 200.0, device=device),
        rated_power=2000.0,
    )

    condition = output["hidden"].detach()
    adjacency = output["adjacency"].detach()
    cvae = GraphConditionalVAE(
        forecast_steps=horizon,
        condition_dim=condition.shape[-1],
        latent_dim=8,
        hidden_dim=32,
    ).to(device)
    cvae_total, cvae_parts = cvae_loss(
        cvae, error, condition, adjacency, target_mask, beta=1e-3
    )
    cvae_total.backward()
    cvae_scenarios = sample_cvae(
        cvae, condition, adjacency, n_scenarios=4
    )

    diffusion = GraphConditionalDiffusion(
        forecast_steps=horizon,
        condition_dim=condition.shape[-1],
        hidden_dim=32,
        diffusion_steps=10,
    ).to(device)
    diffusion_result = diffusion.training_loss(
        error, condition, adjacency, target_mask
    )
    diffusion_result.loss.backward()
    ddim_scenarios = diffusion.sample_ddim(
        condition, adjacency, n_scenarios=4, sampling_steps=4
    )
    vae_cfm = GraphVaeCFM(
        forecast_steps=horizon, condition_dim=condition.shape[-1],
        latent_dim=8, hidden_dim=32,
    ).to(device)
    vae_cfm_total, _ = vae_cfm_loss(
        vae_cfm, error, condition, adjacency, target_mask
    )
    vae_cfm_total.backward()
    vae_cfm_scenarios = sample_vae_cfm(
        vae_cfm, condition, adjacency, n_scenarios=4, steps=4
    )
    baseline_shapes = {}
    for architecture in ("gru", "tcn", "graphwavenet", "agcrn"):
        baseline = TemporalDeterministicBaseline(
            architecture=architecture, n_nodes=nodes, n_features=features,
            forecast_steps=horizon, hidden_dim=16, rated_power=2000.0,
            future_weather_features=3,
        ).to(device)
        result = baseline(
            x, mask, delta, static, directional, p_pc,
            future_weather, future_weather_mask,
        )
        baseline_shapes[architecture] = tuple(result["y_det"].shape)
    marginal_losses = {}
    for method in ("gaussian", "student_t", "quantile"):
        marginal = GraphMarginalForecaster(
            method=method, forecast_steps=horizon,
            condition_dim=condition.shape[-1], hidden_dim=16,
        ).to(device)
        marginal_losses[method] = float(
            marginal_loss(marginal, error, condition, adjacency, target_mask).detach()
        )
    print(
        {
            "device": str(device),
            "point_loss": float(point_loss.detach()),
            "flow_loss": float(flow_loss.detach()),
            "cvae_loss": float(cvae_total.detach()),
            "cvae_kl": float(cvae_parts["kl"]),
            "diffusion_loss": float(diffusion_result.loss.detach()),
            "vae_cfm_loss": float(vae_cfm_total.detach()),
            "y_det": tuple(output["y_det"].shape),
            "hidden": tuple(output["hidden"].shape),
            "scenarios": tuple(scenarios.shape),
            "cvae_scenarios": tuple(cvae_scenarios.shape),
            "ddim_scenarios": tuple(ddim_scenarios.shape),
            "vae_cfm_scenarios": tuple(vae_cfm_scenarios.shape),
            "deterministic_baselines": baseline_shapes,
            "marginal_losses": marginal_losses,
            "graph_weights": tuple(output["graph_weights"].shape),
        }
    )


if __name__ == "__main__":
    main()
