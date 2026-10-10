"""Graph-conditioned CVAE and DDIM joint residual scenario baselines."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .backbone import normalize_adjacency, propagate_source_to_destination
from .flow import DirectCFMVelocity, conditional_flow_matching_loss, sample_reverse_ode


class _GraphBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.self_projection = nn.Linear(hidden_dim, hidden_dim)
        self.neighbour_projection = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: Tensor, adjacency: Tensor) -> Tensor:
        neighbour = propagate_source_to_destination(normalize_adjacency(adjacency), hidden)
        update = F.gelu(
            self.self_projection(hidden) + self.neighbour_projection(neighbour)
        )
        return self.norm(hidden + self.dropout(update))


class _GraphNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        hidden_dim: int,
        graph_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [_GraphBlock(hidden_dim, dropout) for _ in range(graph_layers)]
        )
        self.output = nn.Sequential(
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features: Tensor, adjacency: Tensor) -> Tensor:
        if features.ndim != 3 or adjacency.shape != (
            features.shape[0],
            features.shape[1],
            features.shape[1],
        ):
            raise ValueError("features/adjacency shape mismatch")
        hidden = F.gelu(self.input_projection(features))
        for block in self.blocks:
            hidden = block(hidden, adjacency)
        return self.output(hidden)


class GraphConditionalVAE(nn.Module):
    """CVAE baseline over the complete turbine-by-horizon residual field."""

    def __init__(
        self,
        *,
        forecast_steps: int,
        condition_dim: int,
        latent_dim: int = 16,
        hidden_dim: int = 128,
        graph_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if forecast_steps <= 0 or condition_dim <= 0 or latent_dim <= 0:
            raise ValueError("model dimensions must be positive")
        self.forecast_steps = int(forecast_steps)
        self.condition_dim = int(condition_dim)
        self.latent_dim = int(latent_dim)
        self.encoder = _GraphNetwork(
            forecast_steps * 2 + condition_dim,
            latent_dim * 2,
            hidden_dim=hidden_dim,
            graph_layers=graph_layers,
            dropout=dropout,
        )
        self.decoder = _GraphNetwork(
            latent_dim + condition_dim,
            forecast_steps,
            hidden_dim=hidden_dim,
            graph_layers=graph_layers,
            dropout=dropout,
        )

    def encode(
        self,
        normalized_error: Tensor,
        condition: Tensor,
        adjacency: Tensor,
        mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if (
            normalized_error.shape != mask.shape
            or normalized_error.shape[-1] != self.forecast_steps
        ):
            raise ValueError("normalized_error/mask shape mismatch")
        if condition.shape[:2] != normalized_error.shape[:2]:
            raise ValueError("condition shape mismatch")
        features = torch.cat(
            [normalized_error * mask, mask.to(normalized_error.dtype), condition], dim=-1
        )
        parameters = self.encoder(features, adjacency)
        mean, log_variance = parameters.chunk(2, dim=-1)
        return mean, log_variance.clamp(-12.0, 8.0)

    def decode(self, latent: Tensor, condition: Tensor, adjacency: Tensor) -> Tensor:
        if latent.shape[:2] != condition.shape[:2] or latent.shape[-1] != self.latent_dim:
            raise ValueError("latent/condition shape mismatch")
        return self.decoder(torch.cat([latent, condition], dim=-1), adjacency)

    def forward(
        self,
        normalized_error: Tensor,
        condition: Tensor,
        adjacency: Tensor,
        mask: Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> dict[str, Tensor]:
        mean, log_variance = self.encode(normalized_error, condition, adjacency, mask)
        noise = torch.randn(
            mean.shape,
            device=mean.device,
            dtype=mean.dtype,
            generator=generator,
        )
        latent = mean + torch.exp(0.5 * log_variance) * noise
        reconstruction = self.decode(latent, condition, adjacency)
        return {
            "reconstruction": reconstruction,
            "latent_mean": mean,
            "latent_log_variance": log_variance,
        }


def cvae_loss(
    model: GraphConditionalVAE,
    normalized_error: Tensor,
    condition: Tensor,
    adjacency: Tensor,
    mask: Tensor,
    *,
    beta: float = 1e-3,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Masked reconstruction ELBO loss with a reported KL component."""

    if beta < 0:
        raise ValueError("beta must be non-negative")
    output = model(normalized_error, condition, adjacency, mask)
    weights = mask.to(normalized_error.dtype)
    reconstruction = (
        torch.square(output["reconstruction"] - normalized_error) * weights
    ).sum() / weights.sum().clamp_min(1.0)
    kl_per_node = -0.5 * torch.mean(
        1.0
        + output["latent_log_variance"]
        - torch.square(output["latent_mean"])
        - torch.exp(output["latent_log_variance"]),
        dim=-1,
    )
    valid_node = mask.any(dim=-1).to(normalized_error.dtype)
    kl = (kl_per_node * valid_node).sum() / valid_node.sum().clamp_min(1.0)
    total = reconstruction + beta * kl
    return total, {"reconstruction": reconstruction.detach(), "kl": kl.detach()}


@torch.no_grad()
def sample_cvae(
    model: GraphConditionalVAE,
    condition: Tensor,
    adjacency: Tensor,
    *,
    n_scenarios: int,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Draw normalized residual scenarios with shape ``[M,B,N,H]``."""

    if n_scenarios <= 0:
        raise ValueError("n_scenarios must be positive")
    batch, nodes, _ = condition.shape
    condition_m = condition.repeat(n_scenarios, 1, 1)
    adjacency_m = adjacency.repeat(n_scenarios, 1, 1)
    latent = torch.randn(
        (n_scenarios * batch, nodes, model.latent_dim),
        device=condition.device,
        dtype=condition.dtype,
        generator=generator,
    )
    decoded = model.decode(latent, condition_m, adjacency_m)
    return decoded.reshape(n_scenarios, batch, nodes, model.forecast_steps)


class GraphNoisePredictor(nn.Module):
    """Graph-conditioned epsilon predictor used by the DDPM/DDIM baseline."""

    def __init__(
        self,
        *,
        forecast_steps: int,
        condition_dim: int,
        hidden_dim: int = 128,
        graph_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.forecast_steps = int(forecast_steps)
        self.network = _GraphNetwork(
            forecast_steps + condition_dim + 3,
            forecast_steps,
            hidden_dim=hidden_dim,
            graph_layers=graph_layers,
            dropout=dropout,
        )

    @staticmethod
    def _time_features(time: Tensor, nodes: int) -> Tensor:
        time = time.reshape(-1, 1, 1)
        features = torch.cat(
            [time, torch.sin(torch.pi * time), torch.cos(torch.pi * time)], dim=-1
        )
        return features.expand(-1, nodes, -1)

    def forward(
        self,
        noisy_error: Tensor,
        normalized_time: Tensor,
        condition: Tensor,
        adjacency: Tensor,
    ) -> Tensor:
        if noisy_error.ndim != 3 or noisy_error.shape[-1] != self.forecast_steps:
            raise ValueError("noisy_error must have [B,N,H]")
        if condition.shape[:2] != noisy_error.shape[:2]:
            raise ValueError("condition shape mismatch")
        features = torch.cat(
            [
                noisy_error,
                condition,
                self._time_features(normalized_time, noisy_error.shape[1]),
            ],
            dim=-1,
        )
        return self.network(features, adjacency)


@dataclass(frozen=True)
class DiffusionLoss:
    loss: Tensor
    timestep_mean: Tensor


class GraphConditionalDiffusion(nn.Module):
    """Discrete VP diffusion trained with epsilon prediction and sampled by DDIM."""

    def __init__(
        self,
        *,
        forecast_steps: int,
        condition_dim: int,
        hidden_dim: int = 128,
        graph_layers: int = 2,
        dropout: float = 0.1,
        diffusion_steps: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
    ) -> None:
        super().__init__()
        if diffusion_steps < 2:
            raise ValueError("diffusion_steps must be at least 2")
        if not 0.0 < beta_start < beta_end < 1.0:
            raise ValueError("betas must satisfy 0 < beta_start < beta_end < 1")
        self.forecast_steps = int(forecast_steps)
        self.diffusion_steps = int(diffusion_steps)
        self.predictor = GraphNoisePredictor(
            forecast_steps=forecast_steps,
            condition_dim=condition_dim,
            hidden_dim=hidden_dim,
            graph_layers=graph_layers,
            dropout=dropout,
        )
        beta = torch.linspace(beta_start, beta_end, diffusion_steps)
        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_bar", alpha_bar)

    def training_loss(
        self,
        normalized_error: Tensor,
        condition: Tensor,
        adjacency: Tensor,
        mask: Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> DiffusionLoss:
        if (
            normalized_error.shape != mask.shape
            or normalized_error.shape[-1] != self.forecast_steps
        ):
            raise ValueError("normalized_error/mask shape mismatch")
        batch = normalized_error.shape[0]
        timestep = torch.randint(
            0,
            self.diffusion_steps,
            (batch,),
            device=normalized_error.device,
            generator=generator,
        )
        noise = torch.randn(
            normalized_error.shape,
            device=normalized_error.device,
            dtype=normalized_error.dtype,
            generator=generator,
        )
        alpha_bar = self.alpha_bar[timestep].to(normalized_error.dtype)[:, None, None]
        clean = normalized_error * mask
        noisy = torch.sqrt(alpha_bar) * clean + torch.sqrt(1.0 - alpha_bar) * noise
        noisy = noisy * mask
        normalized_time = timestep.to(normalized_error.dtype) / (self.diffusion_steps - 1)
        predicted = self.predictor(noisy, normalized_time, condition, adjacency)
        weights = mask.to(normalized_error.dtype)
        loss = (torch.square(predicted - noise) * weights).sum()
        loss = loss / weights.sum().clamp_min(1.0)
        return DiffusionLoss(loss=loss, timestep_mean=timestep.float().mean().detach())

    @torch.no_grad()
    def sample_ddim(
        self,
        condition: Tensor,
        adjacency: Tensor,
        *,
        n_scenarios: int,
        sampling_steps: int = 20,
        eta: float = 0.0,
        generator: torch.Generator | None = None,
        clip_denoised: float | None = 8.0,
    ) -> Tensor:
        """Sample normalized residuals using a strided DDIM trajectory."""

        if n_scenarios <= 0:
            raise ValueError("n_scenarios must be positive")
        if sampling_steps <= 0 or sampling_steps > self.diffusion_steps:
            raise ValueError("sampling_steps must be in [1, diffusion_steps]")
        if eta < 0:
            raise ValueError("eta must be non-negative")
        batch, nodes, _ = condition.shape
        condition_m = condition.repeat(n_scenarios, 1, 1)
        adjacency_m = adjacency.repeat(n_scenarios, 1, 1)
        state = torch.randn(
            (n_scenarios * batch, nodes, self.forecast_steps),
            device=condition.device,
            dtype=condition.dtype,
            generator=generator,
        )
        timesteps = torch.linspace(
            self.diffusion_steps - 1,
            0,
            sampling_steps,
            device=condition.device,
        ).round().to(torch.long)
        for index, timestep in enumerate(timesteps):
            previous = timesteps[index + 1] if index + 1 < len(timesteps) else None
            normalized_time = torch.full(
                (state.shape[0],),
                float(timestep) / (self.diffusion_steps - 1),
                device=state.device,
                dtype=state.dtype,
            )
            predicted_noise = self.predictor(
                state, normalized_time, condition_m, adjacency_m
            )
            alpha_bar_t = self.alpha_bar[timestep].to(state.dtype)
            predicted_clean = (
                state - torch.sqrt(1.0 - alpha_bar_t) * predicted_noise
            ) / torch.sqrt(alpha_bar_t)
            if clip_denoised is not None:
                predicted_clean = predicted_clean.clamp(-clip_denoised, clip_denoised)
            if previous is None:
                state = predicted_clean
                continue
            alpha_bar_previous = self.alpha_bar[previous].to(state.dtype)
            sigma_variance = (
                ((1.0 - alpha_bar_previous) / (1.0 - alpha_bar_t))
                * (1.0 - alpha_bar_t / alpha_bar_previous)
            )
            sigma = eta * torch.sqrt(torch.clamp(sigma_variance, min=0.0))
            direction_scale = torch.sqrt(
                torch.clamp(1.0 - alpha_bar_previous - sigma.square(), min=0.0)
            )
            if eta > 0:
                extra_noise = torch.randn(
                    state.shape,
                    device=state.device,
                    dtype=state.dtype,
                    generator=generator,
                )
            else:
                extra_noise = torch.zeros_like(state)
            state = (
                torch.sqrt(alpha_bar_previous) * predicted_clean
                + direction_scale * predicted_noise
                + sigma * extra_noise
            )
        return state.reshape(n_scenarios, batch, nodes, self.forecast_steps)


class GraphVaeCFM(nn.Module):
    """VAE reconstruction followed by latent CFM, retained only for A4."""

    def __init__(
        self,
        *,
        forecast_steps: int,
        condition_dim: int,
        latent_dim: int = 16,
        hidden_dim: int = 128,
        graph_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.forecast_steps = forecast_steps
        self.latent_dim = latent_dim
        self.vae = GraphConditionalVAE(
            forecast_steps=forecast_steps, condition_dim=condition_dim,
            latent_dim=latent_dim, hidden_dim=hidden_dim,
            graph_layers=graph_layers, dropout=dropout,
        )
        self.velocity = DirectCFMVelocity(
            forecast_steps=latent_dim, condition_dim=condition_dim,
            hidden_dim=hidden_dim, dropout=dropout,
        )


def vae_cfm_loss(
    model: GraphVaeCFM,
    normalized_error: Tensor,
    condition: Tensor,
    adjacency: Tensor,
    mask: Tensor,
    *,
    beta: float = 1e-3,
    flow_weight: float = 1.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    reconstruction_loss, parts = cvae_loss(
        model.vae, normalized_error, condition, adjacency, mask, beta=beta
    )
    mean, _ = model.vae.encode(normalized_error, condition, adjacency, mask)
    latent_mask = mask.any(dim=-1, keepdim=True).expand_as(mean)
    flow = conditional_flow_matching_loss(
        model.velocity, mean.detach(), condition, adjacency, latent_mask
    )
    total = reconstruction_loss + flow_weight * flow
    return total, {
        "reconstruction": parts["reconstruction"],
        "kl": parts["kl"],
        "latent_flow": flow.detach(),
    }


@torch.no_grad()
def sample_vae_cfm(
    model: GraphVaeCFM,
    condition: Tensor,
    adjacency: Tensor,
    *,
    n_scenarios: int,
    steps: int,
    generator: torch.Generator | None = None,
) -> Tensor:
    latent = sample_reverse_ode(
        model.velocity, condition, adjacency,
        n_scenarios=n_scenarios, steps=steps, generator=generator,
    )
    batch = condition.shape[0]
    decoded = model.vae.decode(
        latent.reshape(n_scenarios * batch, condition.shape[1], model.latent_dim),
        condition.repeat(n_scenarios, 1, 1),
        adjacency.repeat(n_scenarios, 1, 1),
    )
    return decoded.reshape(
        n_scenarios, batch, condition.shape[1], model.forecast_steps
    )
