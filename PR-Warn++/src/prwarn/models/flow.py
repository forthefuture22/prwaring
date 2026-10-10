"""Direct Conditional Flow Matching on final deterministic forecast errors."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .backbone import normalize_adjacency, propagate_source_to_destination


@dataclass
class ErrorStandardizer:
    """Train-only per-horizon error normalization state."""

    mean: Tensor
    scale: Tensor

    @classmethod
    def fit(cls, error: Tensor, mask: Tensor, eps: float = 1e-6) -> "ErrorStandardizer":
        if error.shape != mask.shape or error.ndim != 3:
            raise ValueError("error and mask must share [B,N,H]")
        weights = mask.to(error.dtype)
        denominator = weights.sum(dim=(0, 1)).clamp_min(1.0)
        mean = (error * weights).sum(dim=(0, 1)) / denominator
        variance = (torch.square(error - mean) * weights).sum(dim=(0, 1)) / denominator
        return cls(mean=mean.detach(), scale=variance.sqrt().clamp_min(eps).detach())

    def transform(self, error: Tensor) -> Tensor:
        return (error - self.mean) / self.scale

    def inverse(self, normalized_error: Tensor) -> Tensor:
        return normalized_error * self.scale + self.mean


class DirectCFMVelocity(nn.Module):
    """Graph-conditioned velocity field for a `[B,N,H]` error state."""

    def __init__(
        self,
        *,
        forecast_steps: int,
        condition_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        coupling: str = "independent",
    ) -> None:
        super().__init__()
        self.forecast_steps = forecast_steps
        if coupling not in {"independent", "optimal_transport"}:
            raise ValueError("coupling must be independent or optimal_transport")
        self.coupling = coupling
        input_dim = forecast_steps + condition_dim + 3
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.self_projection = nn.Linear(hidden_dim, hidden_dim)
        self.neighbour_projection = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Sequential(
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, forecast_steps),
        )

    @staticmethod
    def _time_features(tau: Tensor, nodes: int) -> Tensor:
        tau = tau.reshape(-1, 1, 1)
        features = torch.cat(
            [tau, torch.sin(torch.pi * tau), torch.cos(torch.pi * tau)], dim=-1
        )
        return features.expand(-1, nodes, -1)

    def forward(self, state: Tensor, tau: Tensor, condition: Tensor, adjacency: Tensor) -> Tensor:
        if state.ndim != 3 or state.shape[-1] != self.forecast_steps:
            raise ValueError("state must have [B,N,H]")
        if condition.shape[:2] != state.shape[:2] or adjacency.shape != (*state.shape[:2], state.shape[1]):
            raise ValueError("condition or adjacency shape mismatch")
        features = torch.cat([state, condition, self._time_features(tau, state.shape[1])], dim=-1)
        hidden = F.gelu(self.input_projection(features))
        neighbour = propagate_source_to_destination(normalize_adjacency(adjacency), hidden)
        hidden = self.norm(hidden + F.gelu(self.self_projection(hidden) + self.neighbour_projection(neighbour)))
        return self.output(hidden)


def conditional_flow_matching_loss(
    velocity: DirectCFMVelocity,
    normalized_error: Tensor,
    condition: Tensor,
    adjacency: Tensor,
    mask: Tensor,
) -> Tensor:
    """Standard independent-pair CFM, deliberately not labelled OT-CFM."""

    if normalized_error.shape != mask.shape:
        raise ValueError("normalized_error and mask shapes differ")
    batch = normalized_error.shape[0]
    tau = torch.rand(batch, device=normalized_error.device, dtype=normalized_error.dtype)
    noise = torch.randn_like(normalized_error)
    if velocity.coupling == "optimal_transport":
        try:
            from scipy.optimize import linear_sum_assignment
        except ImportError as exc:
            raise RuntimeError("optimal_transport coupling requires scipy") from exc
        with torch.no_grad():
            target_flat = (normalized_error * mask).reshape(batch, -1)
            noise_flat = noise.reshape(batch, -1)
            cost = torch.cdist(target_flat, noise_flat).square().cpu().numpy()
            row, column = linear_sum_assignment(cost)
            if not torch.equal(
                torch.from_numpy(row), torch.arange(batch, dtype=torch.int64)
            ):
                raise RuntimeError("unexpected OT assignment row ordering")
            column_index = torch.from_numpy(column).to(noise.device)
            noise = noise[column_index]
    tau_view = tau[:, None, None]
    path = (1.0 - tau_view) * normalized_error + tau_view * noise
    target_velocity = noise - normalized_error
    predicted_velocity = velocity(path, tau, condition, adjacency)
    weights = mask.to(normalized_error.dtype)
    squared = torch.square(predicted_velocity - target_velocity)
    return (squared * weights).sum() / weights.sum().clamp_min(1.0)


@torch.no_grad()
def sample_reverse_ode(
    velocity: DirectCFMVelocity,
    condition: Tensor,
    adjacency: Tensor,
    *,
    n_scenarios: int,
    steps: int = 8,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Euler-integrate from Gaussian `tau=1` back to data `tau=0`.

    Returns normalized errors with shape `[M,B,N,H]`.
    """

    if n_scenarios <= 0 or steps <= 0:
        raise ValueError("n_scenarios and steps must be positive")
    batch, nodes, _ = condition.shape
    condition_m = condition.repeat(n_scenarios, 1, 1)
    adjacency_m = adjacency.repeat(n_scenarios, 1, 1)
    state = torch.randn(
        n_scenarios * batch,
        nodes,
        velocity.forecast_steps,
        device=condition.device,
        dtype=condition.dtype,
        generator=generator,
    )
    dt = -1.0 / steps
    for step in range(steps):
        tau_value = 1.0 - step / steps
        tau = torch.full((state.shape[0],), tau_value, device=state.device, dtype=state.dtype)
        state = state + dt * velocity(state, tau, condition_m, adjacency_m)
    return state.reshape(n_scenarios, batch, nodes, velocity.forecast_steps)
