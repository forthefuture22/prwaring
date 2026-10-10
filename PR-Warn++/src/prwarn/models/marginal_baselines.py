"""Graph-conditioned marginal Gaussian, Student-t and quantile baselines."""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .backbone import normalize_adjacency, propagate_source_to_destination


class GraphMarginalForecaster(nn.Module):
    def __init__(
        self,
        *,
        method: str,
        forecast_steps: int,
        condition_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        quantiles: tuple[float, ...] | list[float] = (0.05, 0.5, 0.95),
    ) -> None:
        super().__init__()
        if method not in {"gaussian", "student_t", "quantile"}:
            raise ValueError("invalid marginal method")
        levels = tuple(float(value) for value in quantiles)
        if sorted(levels) != list(levels) or len(set(levels)) != len(levels):
            raise ValueError("quantiles must be unique and increasing")
        if any(value <= 0.0 or value >= 1.0 for value in levels):
            raise ValueError("quantiles must lie in (0,1)")
        self.method = method
        self.forecast_steps = forecast_steps
        self.quantiles = levels
        self.input_projection = nn.Linear(condition_dim, hidden_dim)
        self.self_projection = nn.Linear(hidden_dim, hidden_dim)
        self.neighbour_projection = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        multiplier = 2 if method == "gaussian" else 3 if method == "student_t" else len(levels)
        self.output = nn.Sequential(
            nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, forecast_steps * multiplier)
        )

    def forward(self, condition: Tensor, adjacency: Tensor) -> dict[str, Tensor]:
        hidden = F.gelu(self.input_projection(condition))
        neighbour = propagate_source_to_destination(normalize_adjacency(adjacency), hidden)
        hidden = self.norm(
            hidden + F.gelu(self.self_projection(hidden) + self.neighbour_projection(neighbour))
        )
        raw = self.output(hidden)
        batch, nodes = condition.shape[:2]
        if self.method == "gaussian":
            raw = raw.reshape(batch, nodes, self.forecast_steps, 2)
            return {"location": raw[..., 0], "scale": F.softplus(raw[..., 1]) + 1e-4}
        if self.method == "student_t":
            raw = raw.reshape(batch, nodes, self.forecast_steps, 3)
            return {
                "location": raw[..., 0],
                "scale": F.softplus(raw[..., 1]) + 1e-4,
                "degrees_freedom": F.softplus(raw[..., 2]) + 2.01,
            }
        raw = raw.reshape(batch, nodes, self.forecast_steps, len(self.quantiles))
        return {"quantiles": torch.sort(raw, dim=-1).values}


def marginal_loss(
    model: GraphMarginalForecaster,
    normalized_error: Tensor,
    condition: Tensor,
    adjacency: Tensor,
    mask: Tensor,
) -> Tensor:
    output = model(condition, adjacency)
    weights = mask.to(normalized_error.dtype)
    if model.method == "gaussian":
        variance = output["scale"].square()
        element = 0.5 * (
            torch.log(2.0 * torch.pi * variance)
            + torch.square(normalized_error - output["location"]) / variance
        )
    elif model.method == "student_t":
        distribution = torch.distributions.StudentT(
            output["degrees_freedom"], output["location"], output["scale"]
        )
        element = -distribution.log_prob(normalized_error)
    else:
        levels = torch.tensor(
            model.quantiles, device=normalized_error.device, dtype=normalized_error.dtype
        )
        difference = normalized_error[..., None] - output["quantiles"]
        element = torch.maximum(levels * difference, (levels - 1.0) * difference).mean(-1)
    return (element * weights).sum() / weights.sum().clamp_min(1.0)


@torch.no_grad()
def sample_marginal_errors(
    model: GraphMarginalForecaster,
    condition: Tensor,
    adjacency: Tensor,
    *,
    n_scenarios: int,
    seed: int,
) -> Tensor:
    """Sample independent marginal errors; this is not a joint-dependence model."""

    if n_scenarios <= 0:
        raise ValueError("n_scenarios must be positive")
    output = model(condition, adjacency)
    generator = torch.Generator(device=condition.device)
    generator.manual_seed(seed)
    if model.method == "gaussian":
        noise = torch.randn(
            (n_scenarios,) + output["location"].shape,
            device=condition.device, dtype=condition.dtype, generator=generator,
        )
        return output["location"][None] + output["scale"][None] * noise
    if model.method == "student_t":
        devices = (
            [condition.device.index if condition.device.index is not None else torch.cuda.current_device()]
            if condition.device.type == "cuda"
            else []
        )
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            distribution = torch.distributions.StudentT(
                output["degrees_freedom"], output["location"], output["scale"]
            )
            return distribution.sample((n_scenarios,))
    uniform = torch.rand(
        (n_scenarios,) + output["quantiles"].shape[:-1],
        device=condition.device, dtype=condition.dtype, generator=generator,
    )
    levels = torch.tensor(model.quantiles, device=condition.device, dtype=condition.dtype)
    index = torch.searchsorted(levels, uniform).clamp(1, len(levels) - 1)
    lower_index = index - 1
    upper_index = index
    expanded = output["quantiles"][None].expand(n_scenarios, -1, -1, -1, -1)
    lower = torch.gather(expanded, -1, lower_index[..., None]).squeeze(-1)
    upper = torch.gather(expanded, -1, upper_index[..., None]).squeeze(-1)
    lower_level = levels[lower_index]
    upper_level = levels[upper_index]
    fraction = (uniform - lower_level) / (upper_level - lower_level).clamp_min(1e-8)
    return lower + fraction * (upper - lower)
