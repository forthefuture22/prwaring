"""Dynamic multi-graph deterministic residual forecaster.

This is the v3.2 point-forecast layer.  It predicts `R = Y - P_pc`; it does not
mix point, flow, conformal, or risk losses.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def normalize_adjacency(adjacency: Tensor, eps: float = 1e-8) -> Tensor:
    return adjacency / adjacency.sum(dim=-1, keepdim=True).clamp_min(eps)


def propagate_source_to_destination(adjacency: Tensor, hidden: Tensor) -> Tensor:
    """Aggregate `A[source,destination]` messages at each destination node."""

    return torch.einsum("bji,bjd->bid", adjacency, hidden)


def mix_graphs_from_weights(
    static_graphs: Tensor,
    directional_graph: Tensor,
    adaptive_graph: Tensor,
    weights: Tensor,
) -> Tensor:
    """Reconstruct the gated adjacency from cached deterministic outputs."""

    if static_graphs.ndim != 3 or directional_graph.ndim != 3 or adaptive_graph.ndim != 2:
        raise ValueError("graph ranks must be [K,N,N], [B,N,N], and [N,N]")
    batch, nodes, other = directional_graph.shape
    if nodes != other or static_graphs.shape[1:] != (nodes, nodes):
        raise ValueError("explicit graph shapes differ")
    if adaptive_graph.shape != (nodes, nodes):
        raise ValueError("adaptive graph shape differs")
    expected_graphs = static_graphs.shape[0] + 2
    if weights.shape != (batch, expected_graphs):
        raise ValueError("graph gate weights have the wrong shape")
    static = static_graphs.unsqueeze(0).expand(batch, -1, -1, -1)
    adaptive = adaptive_graph.expand(batch, -1, -1)
    stack = torch.cat([static, directional_graph[:, None], adaptive[:, None]], dim=1)
    stack = normalize_adjacency(stack)
    mixed = torch.einsum("bk,bkij->bij", weights, stack)
    return normalize_adjacency(mixed)


class AdaptiveAdjacency(nn.Module):
    def __init__(self, n_nodes: int, embedding_dim: int = 16) -> None:
        super().__init__()
        self.source = nn.Parameter(torch.empty(n_nodes, embedding_dim))
        self.target = nn.Parameter(torch.empty(n_nodes, embedding_dim))
        nn.init.xavier_uniform_(self.source)
        nn.init.xavier_uniform_(self.target)

    def forward(self) -> Tensor:
        logits = torch.relu(self.source @ self.target.transpose(0, 1))
        return torch.softmax(logits, dim=-1)


class GraphResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.self_projection = nn.Linear(hidden_dim, hidden_dim)
        self.neighbour_projection = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: Tensor, adjacency: Tensor) -> Tensor:
        neighbour = propagate_source_to_destination(adjacency, hidden)
        update = F.gelu(self.self_projection(hidden) + self.neighbour_projection(neighbour))
        return self.norm(hidden + self.dropout(update))


class DynamicMultiGraphResidualForecaster(nn.Module):
    """Forecast turbine residuals with explicit and learned graph mixture.

    Parameters
    ----------
    n_static_graphs:
        Number of fixed graphs passed at runtime (normally geo and corr).  A
        batch directional graph and one adaptive graph are always added.
    """

    def __init__(
        self,
        *,
        n_nodes: int,
        n_features: int,
        forecast_steps: int,
        n_static_graphs: int = 2,
        hidden_dim: int = 64,
        temporal_kernel: int = 3,
        graph_layers: int = 2,
        dropout: float = 0.1,
        rated_power: float | None = None,
        missing_inputs: tuple[str, ...] | list[str] = ("mask", "delta_t"),
        target_mode: str = "physics_residual",
        graph_components: tuple[str, ...] | list[str] = (
            "geo", "corr", "directional", "adaptive"
        ),
        future_weather_features: int = 0,
    ) -> None:
        super().__init__()
        if temporal_kernel % 2 == 0:
            raise ValueError("temporal_kernel must be odd")
        self.n_nodes = n_nodes
        self.n_static_graphs = n_static_graphs
        self.rated_power = rated_power
        if target_mode not in {"physics_residual", "direct_power"}:
            raise ValueError("target_mode must be physics_residual or direct_power")
        self.target_mode = target_mode
        expected_names = ("geo", "corr")[:n_static_graphs] + (
            "directional", "adaptive"
        )
        if not graph_components or not set(graph_components).issubset(expected_names):
            raise ValueError("graph_components contains an unavailable graph")
        self.graph_components = tuple(graph_components)
        if future_weather_features < 0:
            raise ValueError("future_weather_features must be non-negative")
        self.future_weather_features = int(future_weather_features)
        enabled = torch.tensor(
            [name in self.graph_components for name in expected_names], dtype=torch.bool
        )
        self.register_buffer("enabled_graph_mask", enabled)
        allowed_missing_inputs = {"mask", "delta_t"}
        if len(set(missing_inputs)) != len(missing_inputs) or not set(
            missing_inputs
        ).issubset(allowed_missing_inputs):
            raise ValueError("missing_inputs may contain mask and/or delta_t once")
        self.missing_inputs = tuple(missing_inputs)
        input_multiplier = 1 + len(self.missing_inputs)
        self.temporal = nn.Sequential(
            nn.Conv2d(
                input_multiplier * n_features,
                hidden_dim,
                kernel_size=(1, temporal_kernel),
                padding=(0, temporal_kernel // 2),
            ),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, temporal_kernel), padding=(0, temporal_kernel // 2)),
            nn.GELU(),
        )
        self.adaptive = AdaptiveAdjacency(n_nodes)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_static_graphs + 2),
        )
        self.graph_blocks = nn.ModuleList(
            [GraphResidualBlock(hidden_dim, dropout) for _ in range(graph_layers)]
        )
        self.forecast_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, forecast_steps),
        )
        if self.future_weather_features:
            self.weather_projection = nn.Linear(
                2 * self.future_weather_features, hidden_dim
            )
            self.weather_head = nn.Linear(hidden_dim, 1)
        else:
            self.weather_projection = None
            self.weather_head = None

    def _mix_graphs(
        self, hidden: Tensor, static_graphs: Tensor, directional_graph: Tensor
    ) -> tuple[Tensor, Tensor]:
        batch, nodes, _ = hidden.shape
        if nodes != self.n_nodes:
            raise ValueError(f"expected {self.n_nodes} nodes, got {nodes}")
        if static_graphs.shape != (self.n_static_graphs, nodes, nodes):
            raise ValueError("static_graphs has wrong shape")
        if directional_graph.shape != (batch, nodes, nodes):
            raise ValueError("directional_graph has wrong shape")
        logits = self.gate(hidden.mean(dim=1))
        logits = logits.masked_fill(~self.enabled_graph_mask[None], float("-inf"))
        weights = torch.softmax(logits, dim=-1)
        mixed = mix_graphs_from_weights(
            static_graphs, directional_graph, self.adaptive(), weights
        )
        return mixed, weights

    def forward(
        self,
        x_fill: Tensor,
        mask: Tensor,
        delta_t: Tensor,
        static_graphs: Tensor,
        directional_graph: Tensor,
        p_pc_future: Tensor,
        future_weather: Tensor | None = None,
        future_weather_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if x_fill.ndim != 4 or mask.shape != x_fill.shape or delta_t.shape != x_fill.shape:
            raise ValueError("x_fill/mask/delta_t must share [B,N,L,D]")
        inputs = [x_fill]
        if "mask" in self.missing_inputs:
            inputs.append(mask)
        if "delta_t" in self.missing_inputs:
            # Log scaling keeps long missing runs from dominating feature values.
            inputs.append(torch.log1p(delta_t.clamp_min(0.0)))
        model_input = torch.cat(inputs, dim=-1)
        encoded = self.temporal(model_input.permute(0, 3, 1, 2))
        hidden = encoded[..., -1].permute(0, 2, 1)
        adjacency, graph_weights = self._mix_graphs(hidden, static_graphs, directional_graph)
        for block in self.graph_blocks:
            hidden = block(hidden, adjacency)
        weather_effect = None
        if self.future_weather_features:
            if future_weather is None or future_weather_mask is None:
                raise ValueError("configured future weather inputs are missing")
            expected = (*p_pc_future.shape, self.future_weather_features)
            if future_weather.shape != expected or future_weather_mask.shape != expected:
                raise ValueError("future weather must have [B,N,H,Fw]")
            weather_encoded = F.gelu(
                self.weather_projection(
                    torch.cat([future_weather, future_weather_mask], dim=-1)
                )
            )
            hidden = hidden + weather_encoded.mean(dim=2)
            weather_effect = self.weather_head(weather_encoded).squeeze(-1)
        elif future_weather is not None or future_weather_mask is not None:
            raise ValueError("model was not configured for future weather")
        residual = self.forecast_head(hidden)
        if weather_effect is not None:
            residual = residual + weather_effect
        if p_pc_future.shape != residual.shape:
            raise ValueError("p_pc_future must have shape [B,N,H]")
        centre = residual if self.target_mode == "direct_power" else p_pc_future + residual
        if self.rated_power is not None:
            centre = centre.clamp(0.0, self.rated_power)
        else:
            centre = centre.clamp_min(0.0)
        return {
            "residual": residual,
            "y_det": centre,
            "hidden": hidden,
            "adjacency": adjacency,
            "graph_weights": graph_weights,
        }


def masked_huber_loss(prediction: Tensor, target: Tensor, mask: Tensor, delta: float = 1.0) -> Tensor:
    if prediction.shape != target.shape or prediction.shape != mask.shape:
        raise ValueError("prediction, target and mask shapes differ")
    elementwise = F.huber_loss(prediction, target, delta=delta, reduction="none")
    weights = mask.to(dtype=elementwise.dtype)
    return (elementwise * weights).sum() / weights.sum().clamp_min(1.0)
