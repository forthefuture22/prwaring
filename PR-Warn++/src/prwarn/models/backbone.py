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
    gate_mode:
        How the per-sample graph mixture weights are produced.

        ``learned`` (default)
            The original behaviour: an MLP reads ``hidden`` and softmax produces
            input-dependent weights.  This path is bit-for-bit identical to the
            pre-intervention implementation.
        ``uniform``
            Every enabled graph gets equal weight (1 / #enabled graphs); the gate
            MLP output is ignored.  This is the zero-hypothesis baseline of
            Wiegreffe & Pinter (2019).
        ``parameter_matched``
            The gate MLP is retained with exactly the same parameter count as
            ``learned`` (so total parameter count matches learned), but its
            output is ignored, weights are fixed to the uniform distribution,
            and the gate parameters are frozen (``requires_grad=False``).  This
            rules out "improvement just from extra parameter capacity" (Michel
            et al. 2019; Wiegreffe & Pinter 2019).

        ``frozen_mean`` and ``shuffled`` are *post-hoc* interventions on a trained
        ``learned`` checkpoint and are intentionally NOT modes here; they are
        driven by ``prwarn.cli.evaluate_gate_intervention`` through the
        ``_posthoc_weights`` instance attribute documented on ``_mix_graphs``.
    gate_hidden_dim:
        Width of the gate MLP's hidden layer.  When ``None`` (default) it falls
        back to ``hidden_dim``, so the gate is byte-identical to the original
        coupled implementation.  Setting it decouples the gate width from the
        backbone width (gap-4 OFAT gate-width sweep).
    """

    GATE_MODES = ("learned", "uniform", "parameter_matched")

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
        gate_mode: str = "learned",
        gate_hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        if temporal_kernel % 2 == 0:
            raise ValueError("temporal_kernel must be odd")
        if gate_mode not in self.GATE_MODES:
            raise ValueError(
                f"gate_mode must be one of {self.GATE_MODES}, got {gate_mode!r}"
            )
        self.gate_mode = gate_mode
        if gate_hidden_dim is not None and int(gate_hidden_dim) <= 0:
            raise ValueError("gate_hidden_dim must be positive when set")
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
        # Gate hidden width decoupled from the backbone width (gap-4 OFAT).
        # None reproduces the original coupled `hidden_dim` width exactly.
        gate_inner = int(gate_hidden_dim) if gate_hidden_dim is not None else hidden_dim
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, gate_inner),
            nn.GELU(),
            nn.Linear(gate_inner, n_static_graphs + 2),
        )
        if gate_mode == "parameter_matched":
            # Keep the gate parameters (matching learned parameter count) but make
            # them untrainable: the fixed uniform distribution is used instead.
            for parameter in self.gate.parameters():
                parameter.requires_grad_(False)
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
        if self.gate_mode == "learned":
            weights = torch.softmax(logits, dim=-1)
        elif self.gate_mode in ("uniform", "parameter_matched"):
            # Equal weight across enabled graphs; disabled graphs receive zero.
            enabled = self.enabled_graph_mask.to(dtype=logits.dtype)
            weights = (enabled / enabled.sum()).unsqueeze(0).expand(batch, -1)
        else:  # pragma: no cover - guarded in __init__
            raise ValueError(f"unsupported gate_mode: {self.gate_mode!r}")
        # Post-hoc intervention hook (frozen_mean / shuffled), exclusively driven
        # by prwarn.cli.evaluate_gate_intervention on a learned checkpoint.  When
        # left as the default (None) the learned path is unchanged bit-for-bit.
        # When set it must be a (batch, n_static_graphs + 2) Tensor aligned to the
        # current batch and replaces the produced weights.
        posthoc = getattr(self, "_posthoc_weights", None)
        if posthoc is not None:
            weights = posthoc.to(dtype=weights.dtype, device=weights.device)
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
