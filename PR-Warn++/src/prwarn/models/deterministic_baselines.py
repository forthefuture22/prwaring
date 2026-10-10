"""GRU and TCN deterministic baselines with the main cache-compatible API."""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .backbone import AdaptiveAdjacency, normalize_adjacency, propagate_source_to_destination


class _FixedAdaptive(nn.Module):
    def __init__(self, n_nodes: int) -> None:
        super().__init__()
        self.register_buffer("matrix", torch.eye(n_nodes))

    def forward(self) -> Tensor:
        return self.matrix


class TemporalDeterministicBaseline(nn.Module):
    """Controlled GRU/TCN and graph-temporal style baselines.

    ``graphwavenet`` and ``agcrn`` are transparent in-repository style
    implementations, not claims of byte-identical reproduction of external
    author repositories.
    """

    def __init__(
        self,
        *,
        architecture: str,
        n_nodes: int,
        n_features: int,
        forecast_steps: int,
        n_static_graphs: int = 2,
        hidden_dim: int = 64,
        temporal_kernel: int = 3,
        dropout: float = 0.1,
        rated_power: float | None = None,
        missing_inputs: tuple[str, ...] | list[str] = ("mask", "delta_t"),
        target_mode: str = "physics_residual",
        future_weather_features: int = 0,
    ) -> None:
        super().__init__()
        if architecture not in {"gru", "tcn", "graphwavenet", "agcrn"}:
            raise ValueError("unsupported deterministic baseline architecture")
        if target_mode not in {"physics_residual", "direct_power"}:
            raise ValueError("invalid target_mode")
        if temporal_kernel <= 0 or temporal_kernel % 2 == 0:
            raise ValueError("temporal_kernel must be positive and odd")
        allowed = {"mask", "delta_t"}
        if len(set(missing_inputs)) != len(missing_inputs) or not set(
            missing_inputs
        ).issubset(allowed):
            raise ValueError("missing_inputs may contain mask and/or delta_t once")
        self.architecture = architecture
        self.n_nodes = n_nodes
        self.n_static_graphs = n_static_graphs
        self.rated_power = rated_power
        self.missing_inputs = tuple(missing_inputs)
        self.target_mode = target_mode
        if future_weather_features < 0:
            raise ValueError("future_weather_features must be non-negative")
        self.future_weather_features = int(future_weather_features)
        input_dim = n_features * (1 + len(self.missing_inputs))
        if architecture == "gru":
            self.encoder = nn.GRU(input_dim, hidden_dim, batch_first=True)
        elif architecture == "tcn":
            self.encoder = nn.Sequential(
                nn.Conv1d(
                    input_dim, hidden_dim, temporal_kernel,
                    padding=temporal_kernel // 2,
                ),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv1d(
                    hidden_dim, hidden_dim, temporal_kernel,
                    padding=temporal_kernel // 2,
                ),
                nn.GELU(),
            )
        elif architecture == "graphwavenet":
            self.causal_padding = 2 * (temporal_kernel - 1)
            self.input_projection = nn.Conv1d(input_dim, hidden_dim, 1)
            self.filter_conv = nn.Conv1d(
                hidden_dim, hidden_dim, temporal_kernel, dilation=2
            )
            self.gate_conv = nn.Conv1d(
                hidden_dim, hidden_dim, temporal_kernel, dilation=2
            )
            self.spatial_projection = nn.Linear(hidden_dim, hidden_dim)
            self.spatial_norm = nn.LayerNorm(hidden_dim)
        else:
            self.input_projection = nn.Linear(input_dim, hidden_dim)
            self.recurrent_cell = nn.GRUCell(2 * hidden_dim, hidden_dim)
        self.adaptive = (
            AdaptiveAdjacency(n_nodes)
            if architecture in {"graphwavenet", "agcrn"}
            else _FixedAdaptive(n_nodes)
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

    def _encode(self, model_input: Tensor) -> Tensor:
        batch, nodes, history, width = model_input.shape
        sequence = model_input.reshape(batch * nodes, history, width)
        if self.architecture == "gru":
            _, hidden = self.encoder(sequence)
            encoded = hidden[-1]
        elif self.architecture == "tcn":
            encoded = self.encoder(sequence.transpose(1, 2))[..., -1]
        elif self.architecture == "graphwavenet":
            projected = self.input_projection(sequence.transpose(1, 2))
            causal = F.pad(projected, (self.causal_padding, 0))
            filtered = self.filter_conv(causal)
            gated = self.gate_conv(causal)
            encoded = (torch.tanh(filtered) * torch.sigmoid(gated))[..., -1]
        else:
            projected = self.input_projection(model_input)
            adaptive = self.adaptive().expand(batch, -1, -1)
            hidden = torch.zeros(
                batch, nodes, projected.shape[-1],
                device=model_input.device, dtype=model_input.dtype,
            )
            for time_index in range(history):
                neighbour = propagate_source_to_destination(adaptive, hidden)
                cell_input = torch.cat(
                    [projected[:, :, time_index], neighbour], dim=-1
                ).reshape(batch * nodes, -1)
                hidden = self.recurrent_cell(
                    cell_input, hidden.reshape(batch * nodes, -1)
                ).reshape(batch, nodes, -1)
            return hidden
        return encoded.reshape(batch, nodes, -1)

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
            inputs.append(torch.log1p(delta_t.clamp_min(0.0)))
        hidden = self._encode(torch.cat(inputs, dim=-1))
        if self.architecture == "graphwavenet":
            adaptive = self.adaptive().expand(x_fill.shape[0], -1, -1)
            neighbour = propagate_source_to_destination(adaptive, hidden)
            hidden = self.spatial_norm(hidden + F.gelu(self.spatial_projection(neighbour)))
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
        raw = self.forecast_head(hidden)
        if weather_effect is not None:
            raw = raw + weather_effect
        centre = raw if self.target_mode == "direct_power" else p_pc_future + raw
        centre = centre.clamp_min(0.0)
        if self.rated_power is not None:
            centre = centre.clamp_max(self.rated_power)
        batch = x_fill.shape[0]
        adjacency = (
            self.adaptive().expand(batch, -1, -1)
            if self.architecture in {"graphwavenet", "agcrn"}
            else normalize_adjacency(static_graphs[0]).expand(batch, -1, -1)
        )
        weights = torch.zeros(
            batch, self.n_static_graphs + 2,
            device=x_fill.device, dtype=x_fill.dtype,
        )
        if self.architecture in {"graphwavenet", "agcrn"}:
            weights[:, -1] = 1.0
        else:
            weights[:, 0] = 1.0
        return {
            "residual": raw,
            "y_det": centre,
            "hidden": hidden,
            "adjacency": adjacency,
            "graph_weights": weights,
        }
