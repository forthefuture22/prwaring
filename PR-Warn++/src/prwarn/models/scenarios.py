"""Scenario reconstruction contracts."""

from __future__ import annotations

import torch
from torch import Tensor


def reconstruct_power_scenarios(
    normalized_error_scenarios: Tensor,
    y_det: Tensor,
    error_mean: Tensor,
    error_scale: Tensor,
    *,
    rated_power: float,
) -> Tensor:
    """Map `[M,B,N,H]` normalized errors back to bounded power scenarios."""

    if normalized_error_scenarios.ndim != 4 or y_det.shape != normalized_error_scenarios.shape[1:]:
        raise ValueError("scenario/y_det shape mismatch")
    error = normalized_error_scenarios * error_scale + error_mean
    return (y_det.unsqueeze(0) + error).clamp(0.0, rated_power)
