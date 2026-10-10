"""Synthetic missing-data stress tests for historical model inputs.

All functions operate on the window layout ``[batch, node, history, feature]``.
They never accept or modify forecast labels, which makes accidental target
leakage harder when stress tests are assembled by experiment scripts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MissingnessStressResult:
    """A stressed copy of the model inputs and the newly hidden positions."""

    x: np.ndarray
    mask: np.ndarray
    delta_t: np.ndarray
    selected: np.ndarray
    metadata: dict[str, object]


@dataclass(frozen=True)
class IssueTimeDerived:
    """Quantities that must be rebuilt after hiding historical observations."""

    p_pc: np.ndarray
    current_y: np.ndarray
    current_y_mask: np.ndarray
    wind_from: np.ndarray


def _validate_mask(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask)
    if mask.ndim != 4:
        raise ValueError("mask must have shape [batch,node,history,feature]")
    if not np.all(np.isfinite(mask)):
        raise ValueError("mask must be finite")
    return mask > 0


def _candidates(mask: np.ndarray, eligible: np.ndarray | None) -> np.ndarray:
    observed = _validate_mask(mask)
    if eligible is None:
        return observed
    try:
        allowed = np.broadcast_to(np.asarray(eligible, dtype=bool), observed.shape)
    except ValueError as exc:
        raise ValueError("eligible must broadcast to mask shape") from exc
    return observed & allowed


def mcar_missingness(
    mask: np.ndarray,
    *,
    rate: float,
    seed: int,
    eligible: np.ndarray | None = None,
) -> np.ndarray:
    """Select observed entries independently with probability ``rate``."""

    if not 0.0 <= rate <= 1.0:
        raise ValueError("rate must be between 0 and 1")
    candidates = _candidates(mask, eligible)
    rng = np.random.default_rng(seed)
    return candidates & (rng.random(candidates.shape) < rate)


def block_missingness(
    mask: np.ndarray,
    *,
    block_length: int,
    blocks_per_sample: int,
    seed: int,
    eligible: np.ndarray | None = None,
) -> np.ndarray:
    """Hide contiguous time blocks on randomly selected node-feature series."""

    candidates = _candidates(mask, eligible)
    batch, nodes, history, features = candidates.shape
    if block_length <= 0 or block_length > history:
        raise ValueError("block_length must be in [1, history]")
    if blocks_per_sample < 0:
        raise ValueError("blocks_per_sample must be non-negative")
    rng = np.random.default_rng(seed)
    selected = np.zeros_like(candidates)
    for b in range(batch):
        for _ in range(blocks_per_sample):
            n = int(rng.integers(nodes))
            d = int(rng.integers(features))
            start = int(rng.integers(history - block_length + 1))
            selected[b, n, start : start + block_length, d] = True
    return selected & candidates


def spatial_outage_missingness(
    mask: np.ndarray,
    *,
    node_fraction: float,
    duration: int,
    seed: int,
    eligible: np.ndarray | None = None,
) -> np.ndarray:
    """Hide all features for a shared time block at a subset of turbines."""

    candidates = _candidates(mask, eligible)
    batch, nodes, history, _ = candidates.shape
    if not 0.0 < node_fraction <= 1.0:
        raise ValueError("node_fraction must be in (0, 1]")
    if duration <= 0 or duration > history:
        raise ValueError("duration must be in [1, history]")
    node_count = max(1, int(np.ceil(nodes * node_fraction)))
    rng = np.random.default_rng(seed)
    selected = np.zeros_like(candidates)
    for b in range(batch):
        chosen = rng.choice(nodes, size=node_count, replace=False)
        start = int(rng.integers(history - duration + 1))
        selected[b, chosen, start : start + duration, :] = True
    return selected & candidates


def extreme_conditioned_missingness(
    mask: np.ndarray,
    extreme: np.ndarray,
    *,
    base_rate: float,
    extreme_multiplier: float,
    seed: int,
    eligible: np.ndarray | None = None,
) -> np.ndarray:
    """Sample more missing entries where a precomputed extreme flag is true.

    ``extreme`` is normally ``[batch,node,history,1]`` or
    ``[batch,node,history,feature]``. It must be computed from information
    available at the historical timestamp, not from forecast targets.
    """

    if not 0.0 <= base_rate <= 1.0:
        raise ValueError("base_rate must be between 0 and 1")
    if extreme_multiplier < 0.0:
        raise ValueError("extreme_multiplier must be non-negative")
    candidates = _candidates(mask, eligible)
    try:
        extreme_flag = np.broadcast_to(np.asarray(extreme, dtype=bool), candidates.shape)
    except ValueError as exc:
        raise ValueError("extreme must broadcast to mask shape") from exc
    probabilities = np.where(
        extreme_flag,
        min(1.0, base_rate * extreme_multiplier),
        base_rate,
    )
    rng = np.random.default_rng(seed)
    return candidates & (rng.random(candidates.shape) < probabilities)


def apply_missingness_stress(
    x: np.ndarray,
    mask: np.ndarray,
    delta_t: np.ndarray,
    selected: np.ndarray,
    *,
    step_minutes: float,
    fill_value: float | np.ndarray = 0.0,
    fill_strategy: str = "constant",
    metadata: dict[str, object] | None = None,
) -> MissingnessStressResult:
    """Apply a selection mask to copies of ``x``, ``mask`` and ``delta_t``.

    A fill value of zero is appropriate for inputs standardized with training
    means. Elapsed time is recomputed along the history axis after hiding the
    selected observations.
    """

    x_arr = np.asarray(x)
    mask_arr = np.asarray(mask)
    delta_arr = np.asarray(delta_t)
    selected_arr = np.asarray(selected, dtype=bool)
    if x_arr.ndim != 4 or mask_arr.shape != x_arr.shape or delta_arr.shape != x_arr.shape:
        raise ValueError("x, mask and delta_t must share [batch,node,history,feature]")
    if selected_arr.shape != x_arr.shape:
        raise ValueError("selected must have the same shape as x")
    if not np.isfinite(step_minutes) or step_minutes <= 0:
        raise ValueError("step_minutes must be positive and finite")
    if fill_strategy not in {"constant", "forward_fill"}:
        raise ValueError("fill_strategy must be constant or forward_fill")

    observed = mask_arr > 0
    artificial = selected_arr & observed
    stressed_mask_bool = observed & ~artificial
    stressed_x = np.array(x_arr, copy=True)
    broadcast_fill = np.broadcast_to(
        np.asarray(fill_value, dtype=stressed_x.dtype), stressed_x.shape
    )
    if fill_strategy == "constant":
        stressed_x[artificial] = broadcast_fill[artificial]
    else:
        # Original missing entries already contain a legitimate carrier from
        # before the window. Newly hidden t=0 entries cannot use their original
        # value; later missing entries carry the latest still-visible value.
        first_artificial = artificial[:, :, 0, :]
        stressed_x[:, :, 0, :][first_artificial] = broadcast_fill[:, :, 0, :][
            first_artificial
        ]
        for t in range(1, x_arr.shape[2]):
            missing_now = ~stressed_mask_bool[:, :, t, :]
            stressed_x[:, :, t, :] = np.where(
                missing_now,
                stressed_x[:, :, t - 1, :],
                stressed_x[:, :, t, :],
            )

    stressed_delta = np.zeros(x_arr.shape, dtype=np.result_type(delta_arr.dtype, np.float32))
    first_missing = ~stressed_mask_bool[:, :, 0, :]
    original_first = np.maximum(delta_arr[:, :, 0, :], step_minutes)
    stressed_delta[:, :, 0, :] = np.where(first_missing, original_first, 0.0)
    for t in range(1, x_arr.shape[2]):
        stressed_delta[:, :, t, :] = np.where(
            stressed_mask_bool[:, :, t, :],
            0.0,
            stressed_delta[:, :, t - 1, :] + step_minutes,
        )

    info = dict(metadata or {})
    info.update(
        selected_count=int(artificial.sum()),
        selected_fraction=float(artificial.sum() / max(1, observed.sum())),
        fill_strategy=fill_strategy,
    )
    return MissingnessStressResult(
        x=stressed_x,
        mask=stressed_mask_bool.astype(mask_arr.dtype, copy=False),
        delta_t=stressed_delta,
        selected=artificial,
        metadata=info,
    )


def rebuild_issue_time_derived(
    x_standardized: np.ndarray,
    mask: np.ndarray,
    metadata: dict[str, object],
) -> IssueTimeDerived:
    """Rebuild physics/directional carriers from a stressed history window.

    The function reads only the final value of the already forward-filled
    historical carrier. Forecast labels are neither accepted nor accessed.
    """

    from prwarn.physics.density import air_density, equivalent_wind_speed
    from prwarn.physics.power_curve import EmpiricalPowerCurve

    x = np.asarray(x_standardized, dtype=float)
    observed = np.asarray(mask)
    if x.ndim != 4 or observed.shape != x.shape:
        raise ValueError("x_standardized and mask must share [B,N,L,D]")
    feature_index = {
        str(name): int(index)
        for name, index in dict(metadata["feature_index"]).items()
    }
    scaler = dict(metadata["feature_scaler"])
    mean = np.asarray(scaler["mean"], dtype=float)
    scale = np.asarray(scaler["std"], dtype=float)
    if mean.shape != (x.shape[-1],) or scale.shape != mean.shape:
        raise ValueError("feature scaler shape differs from x")
    raw_origin = x[:, :, -1, :] * scale + mean
    columns = dict(metadata["physical_columns"])

    def index(column_key: str) -> int:
        name = str(columns[column_key])
        if name not in feature_index:
            raise KeyError(f"required feature {name!r} is absent")
        return feature_index[name]

    speed = raw_origin[..., index("wind_speed")]
    pressure = raw_origin[..., index("pressure")]
    temperature = raw_origin[..., index("temperature")]
    units = dict(metadata["physical_units"])
    if units["pressure_input"] == "hpa":
        pressure = pressure * 100.0
    elif units["pressure_input"] != "pa":
        raise ValueError("unsupported pressure unit in metadata")
    if units["temperature_input"] == "celsius":
        temperature = temperature + 273.15
    elif units["temperature_input"] != "kelvin":
        raise ValueError("unsupported temperature unit in metadata")
    equivalent = (
        equivalent_wind_speed(speed, air_density(pressure, temperature))
        if bool(metadata.get("density_correction", True))
        else speed
    )
    curve = EmpiricalPowerCurve.from_dict(dict(metadata["power_curve"]))
    p_pc_origin = curve.predict(equivalent).astype(np.float32)
    horizon = int(metadata["forecast_steps"])
    p_pc = np.repeat(p_pc_origin[..., None], horizon, axis=-1)

    target_index = index("target")
    current_y = raw_origin[..., target_index].astype(np.float32)
    current_y_mask = observed[:, :, -1, target_index].astype(np.float32)
    direction = dict(metadata["wind_direction"])
    wind_feature = str(direction["wind_feature"])
    wind_from = raw_origin[..., feature_index[wind_feature]]
    if direction["mode"] == "relative_plus_nacelle":
        nacelle_feature = str(direction["nacelle_feature"])
        wind_from = wind_from + raw_origin[..., feature_index[nacelle_feature]]
    elif direction["mode"] != "global":
        raise ValueError("unsupported wind-direction mode in metadata")
    wind_from = np.mod(wind_from, 360.0).astype(np.float32)
    return IssueTimeDerived(
        p_pc=p_pc,
        current_y=current_y,
        current_y_mask=current_y_mask,
        wind_from=wind_from,
    )
