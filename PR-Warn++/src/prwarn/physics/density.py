"""Air-density normalization used by the weak physical prior."""

from __future__ import annotations

import numpy as np


DRY_AIR_GAS_CONSTANT = 287.05
REFERENCE_DENSITY = 1.225


def air_density(
    pressure_pa: np.ndarray | float,
    temperature_k: np.ndarray | float,
    *,
    gas_constant: float = DRY_AIR_GAS_CONSTANT,
) -> np.ndarray:
    """Compute dry-air density in kg/m^3 from pressure Pa and temperature K."""

    pressure = np.asarray(pressure_pa, dtype=float)
    temperature = np.asarray(temperature_k, dtype=float)
    if np.any((pressure <= 0) & np.isfinite(pressure)):
        raise ValueError("pressure must be positive and expressed in Pa")
    if np.any((temperature <= 0) & np.isfinite(temperature)):
        raise ValueError("temperature must be positive and expressed in K")
    return pressure / (gas_constant * temperature)


def equivalent_wind_speed(
    wind_speed: np.ndarray | float,
    density: np.ndarray | float,
    *,
    reference_density: float = REFERENCE_DENSITY,
) -> np.ndarray:
    """Map wind speed to the reference-density equivalent speed."""

    speed = np.asarray(wind_speed, dtype=float)
    rho = np.asarray(density, dtype=float)
    if reference_density <= 0:
        raise ValueError("reference_density must be positive")
    if np.any((rho <= 0) & np.isfinite(rho)):
        raise ValueError("density must be positive")
    return speed * np.cbrt(rho / reference_density)
