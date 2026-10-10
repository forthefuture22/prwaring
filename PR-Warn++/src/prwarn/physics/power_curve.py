"""Train-only monotone empirical wind-speed to power curve."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _weighted_pava(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted pool-adjacent-violators algorithm for a nondecreasing curve."""

    level: list[float] = []
    weight: list[float] = []
    start: list[int] = []
    end: list[int] = []
    for i, (value, item_weight) in enumerate(zip(values, weights)):
        level.append(float(value))
        weight.append(float(item_weight))
        start.append(i)
        end.append(i)
        while len(level) >= 2 and level[-2] > level[-1]:
            combined_weight = weight[-2] + weight[-1]
            combined_level = (
                level[-2] * weight[-2] + level[-1] * weight[-1]
            ) / combined_weight
            level[-2:] = [combined_level]
            weight[-2:] = [combined_weight]
            end[-2:] = [end[-1]]
            start.pop()
    result = np.empty_like(values, dtype=float)
    for value, left, right in zip(level, start, end):
        result[left : right + 1] = value
    return result


@dataclass
class EmpiricalPowerCurve:
    """Bin-median power curve with monotone smoothing and linear interpolation."""

    n_bins: int = 50
    minimum_bin_count: int = 20
    rated_power: float | None = None
    bin_centres_: np.ndarray | None = None
    bin_power_: np.ndarray | None = None
    bin_counts_: np.ndarray | None = None

    def fit(
        self,
        equivalent_speed: np.ndarray,
        power: np.ndarray,
        valid_mask: np.ndarray | None = None,
    ) -> "EmpiricalPowerCurve":
        speed = np.asarray(equivalent_speed, dtype=float).reshape(-1)
        target = np.asarray(power, dtype=float).reshape(-1)
        if speed.shape != target.shape:
            raise ValueError("equivalent_speed and power must have the same shape")
        valid = np.isfinite(speed) & np.isfinite(target) & (speed >= 0.0)
        if valid_mask is not None:
            valid &= np.asarray(valid_mask, dtype=bool).reshape(-1)
        speed, target = speed[valid], target[valid]
        if speed.size < max(4, self.minimum_bin_count):
            raise ValueError("too few valid Train observations to fit power curve")

        lower, upper = float(speed.min()), float(speed.max())
        if lower == upper:
            raise ValueError("wind speed has no variation")
        edges = np.linspace(lower, upper, self.n_bins + 1)
        bin_id = np.clip(np.digitize(speed, edges[1:-1]), 0, self.n_bins - 1)
        centres, medians, counts = [], [], []
        for i in range(self.n_bins):
            selected = target[bin_id == i]
            if selected.size >= self.minimum_bin_count:
                centres.append((edges[i] + edges[i + 1]) / 2.0)
                medians.append(float(np.median(selected)))
                counts.append(selected.size)
        if len(centres) < 2:
            raise ValueError("fewer than two populated power-curve bins")

        power_values = _weighted_pava(np.asarray(medians), np.asarray(counts))
        upper_power = self.rated_power
        if upper_power is None:
            upper_power = float(np.nanquantile(target, 0.995))
        self.bin_centres_ = np.asarray(centres, dtype=float)
        self.bin_power_ = np.clip(power_values, 0.0, upper_power)
        self.bin_counts_ = np.asarray(counts, dtype=np.int64)
        self.rated_power = upper_power
        return self

    def predict(self, equivalent_speed: np.ndarray) -> np.ndarray:
        if self.bin_centres_ is None or self.bin_power_ is None:
            raise RuntimeError("power curve is not fitted")
        speed = np.asarray(equivalent_speed, dtype=float)
        predicted = np.interp(
            speed,
            self.bin_centres_,
            self.bin_power_,
            left=self.bin_power_[0],
            right=self.bin_power_[-1],
        )
        return np.clip(predicted, 0.0, float(self.rated_power))

    def to_dict(self) -> dict[str, object]:
        if self.bin_centres_ is None or self.bin_power_ is None or self.bin_counts_ is None:
            raise RuntimeError("power curve is not fitted")
        return {
            "n_bins": self.n_bins,
            "minimum_bin_count": self.minimum_bin_count,
            "rated_power": self.rated_power,
            "bin_centres": self.bin_centres_.tolist(),
            "bin_power": self.bin_power_.tolist(),
            "bin_counts": self.bin_counts_.tolist(),
        }

    @classmethod
    def from_dict(cls, state: dict[str, object]) -> "EmpiricalPowerCurve":
        curve = cls(
            n_bins=int(state["n_bins"]),
            minimum_bin_count=int(state["minimum_bin_count"]),
            rated_power=float(state["rated_power"]),
        )
        curve.bin_centres_ = np.asarray(state["bin_centres"], dtype=float)
        curve.bin_power_ = np.asarray(state["bin_power"], dtype=float)
        curve.bin_counts_ = np.asarray(state["bin_counts"], dtype=np.int64)
        return curve
