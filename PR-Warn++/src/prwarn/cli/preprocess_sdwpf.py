"""Preprocess a frozen SDWPF table into leakage-safe window archives.

This command performs Gate-1/2 mechanics only.  It does not download data and
does not silently infer pressure/temperature units.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from prwarn.data.grid import build_rigid_grid, build_sdwpf_bundle, bundle_to_time_node
from prwarn.data.split import chronological_split
from prwarn.data.window import make_windows
from prwarn.graphs.builders import correlation_graph, geographic_graph
from prwarn.physics.density import air_density, equivalent_wind_speed
from prwarn.physics.power_curve import EmpiricalPowerCurve


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".csv", ".gz"}:
        return pd.read_csv(path)
    raise ValueError("input must be parquet or csv(.gz)")


def _history_only_power_curve(values: np.ndarray, origins: np.ndarray, horizon: int) -> np.ndarray:
    """Persist origin-time P_pc; never read future observed weather."""

    origin_value = values[origins]  # [B,N]
    return np.repeat(origin_value[..., None], horizon, axis=-1).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timestamp-col", default="Tmstamp")
    parser.add_argument("--turbine-col", default="TurbID")
    parser.add_argument("--target-col", default="Patv")
    parser.add_argument("--wind-speed-col", default="Wspd")
    parser.add_argument("--pressure-col", default="Sp")
    parser.add_argument("--temperature-col", default="T2m")
    parser.add_argument("--pressure-unit", choices=("pa", "hpa"), default="pa")
    parser.add_argument("--temperature-unit", choices=("kelvin", "celsius"), default="kelvin")
    parser.add_argument(
        "--density-correction",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use --no-density-correction for the A1 raw-wind-speed ablation.",
    )
    parser.add_argument("--wind-direction-col", default="Wdir")
    parser.add_argument("--nacelle-direction-col", default="Ndir")
    parser.add_argument(
        "--wind-direction-mode",
        choices=("global", "relative_plus_nacelle"),
        default="relative_plus_nacelle",
    )
    parser.add_argument("--features", nargs="+", required=True)
    parser.add_argument("--history", type=int, default=24)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--frequency", default="10min")
    parser.add_argument("--step-minutes", type=float, default=10.0)
    parser.add_argument("--rated-power", type=float, required=True)
    parser.add_argument("--curve-bins", type=int, default=50)
    parser.add_argument("--curve-min-count", type=int, default=20)
    parser.add_argument("--locations", type=Path)
    parser.add_argument("--location-x-col", default="x")
    parser.add_argument("--location-y-col", default="y")
    parser.add_argument("--coordinate-unit", default="dataset_native")
    parser.add_argument("--geographic-k", type=int, default=8)
    parser.add_argument("--correlation-k", type=int, default=12)
    parser.add_argument("--correlation-mode", choices=("raw", "difference"), default="difference")
    parser.add_argument("--geographic-distance-scale", type=float)
    args = parser.parse_args()

    required_features = {
        args.target_col,
        args.wind_speed_col,
        args.pressure_col,
        args.temperature_col,
        args.wind_direction_col,
    }
    if not required_features.issubset(args.features):
        raise ValueError(f"--features must include {sorted(required_features)}")
    if (
        args.wind_direction_mode == "relative_plus_nacelle"
        and args.nacelle_direction_col not in args.features
    ):
        raise ValueError("relative_plus_nacelle requires the nacelle direction feature")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    raw = _read_table(args.input)
    rigid = build_rigid_grid(
        raw,
        timestamp_col=args.timestamp_col,
        turbine_col=args.turbine_col,
        frequency=args.frequency,
    )
    splits, manifest = chronological_split(rigid, timestamp_col=args.timestamp_col)

    train_unfilled = build_sdwpf_bundle(
        splits["train"],
        args.features,
        timestamp_col=args.timestamp_col,
        turbine_col=args.turbine_col,
        step_minutes=args.step_minutes,
    )
    train_means = train_unfilled["fitted_means"]
    train_bundle = build_sdwpf_bundle(
        splits["train"],
        args.features,
        timestamp_col=args.timestamp_col,
        turbine_col=args.turbine_col,
        step_minutes=args.step_minutes,
        train_means=train_means,
    )
    train_tensor = bundle_to_time_node(
        train_bundle, timestamp_col=args.timestamp_col, turbine_col=args.turbine_col
    )
    feature_index = {name: index for index, name in enumerate(args.features)}
    train_valid = train_tensor["mask"].astype(bool)
    train_values = train_tensor["x_fill"].astype(np.float64)
    feature_count = train_valid.sum(axis=(0, 1))
    if np.any(feature_count == 0):
        missing_features = [args.features[i] for i in np.flatnonzero(feature_count == 0)]
        raise ValueError(f"features have no valid Train observations: {missing_features}")
    feature_mean = (train_values * train_valid).sum(axis=(0, 1)) / feature_count
    centered = np.where(train_valid, train_values - feature_mean, 0.0)
    feature_std = np.sqrt(np.square(centered).sum(axis=(0, 1)) / feature_count)
    feature_std = np.maximum(feature_std, 1e-6)
    speed = train_tensor["x_fill"][..., feature_index[args.wind_speed_col]]
    pressure = train_tensor["x_fill"][..., feature_index[args.pressure_col]]
    temperature = train_tensor["x_fill"][..., feature_index[args.temperature_col]]
    power = train_tensor["x_fill"][..., feature_index[args.target_col]]
    valid = train_tensor["mask"][..., feature_index[args.target_col]].astype(bool)
    pressure_for_density = pressure * (100.0 if args.pressure_unit == "hpa" else 1.0)
    temperature_for_density = temperature + (
        273.15 if args.temperature_unit == "celsius" else 0.0
    )
    equivalent = (
        equivalent_wind_speed(
            speed, air_density(pressure_for_density, temperature_for_density)
        )
        if args.density_correction
        else speed
    )
    curve = EmpiricalPowerCurve(
        n_bins=args.curve_bins,
        minimum_bin_count=args.curve_min_count,
        rated_power=args.rated_power,
    ).fit(equivalent, power, valid)

    correlation_input = np.where(valid, power, np.nan)
    a_corr = correlation_graph(
        correlation_input, mode=args.correlation_mode, k=args.correlation_k
    )
    np.save(args.output_dir / "a_corr.npy", a_corr)

    coordinates = None
    if args.locations is not None:
        locations = _read_table(args.locations)
        aligned = locations.set_index(args.turbine_col).reindex(train_tensor["turbines"])
        if aligned[[args.location_x_col, args.location_y_col]].isna().any().any():
            raise ValueError("location file does not cover every turbine")
        coordinates = aligned[[args.location_x_col, args.location_y_col]].to_numpy(float)
        np.save(args.output_dir / "coordinates.npy", coordinates)
        np.save(
            args.output_dir / "a_geo.npy",
            geographic_graph(
                coordinates,
                k=args.geographic_k,
                distance_scale=args.geographic_distance_scale,
            ),
        )

    for split_name, split_frame in splits.items():
        bundle = build_sdwpf_bundle(
            split_frame,
            args.features,
            timestamp_col=args.timestamp_col,
            turbine_col=args.turbine_col,
            step_minutes=args.step_minutes,
            train_means=train_means,
        )
        tensor = bundle_to_time_node(
            bundle, timestamp_col=args.timestamp_col, turbine_col=args.turbine_col
        )
        x_raw = tensor["x_fill"]
        target = x_raw[..., feature_index[args.target_col]]
        target_mask = tensor["mask"][..., feature_index[args.target_col]]
        split_speed = x_raw[..., feature_index[args.wind_speed_col]]
        split_pressure = x_raw[..., feature_index[args.pressure_col]]
        split_temperature = x_raw[..., feature_index[args.temperature_col]]
        pressure_for_density = split_pressure * (100.0 if args.pressure_unit == "hpa" else 1.0)
        temperature_for_density = split_temperature + (
            273.15 if args.temperature_unit == "celsius" else 0.0
        )
        split_equivalent = (
            equivalent_wind_speed(
                split_speed, air_density(pressure_for_density, temperature_for_density)
            )
            if args.density_correction
            else split_speed
        )
        p_pc_time = curve.predict(split_equivalent)
        x_model = ((x_raw - feature_mean) / feature_std).astype(np.float32)
        windows = make_windows(
            x_model,
            tensor["mask"],
            tensor["delta_t"],
            target,
            target_mask,
            history_steps=args.history,
            forecast_steps=args.horizon,
        )
        p_pc = _history_only_power_curve(p_pc_time, windows["origin_index"], args.horizon)
        origin_time = tensor["timestamps"][windows["origin_index"]].astype("datetime64[ns]")
        current_y = target[windows["origin_index"]].astype(np.float32)
        current_y_mask = target_mask[windows["origin_index"]].astype(np.float32)
        wind_from = x_raw[
            windows["origin_index"], :, feature_index[args.wind_direction_col]
        ].astype(np.float32)
        if args.wind_direction_mode == "relative_plus_nacelle":
            wind_from += x_raw[
                windows["origin_index"], :, feature_index[args.nacelle_direction_col]
            ]
        wind_from = np.mod(wind_from, 360.0).astype(np.float32)
        np.savez_compressed(
            args.output_dir / f"{split_name}.npz",
            x=windows["x"].astype(np.float32),
            mask=windows["mask"].astype(np.float32),
            delta_t=windows["delta_t"].astype(np.float32),
            y=windows["y"].astype(np.float32),
            y_mask=windows["y_mask"].astype(np.float32),
            p_pc=p_pc,
            current_y=current_y,
            current_y_mask=current_y_mask,
            wind_from=wind_from,
            origin_time=origin_time,
            turbines=tensor["turbines"],
        )

    metadata = {
        "input": str(args.input.resolve()),
        "input_sha256": _sha256(args.input),
        "features": args.features,
        "feature_index": feature_index,
        "split_manifest": manifest,
        "train_means": train_means,
        "feature_scaler": {
            "mean": feature_mean.tolist(),
            "std": feature_std.tolist(),
            "fit_split": "train",
        },
        "power_curve": curve.to_dict(),
        "density_correction": bool(args.density_correction),
        "physical_units": {
            "pressure_input": args.pressure_unit,
            "temperature_input": args.temperature_unit,
            "power_and_rated_power": "caller_declared_same_unit",
            "coordinate": args.coordinate_unit,
        },
        "physical_columns": {
            "target": args.target_col,
            "wind_speed": args.wind_speed_col,
            "pressure": args.pressure_col,
            "temperature": args.temperature_col,
        },
        "resolution_minutes": args.step_minutes,
        "history_steps": args.history,
        "forecast_steps": args.horizon,
        "weather_protocol": "history_only",
        "wind_direction": {
            "mode": args.wind_direction_mode,
            "wind_feature": args.wind_direction_col,
            "nacelle_feature": args.nacelle_direction_col,
            "stored_units": "degrees_meteorological_from",
        },
        "coordinates_saved": coordinates is not None,
        "graph_build": {
            "geographic_k": args.geographic_k,
            "geographic_distance_scale": args.geographic_distance_scale,
            "correlation_k": args.correlation_k,
            "correlation_mode": args.correlation_mode,
            "fit_split": "train",
        },
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output_dir), "splits": manifest}, indent=2))


if __name__ == "__main__":
    main()
