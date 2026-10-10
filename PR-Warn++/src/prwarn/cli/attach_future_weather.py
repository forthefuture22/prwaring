"""Attach oracle or issue-time future weather to processed SDWPF windows.

The weather table is long-form with one row per turbine/valid time (and per
issue for a forecast archive). All standardization statistics are fitted on
Train only. Issue-time selection always uses the latest issue <= origin.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from prwarn.data.weather_protocol import audit_weather_availability


def _read_table(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _to_epoch_ns(values) -> np.ndarray:
    parsed = pd.to_datetime(values, utc=True)
    return parsed.to_numpy(dtype="datetime64[ns]").astype(np.int64)


def _align_weather(
    origin_time: np.ndarray,
    turbines: np.ndarray,
    horizon: int,
    resolution_minutes: float,
    archive: pd.DataFrame,
    *,
    protocol: str,
    turbine_column: str,
    valid_column: str,
    issue_column: str,
    feature_columns: list[str],
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    origins = np.asarray(origin_time).astype("datetime64[ns]").astype(np.int64)
    step_ns = int(resolution_minutes * 60 * 1_000_000_000)
    batch, nodes = origins.size, turbines.size
    sample = np.repeat(np.arange(batch), nodes * horizon)
    node = np.tile(np.repeat(np.arange(nodes), horizon), batch)
    lead = np.tile(np.arange(1, horizon + 1), batch * nodes)
    request = pd.DataFrame(
        {
            "_sample": sample,
            "_node": node,
            "_lead": lead - 1,
            "_origin": origins[sample],
            turbine_column: turbines[node],
            "_valid": origins[sample] + lead * step_ns,
        }
    )
    weather = archive.copy()
    weather["_valid"] = _to_epoch_ns(weather[valid_column])
    required = {turbine_column, *feature_columns}
    absent = required - set(weather.columns)
    if absent:
        raise KeyError(f"weather archive lacks columns: {sorted(absent)}")
    columns = [turbine_column, "_valid", *feature_columns]
    if protocol == "issue_time_forecast":
        if issue_column not in weather:
            raise KeyError("issue-time protocol requires an issue-time column")
        weather["_issue"] = _to_epoch_ns(weather[issue_column])
        columns.append("_issue")
    elif protocol != "oracle_era5":
        raise ValueError("future weather attachment supports oracle_era5 or issue_time_forecast")
    weather = weather[columns]
    archive_key = [turbine_column, "_valid"]
    if protocol == "issue_time_forecast":
        archive_key.append("_issue")
    if weather.duplicated(archive_key).any():
        raise ValueError(f"weather archive has duplicate key rows: {archive_key}")
    merged = request.merge(weather, on=[turbine_column, "_valid"], how="left")
    key = ["_sample", "_node", "_lead"]
    if protocol == "issue_time_forecast":
        merged = merged[merged["_issue"].isna() | (merged["_issue"] <= merged["_origin"])]
        merged = merged.sort_values([*key, "_issue"], na_position="first")
        merged = merged.drop_duplicates(key, keep="last")
    else:
        if merged.duplicated(key).any():
            raise ValueError("oracle archive has duplicate turbine/valid-time rows")
    # A left merge can lose requests when every issue is after the origin.
    merged = request[key + ["_origin", "_valid"]].merge(
        merged[key + (["_issue"] if "_issue" in merged else []) + feature_columns],
        on=key, how="left",
    )
    shape = (batch, nodes, horizon, len(feature_columns))
    values = np.full(shape, np.nan, dtype=np.float32)
    for feature_index, name in enumerate(feature_columns):
        values[
            merged["_sample"].to_numpy(),
            merged["_node"].to_numpy(),
            merged["_lead"].to_numpy(),
            feature_index,
        ] = pd.to_numeric(merged[name], errors="coerce").to_numpy(np.float32)
    mask = np.isfinite(values)
    selected = mask.any(axis=-1).reshape(-1)
    selected_origin = merged["_origin"].to_numpy()[selected]
    selected_valid = merged["_valid"].to_numpy()[selected]
    selected_issue = (
        merged["_issue"].to_numpy()[selected]
        if protocol == "issue_time_forecast" else None
    )
    audit = audit_weather_availability(
        selected_origin, selected_valid, protocol=protocol,
        issue_time=selected_issue,
    ).to_dict()
    audit["requested_rows"] = int(batch * nodes * horizon)
    audit["matched_rows"] = int(selected.sum())
    audit["coverage"] = float(selected.mean()) if selected.size else 0.0
    audit["feature_coverage"] = {
        name: float(mask[..., index].mean())
        for index, name in enumerate(feature_columns)
    }
    return values, mask, audit


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _write_attached(
    source: dict[str, np.ndarray], path: Path, values: np.ndarray,
    mask: np.ndarray, mean: np.ndarray, scale: np.ndarray,
) -> None:
    filled = np.where(mask, values, mean).astype(np.float32)
    standardized = ((filled - mean) / scale).astype(np.float32)
    np.savez_compressed(
        path, **source, future_weather=standardized,
        future_weather_mask=mask.astype(np.float32),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--weather", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", choices=["oracle_era5", "issue_time_forecast"], required=True)
    parser.add_argument("--features", nargs="+", required=True)
    parser.add_argument("--turbine-column", default="TurbID")
    parser.add_argument("--valid-column", default="valid_time")
    parser.add_argument("--issue-column", default="issue_time")
    parser.add_argument("--minimum-coverage", type=float, default=0.95)
    args = parser.parse_args()
    if not 0.0 <= args.minimum_coverage <= 1.0:
        raise ValueError("minimum coverage must be in [0,1]")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    metadata = json.loads((args.input_dir / "metadata.json").read_text(encoding="utf-8"))
    weather = _read_table(args.weather)
    horizon = int(metadata["forecast_steps"])
    resolution = float(metadata["resolution_minutes"])
    audits: dict[str, object] = {}

    train = _load_npz(args.input_dir / "train.npz")
    train_values, train_mask, audits["train"] = _align_weather(
        train["origin_time"], train["turbines"], horizon, resolution, weather,
        protocol=args.protocol, turbine_column=args.turbine_column,
        valid_column=args.valid_column, issue_column=args.issue_column,
        feature_columns=args.features,
    )
    if float(audits["train"]["coverage"]) < args.minimum_coverage:
        raise ValueError("Train future-weather coverage is below minimum")
    counts = train_mask.sum(axis=(0, 1, 2))
    if np.any(counts == 0):
        names = [args.features[i] for i in np.flatnonzero(counts == 0)]
        raise ValueError(f"future weather features absent on Train: {names}")
    mean = np.where(train_mask, train_values, 0.0).sum(axis=(0, 1, 2)) / counts
    centered = np.where(train_mask, train_values - mean, 0.0)
    scale = np.sqrt(np.square(centered).sum(axis=(0, 1, 2)) / counts)
    scale = np.maximum(scale, 1e-6).astype(np.float32)
    mean = mean.astype(np.float32)
    _write_attached(train, args.output_dir / "train.npz", train_values, train_mask, mean, scale)
    for split in ("val", "calib", "test"):
        source = _load_npz(args.input_dir / f"{split}.npz")
        values, mask, audits[split] = _align_weather(
            source["origin_time"], source["turbines"], horizon, resolution, weather,
            protocol=args.protocol, turbine_column=args.turbine_column,
            valid_column=args.valid_column, issue_column=args.issue_column,
            feature_columns=args.features,
        )
        if float(audits[split]["coverage"]) < args.minimum_coverage:
            raise ValueError(f"{split} future-weather coverage is below minimum")
        _write_attached(source, args.output_dir / f"{split}.npz", values, mask, mean, scale)
    for name in ("coordinates.npy", "a_geo.npy", "a_corr.npy"):
        source = args.input_dir / name
        if source.exists():
            shutil.copy2(source, args.output_dir / name)
    metadata["weather_protocol"] = args.protocol
    metadata["future_weather"] = {
        "archive": str(args.weather.resolve()), "archive_sha256": _sha256(args.weather),
        "features": args.features, "minimum_coverage": args.minimum_coverage,
        "train_mean": mean.tolist(), "train_scale": scale.tolist(),
        "selection": "latest_issue_at_or_before_origin" if args.protocol == "issue_time_forecast" else "oracle_valid_time",
        "audits": audits,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(args.output_dir), "audits": audits}, indent=2))


if __name__ == "__main__":
    main()
