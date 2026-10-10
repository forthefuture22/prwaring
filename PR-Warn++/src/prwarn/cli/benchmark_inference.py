"""Benchmark full deterministic-plus-scenario inference latency and memory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from prwarn.cli.train_deterministic import _loader
from prwarn.data.processed import load_processed_split
from prwarn.models.scenario_checkpoint import (
    load_deterministic_checkpoint,
    load_scenario_checkpoint,
)


def _parameter_count(model: object) -> int:
    if not isinstance(model, torch.nn.Module):
        return 0
    return int(sum(parameter.numel() for parameter in model.parameters()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--scenario-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--scenarios", type=int, default=100)
    parser.add_argument("--sampling-steps", type=int)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=100)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if min(args.batch_size, args.scenarios, args.repetitions) <= 0 or args.warmup < 0:
        raise ValueError("benchmark counts must be positive and warmup non-negative")
    device = torch.device(args.device)
    scenario = load_scenario_checkpoint(args.scenario_run, device)
    deterministic = load_deterministic_checkpoint(scenario.deterministic_run, device)
    config = json.loads(
        (scenario.deterministic_run / "config.json").read_text(encoding="utf-8")
    )
    split = load_processed_split(args.data_dir / "test.npz")
    coordinates = np.load(args.data_dir / "coordinates.npy")
    static = torch.from_numpy(
        np.stack(
            [np.load(args.data_dir / "a_geo.npy"), np.load(args.data_dir / "a_corr.npy")]
        ).astype(np.float32)
    ).to(device)
    loader = _loader(split, split.wind_from, coordinates, config, args, shuffle=False)
    batch = next(iter(loader))
    values = {key: value.to(device) for key, value in batch.items()}

    @torch.no_grad()
    def execute(seed: int) -> None:
        centre = deterministic(
            values["x"], values["mask"], values["delta_t"], static,
            values["directional_graph"], values["p_pc"],
            values.get("future_weather"), values.get("future_weather_mask"),
        )
        scenario.sample_power(
            centre["y_det"], centre["hidden"], centre["adjacency"],
            n_scenarios=args.scenarios, seed=seed,
            sampling_steps=args.sampling_steps,
        )

    for index in range(args.warmup):
        execute(10_000 + index)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    latency = []
    for index in range(args.repetitions):
        started = perf_counter()
        execute(20_000 + index)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        latency.append((perf_counter() - started) * 1000.0)
    values_ms = np.asarray(latency)
    peak_vram = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    )
    payload = {
        "schema_version": 1,
        "method": scenario.method,
        "device": str(device),
        "batch_size": args.batch_size,
        "scenarios": args.scenarios,
        "sampling_steps": args.sampling_steps or scenario.default_sampling_steps,
        "warmup": args.warmup,
        "repetitions": args.repetitions,
        "parameters": {
            "deterministic": _parameter_count(deterministic),
            "scenario": _parameter_count(scenario.model),
        },
        "latency_ms": {
            "p50": float(np.quantile(values_ms, 0.50)),
            "p95": float(np.quantile(values_ms, 0.95)),
            "p99": float(np.quantile(values_ms, 0.99)),
            "mean": float(values_ms.mean()),
        },
        "throughput_origins_per_second": float(
            args.batch_size / (values_ms.mean() / 1000.0)
        ),
        "peak_vram_bytes": peak_vram,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
