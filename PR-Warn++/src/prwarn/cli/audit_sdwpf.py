"""Create a machine-readable raw audit before any model training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from prwarn.data.audit import audit_sdwpf_frame


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timestamp-col", default="Tmstamp")
    parser.add_argument("--turbine-col", default="TurbID")
    parser.add_argument("--expected-turbines", type=int, default=134)
    parser.add_argument("--frequency-minutes", type=int, default=10)
    args = parser.parse_args()
    report = audit_sdwpf_frame(
        _read(args.input),
        timestamp_col=args.timestamp_col,
        turbine_col=args.turbine_col,
        expected_turbines=args.expected_turbines,
        frequency_minutes=args.frequency_minutes,
    )
    report["source_path"] = str(args.input.resolve())
    report["source_sha256"] = _sha256(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "gate_ready": report["gate_ready"], "issues": report["issues"]}, indent=2))


if __name__ == "__main__":
    main()
