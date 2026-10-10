"""Audit an aligned weather table before any A11 model training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from prwarn.data.weather_protocol import audit_weather_availability


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--protocol", choices=["history_only", "oracle_era5", "issue_time_forecast"],
        required=True,
    )
    parser.add_argument("--origin-column", default="origin_time")
    parser.add_argument("--valid-column", default="valid_time")
    parser.add_argument("--issue-column", default="issue_time")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    table = pd.read_parquet(args.input) if args.input.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(args.input)
    origin = pd.to_datetime(table[args.origin_column], utc=True).to_numpy(
        dtype="datetime64[ns]"
    ).astype("int64")
    valid = pd.to_datetime(table[args.valid_column], utc=True).to_numpy(
        dtype="datetime64[ns]"
    ).astype("int64")
    issue = None
    if args.issue_column in table:
        issue = pd.to_datetime(table[args.issue_column], utc=True).to_numpy(
            dtype="datetime64[ns]"
        ).astype("int64")
    report = audit_weather_availability(
        origin, valid, protocol=args.protocol, issue_time=issue
    ).to_dict()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
