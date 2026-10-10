"""Aggregate common numeric metrics from immutable seed-run JSON artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prwarn.eval.statistics import aggregate_seed_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if len({path.resolve() for path in args.metrics}) != len(args.metrics):
        raise ValueError("metric paths must be unique")
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in args.metrics]
    payload = {
        "schema_version": 1,
        "run_count": len(runs),
        "sources": [str(path.resolve()) for path in args.metrics],
        "summary": aggregate_seed_metrics(runs),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "run_count": len(runs)}))


if __name__ == "__main__":
    main()
