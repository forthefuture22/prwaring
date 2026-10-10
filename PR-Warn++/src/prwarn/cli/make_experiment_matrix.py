"""Emit seed-expanded v3.2 experiment contracts as JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prwarn.experiments.matrix import build_experiment_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2025, 2026, 2027, 2028, 2029])
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=[
            "ablation", "ablation_optional", "protocol", "joint_probability",
            "forecast_baseline",
        ],
        help="Omit to emit every experiment group.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jobs = build_experiment_matrix(seeds=args.seeds, groups=args.groups)
    payload = {
        "schema_version": 1,
        "evidence_status": "planned_not_run",
        "job_count": len(jobs),
        "jobs": jobs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), "job_count": len(jobs)}))


if __name__ == "__main__":
    main()
