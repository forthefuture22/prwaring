"""Materialize matrix jobs into immutable JSON configs and command manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from prwarn.experiments.resolve import (
    apply_dotted_overrides,
    command_template,
    execution_stage,
    post_command_templates,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, default=Path("configs/sdwpf_v3_2.yaml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--logical-id", action="append")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    base = yaml.safe_load(args.base_config.read_text(encoding="utf-8"))
    selected = set(args.logical_id or [])
    manifest = []
    for job in matrix["jobs"]:
        if selected and job["logical_id"] not in selected:
            continue
        config = apply_dotted_overrides(base, job["overrides"])
        config["experiment"]["seed"] = int(job["seed"])
        config["experiment"]["name"] = str(job["experiment_id"])
        config["experiment"]["logical_id"] = str(job["logical_id"])
        config["experiment"]["variant"] = str(job["variant"])
        path = args.output_dir / f"{job['experiment_id']}.json"
        path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
        manifest.append(
            {
                "experiment_id": job["experiment_id"],
                "logical_id": job["logical_id"],
                "stage": execution_stage(job),
                "config": str(path),
                "command": command_template(job, str(path)),
                "post_commands": post_command_templates(job, str(path)),
                "evidence_status": "planned_not_run",
            }
        )
    (args.output_dir / "manifest.json").write_text(
        json.dumps({"jobs": manifest, "count": len(manifest)}, indent=2), encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(args.output_dir), "count": len(manifest)}))


if __name__ == "__main__":
    main()
