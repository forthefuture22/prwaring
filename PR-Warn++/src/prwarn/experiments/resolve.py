"""Resolve dotted experiment overrides into immutable run configurations."""

from __future__ import annotations

from copy import deepcopy
from typing import Mapping


def apply_dotted_overrides(
    base: Mapping[str, object], overrides: Mapping[str, object]
) -> dict[str, object]:
    result = deepcopy(dict(base))
    for dotted_key, value in overrides.items():
        parts = dotted_key.split(".")
        if not parts or any(not part for part in parts):
            raise ValueError(f"invalid dotted override: {dotted_key!r}")
        cursor = result
        for part in parts[:-1]:
            existing = cursor.get(part)
            if existing is None:
                cursor[part] = {}
            elif not isinstance(existing, dict):
                raise ValueError(f"override crosses non-mapping key: {dotted_key}")
            cursor = cursor[part]
        cursor[parts[-1]] = deepcopy(value)
    return result


def execution_stage(job: Mapping[str, object]) -> str:
    logical_id = str(job["logical_id"])
    if logical_id in {"A1", "A3", "A11"}:
        return "preprocess"
    if logical_id in {"A0", "A2", "A9"}:
        return "deterministic"
    if logical_id in {"A5", "A6", "G6"}:
        return "flow"
    if logical_id in {"A7", "A8"}:
        return "calibration"
    if logical_id == "A10":
        return "risk"
    if logical_id == "A4":
        return "special_ablation"
    if logical_id in {"G1", "G2", "G3"}:
        return "residual_baseline"
    if logical_id in {"G4", "G5"}:
        return "neural_baseline"
    if logical_id in {"B0", "B1"}:
        return "point_reference"
    if logical_id in {"B2", "B3", "B8", "B9"}:
        return "deterministic"
    if logical_id in {"B4", "B5", "B6"}:
        return "marginal_baseline"
    if logical_id == "B7":
        return "deep_ensemble"
    raise ValueError(f"no execution stage for {logical_id}")


def command_template(job: Mapping[str, object], config_path: str) -> list[str]:
    stage = execution_stage(job)
    experiment_id = str(job["experiment_id"])
    variant = str(job["variant"])
    common = ["--config", config_path]
    if stage == "preprocess":
        logical_id = str(job["logical_id"])
        if logical_id == "A11":
            protocol = str(dict(job["overrides"])["data.weather_protocol"])
            if protocol == "history_only":
                return ["USE_ARTIFACT", "<HISTORY_ONLY_PROCESSED_DATA>"]
            return [
                "python", "-m", "prwarn.cli.attach_future_weather",
                "--input-dir", "<HISTORY_ONLY_PROCESSED_DATA>",
                "--weather", "<WEATHER_ARCHIVE>",
                "--output-dir", f"processed/{experiment_id}",
                "--protocol", protocol,
                "--features", "<FUTURE_WEATHER_FEATURES>",
            ]
        command = [
            "python", "-m", "prwarn.cli.preprocess_sdwpf",
            "--input", "<FROZEN_SDWPF_TABLE>",
            "--output-dir", f"processed/{experiment_id}",
            "--rated-power", "<RATED_POWER>",
            "--locations", "<TURBINE_LOCATIONS>",
            "--features", "Wspd", "Wdir", "Ndir", "Pab1", "Pab2", "Pab3",
            "Prtv", "Patv", "Etmp", "Itmp", "Sp", "T2m",
        ]
        if logical_id == "A1" and variant == "density_off":
            command.append("--no-density-correction")
        if logical_id == "A3":
            command.extend(["--correlation-mode", "raw" if variant == "raw_corr" else "difference"])
        return command
    if stage == "deterministic":
        return [
            "python", "-m", "prwarn.cli.train_deterministic", *common,
            "--data-dir", "<PROCESSED_DATA_DIR>", "--experiment-id", experiment_id,
            "--device", "cuda",
        ]
    if stage == "flow":
        return [
            "python", "-m", "prwarn.cli.train_flow", *common,
            "--deterministic-run", "<DETERMINISTIC_RUN>",
            "--experiment-id", experiment_id, "--device", "cuda",
        ]
    if stage == "residual_baseline":
        method = str(dict(job["overrides"])["scenario.method"])
        return [
            "python", "-m", "prwarn.cli.fit_probabilistic_baselines", *common,
            "--deterministic-run", "<DETERMINISTIC_RUN>", "--methods", method,
            "--experiment-id", experiment_id,
        ]
    if stage == "neural_baseline":
        method = str(dict(job["overrides"])["scenario.method"])
        return [
            "python", "-m", "prwarn.cli.train_neural_baseline", *common,
            "--method", method, "--deterministic-run", "<DETERMINISTIC_RUN>",
            "--experiment-id", experiment_id, "--device", "cuda",
        ]
    if stage == "marginal_baseline":
        method = str(dict(job["overrides"])["marginal.method"])
        return [
            "python", "-m", "prwarn.cli.train_marginal_baseline", *common,
            "--method", method, "--deterministic-run", "<DETERMINISTIC_RUN>",
            "--experiment-id", experiment_id, "--device", "cuda",
        ]
    if stage == "point_reference":
        return [
            "python", "-m", "prwarn.cli.evaluate_point_references",
            "--deterministic-run", "<DETERMINISTIC_RUN>",
            "--output", f"outputs/{experiment_id}/metrics.json",
        ]
    if stage == "deep_ensemble":
        return [
            "python", "-m", "prwarn.cli.build_deep_ensemble",
            "--member", "<MEMBER_RUN_1>", "--member", "<MEMBER_RUN_2>",
            "--output-dir", f"outputs/{experiment_id}",
        ]
    if stage == "calibration":
        method = "aci" if variant == "aci" else "context_fallback" if "on" in variant else "static"
        return [
            "python", "-m", "prwarn.cli.evaluate", *common,
            "--flow-run", "<SCENARIO_RUN>", "--calibration-method", method,
            "--output-dir", f"outputs/{experiment_id}",
        ]
    if stage == "special_ablation" and str(job["logical_id"]) == "A4":
        if variant == "direct_cfm":
            return [
                "python", "-m", "prwarn.cli.train_flow", *common,
                "--deterministic-run", "<DETERMINISTIC_RUN>",
                "--experiment-id", experiment_id, "--device", "cuda",
            ]
        return [
            "python", "-m", "prwarn.cli.train_neural_baseline", *common,
            "--method", "vae_cfm", "--deterministic-run", "<DETERMINISTIC_RUN>",
            "--experiment-id", experiment_id, "--device", "cuda",
        ]
    if stage == "risk":
        return [
            "python", "-m", "prwarn.cli.build_risk_ablation_inputs",
            "--scenario-run", "<SCENARIO_RUN>",
            "--deterministic-run", "<DETERMINISTIC_RUN>",
            "--processed-data", "<PROCESSED_DATA_DIR>",
            "--output", f"outputs/{experiment_id}/risk_inputs.npz",
        ]
    return ["MANUAL_GATE", stage, experiment_id, "<REQUIRED_INPUTS_NOT_YET_BOUND>"]


def post_command_templates(
    job: Mapping[str, object], config_path: str
) -> list[list[str]]:
    """Commands that consume the primary run artifact without refitting it."""

    stage = execution_stage(job)
    experiment_id = str(job["experiment_id"])
    if stage == "marginal_baseline":
        return [[
            "python", "-m", "prwarn.cli.evaluate_marginal",
            "--run", f"outputs/{experiment_id}",
            "--output-dir", f"outputs/{experiment_id}/evaluation_cqr",
        ]]
    if stage == "risk":
        return [[
            "python", "-m", "prwarn.cli.evaluate_risk_ablation",
            "--config", config_path,
            "--input", f"outputs/{experiment_id}/risk_inputs.npz",
            "--output", f"outputs/{experiment_id}/metrics.json",
        ]]
    return []
