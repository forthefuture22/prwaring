"""Canonical ablation and probabilistic benchmark matrix.

This module records *comparisons to run*, never expected wins. The distinction
keeps planned experiments separate from evidence produced by actual logs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class ExperimentSpec:
    logical_id: str
    group: str
    comparison: str
    question: str
    primary_metrics: tuple[str, ...]
    variants: tuple[tuple[str, Mapping[str, object]], ...]
    server_required: bool = True


def _core_ablation_specs() -> tuple[ExperimentSpec, ...]:
    return (
        ExperimentSpec(
            "A0", "ablation", "Direct power vs P_pc + residual",
            "Is residual forecasting around the weak-physics center necessary?",
            ("extreme_rmse", "extreme_mae", "ramp_f1"),
            (
                ("direct_power", {"model.target_mode": "direct_power"}),
                ("physics_residual", {"model.target_mode": "physics_residual"}),
            ),
        ),
        ExperimentSpec(
            "A1", "ablation", "Density correction off/on",
            "Does equivalent wind speed improve the physical center?",
            ("power_curve_error", "seasonal_rmse", "extreme_rmse"),
            (
                ("density_off", {"physics.density_correction": False}),
                ("density_on", {"physics.density_correction": True}),
            ),
        ),
        ExperimentSpec(
            "A2", "ablation", "A_geo -> +A_corr -> +A_dir -> +A_adp",
            "Does each graph add independent value?",
            ("rmse", "energy_score", "graph_weights"),
            (
                ("geo", {"graphs.enabled": ["geo"]}),
                ("geo_corr", {"graphs.enabled": ["geo", "corr"]}),
                ("geo_corr_dir", {"graphs.enabled": ["geo", "corr", "directional"]}),
                (
                    "all_graphs",
                    {"graphs.enabled": ["geo", "corr", "directional", "adaptive"]},
                ),
            ),
        ),
        ExperimentSpec(
            "A3", "ablation", "Raw correlation vs difference correlation",
            "Is difference correlation worth retaining?",
            ("rmse", "missing_stress_rmse"),
            (
                ("raw_corr", {"graphs.correlation_mode": "raw"}),
                ("difference_corr", {"graphs.correlation_mode": "difference"}),
            ),
        ),
        ExperimentSpec(
            "A4", "ablation", "Direct CFM vs VAE-CFM",
            "Does the VAE stage provide enough value to justify it?",
            ("energy_score", "variogram_score", "ramp_ks", "latency", "peak_vram"),
            (
                ("direct_cfm", {"flow.variant": "direct_cfm"}),
                ("vae_cfm", {"flow.variant": "vae_cfm"}),
            ),
        ),
        ExperimentSpec(
            "A5", "ablation_optional", "CFM vs genuine OT-CFM",
            "Does genuine optimal-transport coupling improve quality or efficiency?",
            ("energy_score", "ode_steps", "training_time"),
            (
                ("independent_cfm", {"flow.coupling": "independent"}),
                ("genuine_ot_cfm", {"flow.coupling": "optimal_transport"}),
            ),
        ),
        ExperimentSpec(
            "A6", "ablation", "4/8/16/32 ODE steps",
            "What is the sampling speed-quality Pareto frontier?",
            ("energy_score", "variogram_score", "latency", "throughput"),
            tuple((f"steps_{steps}", {"flow.steps": steps}) for steps in (4, 8, 16, 32)),
        ),
        ExperimentSpec(
            "A7", "ablation", "Static conformal vs ACI",
            "Does chronological adaptation help under distribution shift?",
            ("rolling_picp", "interval_width"),
            (
                ("static", {"calibration.primary": "split_per_horizon"}),
                ("aci", {"calibration.primary": "aci"}),
            ),
        ),
        ExperimentSpec(
            "A8", "ablation", "Context/fallback off/on",
            "Does empirical out-of-distribution fallback improve extreme coverage?",
            ("extreme_picp", "interval_width", "fallback_trigger_rate"),
            (
                (
                    "context_fallback_off",
                    {"calibration.primary": "split_per_horizon"},
                ),
                (
                    "context_fallback_on",
                    {"calibration.primary": "context_fallback"},
                ),
            ),
        ),
        ExperimentSpec(
            "A9", "ablation", "X-only vs X+M vs X+M+DeltaT",
            "Do missingness patterns contain useful predictive information?",
            ("missing_stress_crps", "missing_stress_rmse"),
            (
                ("x_only", {"model.missing_inputs": []}),
                ("x_mask", {"model.missing_inputs": ["mask"]}),
                (
                    "x_mask_delta",
                    {"model.missing_inputs": ["mask", "delta_t"]},
                ),
            ),
        ),
        ExperimentSpec(
            "A10", "ablation", "Risk without vs with OOD/data terms",
            "Do auxiliary diagnostics improve event skill?",
            ("auprc", "brier", "stylized_cost"),
            (
                ("risk_core", {"risk.auxiliary_terms": []}),
                ("risk_ood_data", {"risk.auxiliary_terms": ["ood", "data_quality"]}),
            ),
        ),
        ExperimentSpec(
            "A11", "protocol", "History-only vs Oracle ERA5 vs true forecast",
            "What is the gap caused by future-weather information availability?",
            ("rmse", "crps", "energy_score", "ramp_metrics"),
            (
                ("history_only", {"data.weather_protocol": "history_only"}),
                ("oracle_era5", {"data.weather_protocol": "oracle_era5"}),
                ("issue_time_forecast", {"data.weather_protocol": "issue_time_forecast"}),
            ),
        ),
    )


def _generative_specs() -> tuple[ExperimentSpec, ...]:
    metrics = ("crps", "energy_score", "variogram_score", "ramp_brier", "ramp_ks")
    definitions = (
        ("G1", "Gaussian residual", "gaussian_residual", "Parametric residual reference"),
        ("G2", "Residual bootstrap", "residual_bootstrap", "Nonparametric field resampling"),
        (
            "G3",
            "Gaussian copula",
            "gaussian_copula",
            "Empirical marginals with Gaussian dependence",
        ),
        ("G4", "CVAE", "cvae", "Latent-variable joint scenario baseline"),
        ("G5", "Diffusion/DDIM", "ddim", "Iterative score-based scenario baseline"),
        ("G6", "Direct CFM", "direct_cfm", "Proposed direct conditional flow matching"),
    )
    return tuple(
        ExperimentSpec(
            logical_id,
            "joint_probability",
            title,
            question,
            metrics,
            ((method, {"scenario.method": method}),),
        )
        for logical_id, title, method, question in definitions
    )


def _forecast_baseline_specs() -> tuple[ExperimentSpec, ...]:
    """Point and marginal references that complement G1--G6 joint methods."""

    metrics = ("mae", "rmse", "r2_nse", "picp", "pinaw", "pinball")
    definitions = (
        ("B0", "Persistence", "point_reference", {"baseline.method": "persistence"}),
        ("B1", "Physical power curve", "point_reference", {"baseline.method": "p_pc"}),
        ("B2", "GRU", "deterministic", {"model.architecture": "gru"}),
        ("B3", "TCN", "deterministic", {"model.architecture": "tcn"}),
        ("B4", "Marginal Gaussian", "marginal", {"marginal.method": "gaussian"}),
        ("B5", "Marginal Student-t", "marginal", {"marginal.method": "student_t"}),
        ("B6", "Quantile + CQR", "marginal", {"marginal.method": "quantile"}),
        ("B7", "Deep ensemble", "ensemble", {"scenario.method": "deep_ensemble"}),
        (
            "B8", "Graph WaveNet style", "deterministic",
            {"model.architecture": "graphwavenet"},
        ),
        ("B9", "AGCRN style", "deterministic", {"model.architecture": "agcrn"}),
    )
    return tuple(
        ExperimentSpec(
            logical_id, "forecast_baseline", title,
            "How does the proposed stack compare with this controlled reference?",
            metrics, ((stage, overrides),),
        )
        for logical_id, title, stage, overrides in definitions
    )
def experiment_specs() -> tuple[ExperimentSpec, ...]:
    return _core_ablation_specs() + _generative_specs() + _forecast_baseline_specs()


def make_experiment_id(
    logical_id: str,
    variant: str,
    seed: int,
    overrides: Mapping[str, object],
) -> str:
    """Build a stable human-readable ID from a canonical JSON payload."""

    payload = {
        "logical_id": logical_id,
        "variant": variant,
        "seed": int(seed),
        "overrides": overrides,
    }
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    digest = sha256(encoded.encode("utf-8")).hexdigest()[:10]
    safe_variant = "".join(char if char.isalnum() else "-" for char in variant).strip("-")
    return f"{logical_id}-{safe_variant}-s{seed}-{digest}"


def build_experiment_matrix(
    *,
    seeds: Sequence[int] = (2025, 2026, 2027, 2028, 2029),
    groups: Iterable[str] | None = None,
) -> list[dict[str, object]]:
    """Expand logical comparisons into seed-specific executable run contracts."""

    if not seeds:
        raise ValueError("at least one seed is required")
    if len(set(int(seed) for seed in seeds)) != len(seeds):
        raise ValueError("seeds must be unique")
    selected_groups = set(groups) if groups is not None else None
    jobs: list[dict[str, object]] = []
    for spec in experiment_specs():
        if selected_groups is not None and spec.group not in selected_groups:
            continue
        for variant, overrides in spec.variants:
            for seed in seeds:
                job = asdict(spec)
                job.pop("variants")
                job.update(
                    variant=variant,
                    seed=int(seed),
                    overrides=dict(overrides),
                    experiment_id=make_experiment_id(
                        spec.logical_id, variant, int(seed), overrides
                    ),
                    evidence_status="planned_not_run",
                )
                job["primary_metrics"] = list(spec.primary_metrics)
                jobs.append(job)
    return jobs
