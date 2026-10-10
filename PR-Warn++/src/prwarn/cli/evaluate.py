"""Fit Calib-only conformal correction and evaluate frozen Test risk proxies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import yaml

from prwarn.calibration import (
    AdaptiveConformalIntervals,
    ContextFallbackConformal,
    PerHorizonSplitConformal,
)
from prwarn.eval.metrics import (
    average_precision,
    binary_event_metrics,
    brier_score,
    brier_skill_score,
    crps_ensemble_values,
    dependence_diagnostics,
    energy_score_values,
    interval_metrics,
    ks_distance,
    reliability_bins,
    variogram_score_values,
)
from prwarn.risk.proxies import cvar_shortfall, observed_ramp_event, ramp_probability


def _jsonable_bins(result: dict[str, np.ndarray]) -> dict[str, list | int]:
    def finite_or_none(values: np.ndarray) -> list[float | None]:
        return [float(value) if np.isfinite(value) else None for value in values]

    return {
        "edges": result["edges"].tolist(),
        "count": result["count"].tolist(),
        "predicted": finite_or_none(result["predicted"]),
        "observed": finite_or_none(result["observed"]),
    }


def _ramp_metrics(
    source: h5py.File,
    *,
    threshold: float,
    duration_steps: int,
    direction: str,
    chunk_size: int,
) -> dict:
    probabilities, events = [], []
    scenario_changes, observed_changes = [], []
    lead_values = []
    for left in range(0, len(source["y_farm"]), chunk_size):
        right = min(left + chunk_size, len(source["y_farm"]))
        scenarios = source["farm_scenarios"][left:right].transpose(1, 0, 2)
        current = source["current_y_farm"][left:right]
        probability = ramp_probability(
            scenarios,
            threshold=threshold,
            duration_steps=duration_steps,
            direction=direction,
            current_power=current,
        )
        event, valid = observed_ramp_event(
            source["y_farm"][left:right],
            threshold=threshold,
            duration_steps=duration_steps,
            direction=direction,
            current_power=current,
            future_mask=source["y_farm_mask"][left:right],
            current_mask=source["current_y_farm_mask"][left:right],
        )
        selected = valid & np.isfinite(probability)
        probabilities.append(probability[selected])
        events.append(event[selected])
        extended_scenarios = np.concatenate(
            [np.broadcast_to(current[None, :, None], (scenarios.shape[0], len(current), 1)), scenarios],
            axis=-1,
        )
        extended_truth = np.concatenate([current[:, None], source["y_farm"][left:right]], axis=1)
        change_scenario = np.full_like(scenarios, np.nan, dtype=float)
        change_truth = np.full_like(source["y_farm"][left:right], np.nan, dtype=float)
        for horizon_index in range(scenarios.shape[-1]):
            end = horizon_index + 1
            start = end - duration_steps
            if start >= 0:
                change_scenario[..., horizon_index] = (
                    extended_scenarios[..., end] - extended_scenarios[..., start]
                )
                change_truth[..., horizon_index] = (
                    extended_truth[..., end] - extended_truth[..., start]
                )
        scenario_changes.append(change_scenario[:, selected].reshape(-1))
        observed_changes.append(change_truth[selected])
        for row_probability, row_event, row_valid in zip(probability, event, valid):
            actual = np.flatnonzero(row_event.astype(bool) & row_valid)
            if actual.size == 0:
                continue
            first_actual = int(actual[0])
            predicted = np.flatnonzero((row_probability >= 0.5) & row_valid)
            on_time = predicted[predicted <= first_actual]
            if on_time.size:
                lead_values.append(first_actual - int(on_time[0]))
    probability = np.concatenate(probabilities)
    event = np.concatenate(events)
    if event.size == 0:
        return {"count": 0, "positives": 0, "brier": None, "auprc": None}
    auprc = average_precision(event, probability)
    climatology = np.full_like(probability, float(event.mean()))
    classification = binary_event_metrics(event, probability)
    return {
        "count": int(event.size),
        "positives": int(event.sum()),
        "brier": brier_score(event, probability),
        "brier_skill_score": brier_skill_score(event, probability, climatology),
        "auprc": float(auprc) if np.isfinite(auprc) else None,
        "event_f1": classification["f1"],
        "fnr": classification["fnr"] if np.isfinite(classification["fnr"]) else None,
        "precision": classification["precision"],
        "recall": classification["recall"] if np.isfinite(classification["recall"]) else None,
        "lead_time_steps_mean": float(np.mean(lead_values)) if lead_values else None,
        "lead_time_detected_events": len(lead_values),
        "ramp_ks": ks_distance(
            np.concatenate(scenario_changes), np.concatenate(observed_changes)
        ),
        "reliability": _jsonable_bins(reliability_bins(event, probability)),
    }


def _cvar_summary(source: h5py.File, level: float, chunk_size: int) -> dict:
    values = []
    for left in range(0, len(source["y_farm"]), chunk_size):
        right = min(left + chunk_size, len(source["y_farm"]))
        scenarios = source["farm_scenarios"][left:right].transpose(1, 0, 2)
        _, cvar = cvar_shortfall(scenarios, source["y_det_farm"][left:right], alpha=level)
        values.append(cvar)
    result = np.concatenate(values)
    return {
        "mean": float(np.nanmean(result)),
        "median": float(np.nanmedian(result)),
        "p95": float(np.nanquantile(result, 0.95)),
    }


def _available_context(source: h5py.File) -> np.ndarray:
    """Issue-time context: current power/availability plus deterministic path."""

    return np.concatenate(
        [
            source["current_y_farm"][:].astype(float)[:, None],
            source["current_y_farm_mask"][:].astype(float)[:, None],
            source["y_det_farm"][:].astype(float),
        ],
        axis=1,
    )


def _predicted_regime(source: h5py.File, threshold: float) -> np.ndarray:
    current = source["current_y_farm"][:].astype(float)[:, None]
    path = np.concatenate([current, source["y_det_farm"][:].astype(float)], axis=1)
    return (np.max(np.abs(np.diff(path, axis=1)), axis=1) >= threshold).astype(np.int8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flow-run", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/sdwpf_v3_2.yaml"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument(
        "--calibration-method",
        choices=["static", "aci", "context_fallback"],
        help="Overrides calibration.primary from the config.",
    )
    parser.add_argument("--aci-gamma", type=float)
    parser.add_argument("--aci-rolling-window", type=int)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output_dir = args.output_dir or args.flow_run / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=False)
    alpha = float(config["calibration"]["alpha"])
    configured_method = str(config["calibration"].get("primary", "split_per_horizon"))
    aliases = {"split_per_horizon": "static", "static_split": "static"}
    calibration_method = args.calibration_method or aliases.get(
        configured_method, configured_method
    )
    if calibration_method not in {"static", "aci", "context_fallback"}:
        raise ValueError(f"unsupported calibration method: {calibration_method}")

    with h5py.File(args.flow_run / "scenarios_calib.h5", "r") as calib:
        if not np.isclose(float(calib.attrs["interval_alpha"]), alpha):
            raise ValueError("calibration alpha differs from scenario interval alpha")
        calib_lower = calib["lower"][:]
        calib_upper = calib["upper"][:]
        calib_truth = calib["y_farm"][:]
        calib_mask = calib["y_farm_mask"][:]
        if calibration_method == "static":
            conformal = PerHorizonSplitConformal(alpha=alpha).fit(
                calib_lower, calib_upper, calib_truth, calib_mask
            )
        elif calibration_method == "aci":
            gamma = float(
                args.aci_gamma
                if args.aci_gamma is not None
                else config["calibration"].get("aci_gamma", 0.01)
            )
            rolling_window = (
                args.aci_rolling_window
                if args.aci_rolling_window is not None
                else config["calibration"].get("aci_rolling_window")
            )
            conformal = AdaptiveConformalIntervals(
                alpha=alpha,
                gamma=gamma,
                rolling_window=rolling_window,
            ).fit(calib_lower, calib_upper, calib_truth, calib_mask)
        else:
            farm_capacity = float(calib.attrs["rated_power"]) * int(calib.attrs["n_nodes"])
            subgroup_threshold = float(config["risk"]["ramp_threshold_fractions"][1])
            subgroup_threshold *= farm_capacity
            conformal = ContextFallbackConformal(
                alpha=alpha,
                neighbors=int(config["calibration"].get("context_neighbors", 100)),
                ood_quantile=float(
                    config["calibration"].get("context_ood_quantile", 0.95)
                ),
            ).fit(
                calib_lower,
                calib_upper,
                calib_truth,
                _available_context(calib),
                calib_mask,
                subgroup=_predicted_regime(calib, subgroup_threshold),
            )

    metrics = {
        "calibration": {"method": calibration_method, "alpha": alpha},
        "probabilistic": {},
        "interval": {},
        "ramp": {},
        "cvar": {},
    }
    with h5py.File(args.flow_run / "scenarios_test.h5", "r") as test:
        if not np.isclose(float(test.attrs["interval_alpha"]), alpha):
            raise ValueError("test scenario interval alpha differs from calibration config")
        truth = test["y_farm"][:]
        mask = test["y_farm_mask"][:]
        if calibration_method == "static":
            lower, upper = conformal.transform(test["lower"][:], test["upper"][:])
            metrics["calibration"]["correction"] = conformal.correction_.tolist()
        elif calibration_method == "aci":
            adaptive = conformal.transform_stream(
                test["lower"][:], test["upper"][:], truth, mask
            )
            lower, upper = adaptive.lower, adaptive.upper
            metrics["calibration"].update(
                gamma=conformal.gamma,
                rolling_window=conformal.rolling_window,
                final_alpha=conformal.alpha_state_.tolist(),
                mean_correction=np.mean(adaptive.correction, axis=0).tolist(),
                chronological_predict_then_update=True,
            )
        else:
            contextual = conformal.transform(
                test["lower"][:], test["upper"][:], _available_context(test)
            )
            lower, upper = contextual.lower, contextual.upper
            metrics["calibration"].update(
                mean_correction=np.mean(contextual.correction, axis=0).tolist(),
                fallback_trigger_rate=float(np.mean(contextual.fallback_triggered)),
                ood_distance_threshold=conformal.ood_threshold_,
                empirical_conditional_method=True,
            )
        farm_capacity = float(test.attrs["rated_power"]) * int(test.attrs["n_nodes"])
        crps_total = 0.0
        crps_count = 0
        energy_values = []
        variogram_values = []
        complete_truth = []
        complete_scenarios = []
        for left in range(0, len(truth), args.chunk_size):
            right = min(left + args.chunk_size, len(truth))
            scenarios = test["farm_scenarios"][left:right].transpose(1, 0, 2)
            valid = mask[left:right].astype(bool)
            crps = crps_ensemble_values(truth[left:right], scenarios)
            crps_total += float(crps[valid].sum())
            crps_count += int(valid.sum())
            complete = valid.all(axis=1)
            if complete.any():
                selected_truth = truth[left:right][complete]
                selected_scenarios = scenarios[:, complete]
                energy_values.append(energy_score_values(selected_truth, selected_scenarios))
                variogram_values.append(
                    variogram_score_values(selected_truth, selected_scenarios)
                )
                complete_truth.append(selected_truth)
                complete_scenarios.append(selected_scenarios)
        if crps_count == 0 or not energy_values:
            raise ValueError("Test has no valid probabilistic observations")
        joint_truth = np.concatenate(complete_truth, axis=0)
        joint_scenarios = np.concatenate(complete_scenarios, axis=1)
        metrics["probabilistic"] = {
            "farm_crps": crps_total / crps_count,
            "farm_energy_score": float(np.concatenate(energy_values).mean()),
            "farm_variogram_score": float(np.concatenate(variogram_values).mean()),
            "complete_origin_count": int(joint_truth.shape[0]),
            "dependence": dependence_diagnostics(joint_truth, joint_scenarios),
        }
        metrics["interval"]["overall"] = interval_metrics(
            truth, lower, upper, alpha=alpha, mask=mask, normalization=farm_capacity
        )
        for h in range(truth.shape[1]):
            metrics["interval"][f"h{h + 1}"] = interval_metrics(
                truth[:, h], lower[:, h], upper[:, h], alpha=alpha,
                mask=mask[:, h], normalization=farm_capacity
            )
        for fraction in config["risk"]["ramp_threshold_fractions"]:
            threshold = float(fraction) * farm_capacity
            for duration in config["risk"]["ramp_durations_steps"]:
                for direction in ("up", "down"):
                    key = f"{direction}_frac{float(fraction):g}_steps{int(duration)}"
                    metrics["ramp"][key] = _ramp_metrics(
                        test, threshold=threshold, duration_steps=int(duration),
                        direction=direction, chunk_size=args.chunk_size
                    )
        for level in config["risk"]["cvar_levels"]:
            metrics["cvar"][str(level)] = _cvar_summary(test, float(level), args.chunk_size)
        with h5py.File(output_dir / "calibrated_test.h5", "w") as output:
            output.create_dataset("lower", data=lower.astype(np.float32), compression="lzf")
            output.create_dataset("upper", data=upper.astype(np.float32), compression="lzf")
            output.create_dataset("y_farm", data=truth, compression="lzf")
            output.create_dataset("y_farm_mask", data=mask, compression="lzf")
            output.create_dataset("origin_time", data=test["origin_time"][:], compression="lzf")
            output.attrs["alpha"] = alpha
            output.attrs["calibration_method"] = calibration_method
            output.attrs["calibration_source"] = "scenarios_calib.h5"
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
