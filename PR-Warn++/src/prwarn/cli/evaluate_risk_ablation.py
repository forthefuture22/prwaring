"""Run A10 with Calib-fitted risk composition and frozen Test evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from prwarn.eval.metrics import average_precision, brier_score
from prwarn.risk.proxies import (
    FrozenRiskComposer,
    select_cost_threshold,
    stylized_event_cost,
)


def _diagnostics(data: np.lib.npyio.NpzFile, split: str) -> dict[str, np.ndarray]:
    result = {}
    for name in ("ood", "data_quality"):
        key = f"{split}_{name}"
        if key in data:
            result[name] = np.asarray(data[key], dtype=float)
    return result


def _metrics(event, probability, threshold, risk_config) -> dict[str, float]:
    return {
        "auprc": average_precision(event, probability),
        "brier": brier_score(event, probability),
        "stylized_cost": stylized_event_cost(
            event, probability, threshold=threshold,
            false_negative_cost=float(risk_config["false_negative_cost"]),
            false_positive_cost=float(risk_config["false_positive_cost"]),
        ),
        "threshold_fitted_on_calib": float(threshold),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/sdwpf_v3_2.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--auxiliary-terms", nargs="*", choices=["ood", "data_quality"])
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    risk = config["risk"]
    terms = tuple(args.auxiliary_terms if args.auxiliary_terms is not None else risk["auxiliary_terms"])
    with np.load(args.input, allow_pickle=False) as data:
        calib_event = np.asarray(data["calib_event"], dtype=float)
        calib_core = np.asarray(data["calib_core_probability"], dtype=float)
        test_event = np.asarray(data["test_event"], dtype=float)
        test_core = np.asarray(data["test_core_probability"], dtype=float)
        calib_diagnostics = _diagnostics(data, "calib")
        test_diagnostics = _diagnostics(data, "test")

    composer = FrozenRiskComposer(
        terms,
        ood_weight=float(risk["auxiliary_weights"]["ood"]),
        data_quality_weight=float(risk["auxiliary_weights"]["data_quality"]),
    ).fit(**calib_diagnostics)
    calib_augmented = composer.transform(calib_core, **calib_diagnostics)
    test_augmented = composer.transform(test_core, **test_diagnostics)
    cost_kwargs = {
        "false_negative_cost": float(risk["false_negative_cost"]),
        "false_positive_cost": float(risk["false_positive_cost"]),
    }
    core_threshold = select_cost_threshold(calib_event, calib_core, **cost_kwargs)
    augmented_threshold = select_cost_threshold(
        calib_event, calib_augmented, **cost_kwargs
    )
    report = {
        "protocol": "calib_fit_test_frozen",
        "auxiliary_terms": list(terms),
        "normalization": {
            "location": composer.locations_, "scale": composer.scales_,
        },
        "test": {
            "core": _metrics(test_event, test_core, core_threshold, risk),
            "augmented": _metrics(
                test_event, test_augmented, augmented_threshold, risk
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
