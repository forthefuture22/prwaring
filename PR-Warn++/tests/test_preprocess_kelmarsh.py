"""Tests for the Kelmarsh dataset adapter (gap 1).

Covers: generic row-validity labels, pluggable reason-code policy, metadata
identity keys, config consistency, and an end-to-end smoke through the shared
rigid-grid -> bundle -> window -> npz pipeline.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from functools import partial
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

from prwarn.cli.preprocess_kelmarsh import main
from prwarn.data.grid import (
    ReasonCode,
    build_rigid_grid,
    build_sdwpf_bundle,
    generic_scada_reason_codes,
)
from prwarn.data.processed import load_processed_split

REPO_ROOT = Path(__file__).resolve().parents[1]

FEATURES = ["Wind Speed", "Wind Direction", "Active Power", "Ambient Temp"]


def _make_kelmarsh_frame(n_steps: int = 600) -> pd.DataFrame:
    """Synthetic Kelmarsh-style wide table (6 turbines, 10-min cadence)."""

    rows = []
    base = pd.Timestamp("2019-01-01 00:00")
    for step in range(n_steps):
        speed = 7.5 + 4.5 * np.sin(step / 30.0)  # ~3 .. 12 m/s
        wdir = (180.0 + 60.0 * np.sin(step / 200.0)) % 360.0
        for wtg in (1, 2, 3, 4, 5, 6):
            power = min(2.05, 0.02 * speed**3 + 0.05 * wtg)
            rows.append(
                {
                    "Date/Time": base + pd.Timedelta(minutes=10 * step),
                    "WTG": wtg,
                    "Wind Speed": speed + 0.1 * (wtg - 3),
                    "Wind Direction": wdir,
                    "Active Power": power,
                    "Ambient Temp": 8.0 + 2.0 * np.sin(step / 500.0),
                }
            )
    frame = pd.DataFrame(rows)
    # Inject realistic quality issues so the generic policy is actually exercised.
    missing_idx = frame.sample(50, random_state=0).index
    frame.loc[missing_idx, "Active Power"] = np.nan
    bad_speed_idx = frame.sample(20, random_state=1).index
    frame.loc[bad_speed_idx, "Wind Speed"] = 30.0  # anemometer over range
    return frame


def _run_smoke(directory: Path) -> Path:
    """Run preprocess_kelmarsh.main() on a synthetic CSV; return output dir."""

    frame = _make_kelmarsh_frame(n_steps=600)
    raw = directory / "kelmarsh.csv"
    frame.to_csv(raw, index=False)
    out = directory / "processed"
    argv = [
        "preprocess_kelmarsh",
        "--input", str(raw),
        "--output-dir", str(out),
        "--rated-power", "2.05",
        "--curve-bins", "8",
        "--curve-min-count", "3",
        "--history", "24",
        "--horizon", "6",
        "--features", *FEATURES,
    ]
    with patch.object(sys, "argv", argv):
        main()
    return out


class GenericReasonCodeTests(unittest.TestCase):
    def test_generic_reason_codes_labels_and_shapes(self):
        rigid = build_rigid_grid(
            _make_kelmarsh_frame(n_steps=120),
            timestamp_col="Date/Time",
            turbine_col="WTG",
        )
        codes = generic_scada_reason_codes(
            rigid, power_col="Active Power", wind_speed_col="Wind Speed"
        )
        # Shape contract identical to the SDWPF path: one int8 code per rigid row.
        self.assertEqual(codes.shape, (len(rigid),))
        self.assertEqual(codes.dtype, np.int8)
        self.assertTrue((codes == ReasonCode.VALID).any())
        self.assertTrue((codes == ReasonCode.MISSING).any())
        self.assertTrue((codes == ReasonCode.ABNORMAL).any())
        # No UNKNOWN bucket is produced without SDWPF-specific channels.
        self.assertFalse((codes == ReasonCode.UNKNOWN).any())

    def test_build_bundle_accepts_generic_reason_fn(self):
        rigid = build_rigid_grid(
            _make_kelmarsh_frame(n_steps=120),
            timestamp_col="Date/Time",
            turbine_col="WTG",
        )
        reason_fn = partial(
            generic_scada_reason_codes,
            power_col="Active Power",
            wind_speed_col="Wind Speed",
        )
        bundle = build_sdwpf_bundle(
            rigid,
            FEATURES,
            timestamp_col="Date/Time",
            turbine_col="WTG",
            reason_code_fn=reason_fn,
        )
        # mask/delta_t shape contract matches the default SDWPF bundle layout.
        self.assertEqual(bundle["mask"].ndim, 2)
        self.assertEqual(bundle["mask"].shape[1], len(FEATURES))
        self.assertEqual(bundle["delta_t"].shape, bundle["mask"].shape)
        self.assertEqual(bundle["reason_code"].shape, (len(rigid),))


class ConfigTests(unittest.TestCase):
    def test_kelmarsh_config_keys_and_protocol_consistency(self):
        km = yaml.safe_load((REPO_ROOT / "configs" / "kelmarsh_v1.yaml").read_text(encoding="utf-8"))
        sd = yaml.safe_load((REPO_ROOT / "configs" / "sdwpf_v3_2.yaml").read_text(encoding="utf-8"))
        d = km["data"]
        for key in (
            "dataset",
            "resolution_minutes",
            "history_steps",
            "forecast_steps",
            "split",
            "reason_code_strategy",
            "wind_direction_mode",
            "dataset_license",
        ):
            self.assertIn(key, d, f"kelmarsh data section missing {key}")
        self.assertEqual(d["dataset"], "kelmarsh_v1")
        self.assertEqual(d["reason_code_strategy"], "generic_scada")
        self.assertEqual(d["wind_direction_mode"], "global")
        self.assertEqual(d["dataset_license"], "CC-BY-4.0")
        # Same split/window protocol as the main experiment (cross-farm comparability).
        self.assertEqual(d["split"], sd["data"]["split"])
        self.assertEqual(d["history_steps"], sd["data"]["history_steps"])
        self.assertEqual(d["forecast_steps"], sd["data"]["forecast_steps"])
        self.assertEqual(d["resolution_minutes"], sd["data"]["resolution_minutes"])


class SmokePipelineTests(unittest.TestCase):
    def test_metadata_identity_keys_and_disjoint_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            out = _run_smoke(Path(directory))
            meta = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
            for key in (
                "dataset_name",
                "dataset_license",
                "dataset_origin_url",
                "input_sha256",
                "published_columns",
                "derived_columns",
                "feature_scaler",
            ):
                self.assertIn(key, meta, f"metadata.json missing {key}")
            self.assertEqual(meta["dataset_name"], "kelmarsh_v1")
            self.assertEqual(meta["feature_scaler"]["fit_split"], "train")
            overlap = set(meta["published_columns"]) & set(meta["derived_columns"])
            self.assertEqual(overlap, set(), f"published/derived overlap: {overlap}")

    def test_full_pipeline_loads_and_validates(self):
        with tempfile.TemporaryDirectory() as directory:
            out = _run_smoke(Path(directory))
            for name in ("train", "val", "calib", "test"):
                split = load_processed_split(out / f"{name}.npz")
                split.validate()  # raises on any contract violation
            self.assertTrue((out / "a_corr.npy").exists())
            self.assertTrue((out / "metadata.json").exists())


if __name__ == "__main__":
    unittest.main()
