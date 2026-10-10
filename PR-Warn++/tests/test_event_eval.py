"""Tests for gap 2: extreme-event definition and event-level evaluation CLI."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from prwarn.data.processed import ProcessedSplit
from prwarn.eval.event_definition import (
    fit_extreme_thresholds,
    link_event_episodes,
)
from prwarn.eval.statistics import event_bootstrap_ci

_REPO = Path(__file__).resolve().parents[1]
_T0 = np.datetime64("2025-01-01T00:00:00").astype("datetime64[ns]").astype("int64")
_STEP_NS = 10 * 60 * 1_000_000_000


def _make_split(
    *,
    wind_values: np.ndarray,
    temperature_values: np.ndarray,
    origin_time: np.ndarray,
) -> ProcessedSplit:
    """Build a minimal valid ProcessedSplit (node=2, history=3, feature=4)."""

    n = len(origin_time)
    x = np.zeros((n, 2, 3, 4), dtype=np.float32)
    # Fill every history step; the threshold read uses train.x[..., col_index].
    x[..., 0] = wind_values[:, None, None]
    x[..., 1] = temperature_values[:, None, None]
    mask = np.ones_like(x, dtype=bool)
    return ProcessedSplit(
        x=x,
        mask=mask,
        delta_t=np.zeros_like(x, dtype=np.float32),
        y=np.zeros((n, 2, 3), dtype=np.float32),
        y_mask=np.ones((n, 2, 3), dtype=bool),
        p_pc=np.zeros((n, 2, 3), dtype=np.float32),
        current_y=np.zeros((n, 2), dtype=np.float32),
        current_y_mask=np.ones((n, 2), dtype=bool),
        wind_from=np.zeros((n, 2), dtype=np.float32),
        origin_time=origin_time,
        turbines=np.arange(2, dtype=np.int64),
    ).validate()


def _metadata(*, train_start: str | None = None, train_end: str | None = None) -> dict:
    manifest = None
    if train_start is not None and train_end is not None:
        manifest = {"train": {"start": train_start, "end": train_end}}
    return {
        "feature_index": {"Wspd": 0, "T2m": 1, "Patv": 2, "Sp": 3},
        "feature_scaler": {
            "mean": [0.0, 0.0, 0.0, 0.0],
            "std": [1.0, 1.0, 1.0, 1.0],
            "fit_split": "train",
        },
        "physical_columns": {
            "target": "Patv",
            "wind_speed": "Wspd",
            "pressure": "Sp",
            "temperature": "T2m",
        },
        "physical_units": {"temperature_input": "celsius"},
        "resolution_minutes": 10,
        "split_manifest": manifest or {},
    }


class FitThresholdTests(unittest.TestCase):
    def test_thresholds_match_hand_computed_quantile(self):
        wind = np.linspace(2.0, 20.0, 11)  # std=1/mean=0 -> raw == x
        temp = np.linspace(5.0, 25.0, 11)
        origins = _T0 + np.arange(11) * _STEP_NS
        train = _make_split(wind_values=wind, temperature_values=temp, origin_time=origins)
        result = fit_extreme_thresholds(train, _metadata(), wind_quantile=0.95,
                                       temperature_quantile=0.95)
        # The threshold read flattens over all (node, history) positions.
        repeat = 2 * 3
        self.assertAlmostEqual(result["wind_speed_mps"],
                               np.quantile(np.repeat(wind, repeat), 0.95), places=6)
        self.assertAlmostEqual(result["temperature_c"],
                               np.quantile(np.repeat(temp, repeat), 0.95), places=6)

    def test_rejects_split_outside_train_window(self):
        wind = np.full(8, 5.0)
        temp = np.full(8, 10.0)
        in_window = _T0 + np.arange(8) * _STEP_NS
        out_window = _T0 + (100 + np.arange(8)) * _STEP_NS
        metadata = _metadata(train_start="2024-12-31T00:00:00",
                             train_end="2025-01-01T02:00:00")
        ok = _make_split(wind_values=wind, temperature_values=temp, origin_time=in_window)
        # Inside the declared train window -> accepted.
        fit_extreme_thresholds(ok, metadata)
        bad = _make_split(wind_values=wind, temperature_values=temp, origin_time=out_window)
        with self.assertRaises(ValueError):
            fit_extreme_thresholds(bad, metadata)


class LinkEpisodeTests(unittest.TestCase):
    def test_links_ramp_episodes_and_marks_non_event_minus_one(self):
        # Farm power series P[0..12]; down-ramp 100->40 at index 4, up-ramp back
        # 40->100 at index 8. Capacity=200, fraction 0.20 -> threshold 40.
        power = np.array(
            [100, 100, 100, 100, 40, 40, 40, 40, 100, 100, 100, 100, 100], dtype=float
        )
        n, horizon = 10, 3
        origin_time = _T0 + np.arange(n) * _STEP_NS
        y_farm = np.stack([power[i + 1 : i + 1 + horizon] for i in range(n)])
        current = power[:n]
        thresholds = {"ramp_threshold_fractions": [0.20], "ramp_durations_steps": [1]}
        episode_ids, table = link_event_episodes(
            origin_time,
            y_farm,
            np.ones_like(y_farm, dtype=bool),
            current,
            np.ones(n, dtype=bool),
            thresholds,
            farm_capacity=200.0,
            resolution_minutes=10.0,
            link_max_gap_steps=1,
        )
        # Timeline covers future indices 1..12 -> position for series index k is k-1.
        def pos(index: int) -> int:
            return index - 1

        self.assertEqual(len(table), 2)  # one down episode, one up episode
        down_id = episode_ids[pos(4)]
        up_id = episode_ids[pos(8)]
        self.assertGreaterEqual(down_id, 0)
        self.assertGreaterEqual(up_id, 0)
        self.assertNotEqual(down_id, up_id)
        # Non-event positions are -1.
        for index in (1, 2, 3, 5, 6, 7, 9, 10, 11, 12):
            self.assertEqual(episode_ids[pos(index)], -1, f"index {index} should be -1")
        directions = {row["direction"] for row in table}
        self.assertEqual(directions, {"down", "up"})
        for row in table:
            self.assertIn("start", row)
            self.assertIn("end", row)
            self.assertGreater(row["max_magnitude_pct"], 0.0)
            self.assertGreaterEqual(row["duration_steps"], 1)


class _EventEvalFixture:
    """Build a synthetic flow run + processed data dir in a temp directory."""

    def __init__(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.data_dir = root / "data"
        self.flow_run = root / "run"
        (self.data_dir).mkdir()
        (self.flow_run / "evaluation").mkdir(parents=True)

        power = np.array(
            [100, 100, 100, 100, 40, 40, 40, 40, 100, 100, 100, 100, 100], dtype=float
        )
        self.power = power
        n, horizon = 10, 3
        origin_time = (_T0 + np.arange(n) * _STEP_NS).astype("int64")
        y_farm = np.stack([power[i + 1 : i + 1 + horizon] for i in range(n)]).astype("float32")
        current = power[:n].astype("float32")
        # Deterministic forecast = truth + 2.0 -> RMSE/MAE on event steps == 2.0.
        y_det = (y_farm + 2.0).astype("float32")
        lower = (y_farm - 100.0).astype("float32")
        upper = (y_farm + 100.0).astype("float32")
        mask = np.ones_like(y_farm, dtype="float32")

        with h5py.File(self.flow_run / "scenarios_test.h5", "w") as h5:
            h5["y_farm"] = y_farm
            h5["y_farm_mask"] = mask
            h5["y_det_farm"] = y_det
            h5["current_y_farm"] = current
            h5["current_y_farm_mask"] = np.ones(n, dtype="float32")
            h5["origin_time"] = origin_time
            h5["lower"] = lower
            h5["upper"] = upper
            h5.attrs["rated_power"] = 2.0
            h5.attrs["n_nodes"] = 100

        with h5py.File(self.flow_run / "evaluation" / "calibrated_test.h5", "w") as h5:
            h5["lower"] = lower
            h5["upper"] = upper
            h5["y_farm"] = y_farm
            h5["y_farm_mask"] = mask
            h5["origin_time"] = origin_time

        # Processed train split (minimal) + metadata.
        train = _make_split(
            wind_values=np.linspace(2.0, 20.0, 8).astype("float32"),
            temperature_values=np.linspace(5.0, 25.0, 8).astype("float32"),
            origin_time=(_T0 + np.arange(8) * _STEP_NS).astype("int64"),
        )
        np.savez(
            self.data_dir / "train.npz",
            x=train.x, mask=train.mask, delta_t=train.delta_t, y=train.y,
            y_mask=train.y_mask, p_pc=train.p_pc, current_y=train.current_y,
            current_y_mask=train.current_y_mask, wind_from=train.wind_from,
            origin_time=train.origin_time, turbines=train.turbines,
        )
        (self.data_dir / "metadata.json").write_text(
            json.dumps(_metadata()), encoding="utf-8"
        )

        self.config = root / "config.yaml"
        self.config.write_text(
            "calibration:\n  alpha: 0.10\n"
            "risk:\n  ramp_threshold_fractions: [0.20]\n"
            "  ramp_durations_steps: [1]\n",
            encoding="utf-8",
        )
        self.output = self.flow_run / "evaluation" / "events_metrics.json"

    def run_cli(self) -> subprocess.CompletedProcess:
        return subprocess.run(
            [
                sys.executable, "-m", "prwarn.cli.evaluate_events",
                "--flow-run", str(self.flow_run),
                "--data-dir", str(self.data_dir),
                "--config", str(self.config),
                "--output", str(self.output),
                "--bootstrap-n", "200",
                "--bootstrap-seed", "0",
            ],
            capture_output=True, text=True, cwd=str(_REPO),
        )


class EventCliTests(unittest.TestCase):
    def test_cli_end_to_end_values_and_keys(self):
        fixture = _EventEvalFixture()
        try:
            result = fixture.run_cli()
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            payload = json.loads(fixture.output.read_text(encoding="utf-8"))

            # Literal metric keys that matrix.py / aggregate_seed_metrics expect.
            for key in ("extreme_rmse", "extreme_mae", "extreme_picp",
                        "overall_rmse", "overall_mae", "overall_picp"):
                self.assertIn(key, payload)

            # Hand-computed: det = truth + 2 -> RMSE/MAE == 2 on event steps.
            self.assertAlmostEqual(payload["extreme_rmse"], 2.0, places=4)
            self.assertAlmostEqual(payload["extreme_mae"], 2.0, places=4)
            # Intervals [truth-100, truth+100] always contain truth -> PICP == 1.
            self.assertAlmostEqual(payload["extreme_picp"], 1.0, places=4)

            self.assertEqual(payload["event_counts"]["episodes"], 2)
            self.assertGreater(payload["event_counts"]["in_event_origins"], 0)
            self.assertEqual(payload["event_counts"]["overall_origins"], 10)

            thresholds = payload["thresholds"]
            self.assertIn("wind_speed_mps", thresholds)
            self.assertEqual(thresholds["wind_quantile"], 0.95)
            self.assertIn("temperature_c", thresholds)
            self.assertEqual(thresholds["fitted_on"], "train_only")

            table = payload["event_table"]
            self.assertEqual(len(table), 2)
            for row in table:
                for column in ("event_id", "start", "end", "direction",
                               "max_magnitude_pct", "duration_steps"):
                    self.assertIn(column, row)

            ci = payload["extreme_rmse_ci"]
            self.assertIsNotNone(ci)
            self.assertEqual(ci["n_events"], 2)
            self.assertLessEqual(ci["lower"], ci["estimate"])
            self.assertGreaterEqual(ci["upper"], ci["estimate"])
        finally:
            fixture.tmp.cleanup()

    def test_bootstrap_ci_is_reproducible(self):
        values = np.array([0.0, 0.0, 2.0, 2.0, 1.0, 1.0])
        ids = np.array([3, 3, 7, 7, 9, 9])
        first = event_bootstrap_ci(values, ids, n_boot=300, seed=123)
        second = event_bootstrap_ci(values, ids, n_boot=300, seed=123)
        self.assertEqual(first, second)
        self.assertEqual(first.n_events, 3)


if __name__ == "__main__":
    unittest.main()
