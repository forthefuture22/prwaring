"""Gap-3 gate-intervention unit and smoke tests (C2-6 / C2-1).

Covers:
  1. learned gate mode is bit-for-bit the pre-intervention behaviour.
  2. uniform mode yields equal, normalised weights over enabled graphs.
  3. gate_hidden_dim decouples the gate MLP width (None falls back to hidden_dim).
  4. A12 expands to a stable planned run contract.
  5. evaluate_gate_intervention CLI smoke on a random-init checkpoint.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import unittest

from prwarn.models.backbone import DynamicMultiGraphResidualForecaster


def _tiny_kwargs(**overrides) -> dict:
    kwargs = dict(
        n_nodes=4, n_features=3, forecast_steps=2, n_static_graphs=2,
        hidden_dim=8,
    )
    kwargs.update(overrides)
    return kwargs


def _fake_batch(batch: int = 2):
    x = torch.randn(batch, 4, 6, 3)
    mask = torch.ones(batch, 4, 6, 3)
    delta_t = torch.zeros(batch, 4, 6, 3)
    static = torch.rand(2, 4, 4)
    static = static / static.sum(-1, keepdim=True)
    directional = torch.rand(batch, 4, 4)
    directional = directional / directional.sum(-1, keepdim=True)
    p_pc = torch.rand(batch, 4, 2)
    return x, mask, delta_t, static, directional, p_pc


class BackboneGateModeTests(unittest.TestCase):
    def test_learned_default_is_bitwise_regression(self):
        x, mask, dt, static, directional, p_pc = _fake_batch()

        def build():
            torch.manual_seed(1234)
            return DynamicMultiGraphResidualForecaster(**_tiny_kwargs()).eval()

        def build_explicit_learned():
            torch.manual_seed(1234)
            return DynamicMultiGraphResidualForecaster(
                gate_mode="learned", gate_hidden_dim=None, **_tiny_kwargs()
            ).eval()

        a, b = build(), build_explicit_learned()
        with torch.no_grad():
            out_a = a(x, mask, dt, static, directional, p_pc)
            out_b = b(x, mask, dt, static, directional, p_pc)
        self.assertTrue(torch.equal(out_a["y_det"], out_b["y_det"]))
        self.assertTrue(torch.equal(out_a["graph_weights"], out_b["graph_weights"]))

    def test_uniform_weights_equal_and_normalised(self):
        torch.manual_seed(0)
        model = DynamicMultiGraphResidualForecaster(
            gate_mode="uniform", **_tiny_kwargs()
        ).eval()
        x, mask, dt, static, directional, p_pc = _fake_batch()
        with torch.no_grad():
            weights = model(x, mask, dt, static, directional, p_pc)["graph_weights"]
        # All four graphs enabled => each column is 1/4.
        expected = torch.full_like(weights, 0.25)
        self.assertTrue(torch.allclose(weights, expected, atol=1e-6))
        self.assertTrue(torch.allclose(weights.sum(dim=-1), torch.ones(2), atol=1e-6))

    def test_gate_hidden_dim_decouples_width(self):
        torch.manual_seed(0)
        default = DynamicMultiGraphResidualForecaster(**_tiny_kwargs(hidden_dim=8))
        wide = DynamicMultiGraphResidualForecaster(
            gate_hidden_dim=32, **_tiny_kwargs(hidden_dim=8)
        )
        self.assertEqual(default.gate[0].out_features, 8)
        self.assertEqual(wide.gate[0].out_features, 32)
        # The second linear projects back to n_static_graphs + 2 columns.
        self.assertEqual(wide.gate[-1].out_features, 4)

    def test_parameter_matched_keeps_param_count_and_freezes_gate(self):
        torch.manual_seed(0)
        learned = DynamicMultiGraphResidualForecaster(**_tiny_kwargs())
        matched = DynamicMultiGraphResidualForecaster(
            gate_mode="parameter_matched", **_tiny_kwargs()
        )
        self.assertEqual(
            sum(p.numel() for p in learned.parameters()),
            sum(p.numel() for p in matched.parameters()),
        )
        self.assertTrue(all(not p.requires_grad for p in matched.gate.parameters()))

    def test_invalid_gate_mode_rejected(self):
        with self.assertRaises(ValueError):
            DynamicMultiGraphResidualForecaster(
                gate_mode="frozen_mean", **_tiny_kwargs()
            )


class MatrixA12Tests(unittest.TestCase):
    def test_a12_contract_stable_and_planned(self):
        from prwarn.experiments.matrix import build_experiment_matrix

        first = build_experiment_matrix(seeds=[2025, 2026])
        a12 = [j for j in first if j["logical_id"] == "A12"]
        self.assertEqual({j["variant"] for j in a12},
                         {"learned", "uniform", "parameter_matched"})
        self.assertTrue(all(j["evidence_status"] == "planned_not_run" for j in a12))
        # Stable experiment_id across rebuilds.
        second = build_experiment_matrix(seeds=[2025, 2026])
        self.assertEqual(first, second)
        learned = [j for j in a12 if j["variant"] == "learned"][0]
        self.assertTrue(learned["experiment_id"].startswith("A12-learned-s2025-"))

    def test_a12_stage_and_post_commands(self):
        from prwarn.experiments.matrix import build_experiment_matrix
        from prwarn.experiments.resolve import (
            command_template, execution_stage, post_command_templates,
        )

        jobs = [j for j in build_experiment_matrix(seeds=[1]) if j["logical_id"] == "A12"]
        for job in jobs:
            self.assertEqual(execution_stage(job), "deterministic")
            self.assertEqual(command_template(job, "cfg.yaml")[0], "python")
        learned = [j for j in jobs if j["variant"] == "learned"][0]
        posts = post_command_templates(learned, "cfg.yaml")
        self.assertEqual(len(posts), 2)
        modes = {c[c.index("--mode") + 1] for c in posts}
        self.assertEqual(modes, {"frozen_mean", "shuffled"})


def _fabricate_data_dir(root: Path) -> tuple[Path, Path]:
    """Build a minimal processed data dir + config + learned checkpoint."""

    root.mkdir(parents=True, exist_ok=True)
    n_nodes, n_feat, horizon, history = 4, 3, 2, 6
    n_static = 2

    coordinates = np.random.RandomState(0).rand(n_nodes, 2).astype(np.float32) * 1000.0
    np.save(root / "coordinates.npy", coordinates)
    for name in ("a_geo", "a_corr"):
        adj = np.random.RandomState(1).rand(n_nodes, n_nodes).astype(np.float32)
        adj = adj / adj.sum(axis=1, keepdims=True)
        np.save(root / f"{name}.npy", adj)

    rng = np.random.RandomState(2)
    sample = 6
    def _full():
        return rng.rand(sample, n_nodes, history, n_feat).astype(np.float32)

    np.savez(
        root / "val.npz",
        x=_full(),
        mask=np.ones((sample, n_nodes, history, n_feat), dtype=bool),
        delta_t=rng.rand(sample, n_nodes, history, n_feat).astype(np.float32),
        y=rng.rand(sample, n_nodes, horizon).astype(np.float32),
        y_mask=np.ones((sample, n_nodes, horizon), dtype=bool),
        p_pc=rng.rand(sample, n_nodes, horizon).astype(np.float32),
        current_y=rng.rand(sample, n_nodes).astype(np.float32),
        current_y_mask=np.ones((sample, n_nodes), dtype=bool),
        wind_from=rng.rand(sample, n_nodes).astype(np.float32) * 360.0,
        origin_time=(
            np.datetime64("2025-01-01T00:00:00")
            + np.array([10 * i for i in range(sample)], dtype="timedelta64[m]")
        ).astype("datetime64[ns]"),
        turbines=np.arange(n_nodes).astype(np.int64),
    )

    config_path = root / "mini.yaml"
    config_path.write_text(
        "graphs:\n"
        "  direction_distance_scale: 1000.0\n"
        "  direction_sigma_degrees: 30.0\n"
        "  direction_sector_degrees: 90.0\n",
        encoding="utf-8",
    )

    torch.manual_seed(7)
    model_config = dict(
        n_nodes=n_nodes, n_features=n_feat, forecast_steps=horizon,
        n_static_graphs=n_static, hidden_dim=8, rated_power=1.0,
    )
    model = DynamicMultiGraphResidualForecaster(**model_config)
    checkpoint = {"model_state": model.state_dict(), "model_config": model_config}
    ckpt_path = root / "best.pt"
    torch.save(checkpoint, ckpt_path)
    return ckpt_path, config_path


class CLISmokeTests(unittest.TestCase):
    def test_evaluate_gate_intervention_runs(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data"
            ckpt_path, config_path = _fabricate_data_dir(data_dir)
            out_dir = Path(tmp) / "out"
            env = dict(os.environ)
            cmd = [
                sys.executable, "-m", "prwarn.cli.evaluate_gate_intervention",
                "--checkpoint", str(ckpt_path),
                "--data-dir", str(data_dir),
                "--config", str(config_path),
                "--output-dir", str(out_dir),
                "--mode", "frozen_mean",
                "--batch-size", "2",
                "--device", "cpu",
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
            self.assertEqual(
                proc.returncode, 0,
                msg=f"stdout={proc.stdout}\nstderr={proc.stderr}",
            )
            report = json.loads((out_dir / "gate_intervention_frozen_mean.json")
                               .read_text(encoding="utf-8"))
            self.assertEqual(report["split"], "val")
            self.assertEqual(len(report["learned"]["graph_weight_mean"]), 4)
            self.assertEqual(len(report["learned"]["graph_weight_std"]), 4)
            self.assertIn("paired_delta_rmse", report)


if __name__ == "__main__":
    unittest.main()
