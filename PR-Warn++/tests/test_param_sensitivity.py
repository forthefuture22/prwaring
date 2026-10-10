"""缺口 4（C3-7）：one-factor-at-a-time 参数敏感性 sweep 契约与聚合报告单测。

覆盖：
  ① 中心点 = 默认 config（不传覆盖时各参数 = config 现值，gate 回退 hidden_dim）；
  ② 7 组参数各自被扰动，且每次仅一个参数变化（overrides 集合互斥）；
  ③ 聚合报告表字段齐全（max_degradation / at_boundary）；
  ④ model.gate_hidden_dim 覆盖出现在 run 契约中（backbone 接线由 gap3 集成，
     消费侧测试标记为「集成后验证」）。
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from prwarn.cli.aggregate_sensitivity import (
    DEFAULT_CONFIG,
    EVAL_SPLIT,
    SENSITIVITY_GROUPS,
    aggregate_report,
    build_contracts,
    build_ofat_plan,
    load_config,
    main,
    resolve_center_value,
)

CONFIG = load_config(DEFAULT_CONFIG)


class CenterPointTests(unittest.TestCase):
    """① 中心点必须等于默认配置现值。"""

    def test_center_values_match_default_config(self):
        expected = {
            "data.history_steps": 24,
            "graphs.correlation_k": 12,
            "graphs.direction_distance_scale": 1000.0,
            "graphs.direction_sigma_degrees": 30.0,
            "model.temporal_kernel": 3,
            "physics.power_curve_bins": 50,
        }
        plan = {e["key"]: e["center"] for e in build_ofat_plan(CONFIG)}
        for key, value in expected.items():
            self.assertEqual(float(plan[key]), float(value), key)
        # gate_hidden_dim 当前 config 未声明，应回退 = model.hidden_dim = 64
        self.assertEqual(plan["model.gate_hidden_dim"], 64.0)

    def test_every_center_is_inside_levels(self):
        for group in SENSITIVITY_GROUPS:
            center = resolve_center_value(group, CONFIG)
            self.assertIn(center, [float(v) for v in group.levels], group.key)


class OfatMutualExclusionTests(unittest.TestCase):
    """② 每次只扰动一个参数；各参数组的覆盖键互斥。"""

    @classmethod
    def setUpClass(cls):
        cls.contracts = build_contracts(CONFIG, seeds=(2025, 2026))

    def test_seven_groups_and_eval_split_val(self):
        self.assertEqual({j["sensitivity_group"] for j in self.contracts},
                         {g.report_group for g in SENSITIVITY_GROUPS})
        self.assertTrue(all(j["eval_split"] == EVAL_SPLIT for j in self.contracts))

    def test_center_runs_have_empty_overrides(self):
        centers = [j for j in self.contracts if j["is_center"]]
        self.assertTrue(centers)
        self.assertTrue(all(j["overrides"] == {} for j in centers))

    def test_perturbed_runs_change_exactly_one_key(self):
        for job in self.contracts:
            if job["is_center"]:
                continue
            self.assertEqual(len(job["overrides"]), 1, job)
            self.assertIn(job["key"], job["overrides"], job)

    def test_override_keys_are_mutually_exclusive_across_groups(self):
        keys_by_group: dict[str, set[str]] = {}
        for job in self.contracts:
            keys_by_group.setdefault(job["sensitivity_group"], set()).update(
                job["overrides"].keys()
            )
        # 每组只覆盖自己那一个键；组间不串键
        for group in SENSITIVITY_GROUPS:
            self.assertEqual(keys_by_group[group.report_group], {group.key}, group)
        # 七组覆盖键两两不相交
        all_keys = [frozenset(v) for v in keys_by_group.values()]
        for i in range(len(all_keys)):
            for j in range(i + 1, len(all_keys)):
                self.assertTrue(all_keys[i].isdisjoint(all_keys[j]))


class ReportFieldsTests(unittest.TestCase):
    """③ 聚合报告表含 max_degradation / at_boundary 字段。"""

    def _write_run(self, runs_dir: Path, group: str, level: float,
                   rmse: float, crps: float, seed: int) -> None:
        payload = {
            "sensitivity_group": group,
            "level": level,
            "seed": seed,
            "split": EVAL_SPLIT,
            "metrics": {"rmse": rmse, "crps": crps},
        }
        (runs_dir / f"{group}_{level}_s{seed}.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )

    def test_report_has_required_fields_and_boundary_logic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = root / "runs"
            runs.mkdir()
            # history: center=24 rmse=0.10；最差在 12（退化 0.03）；最优=中心点
            for seed in (1, 2):
                self._write_run(runs, "history", 12, 0.13, 0.05, seed)
                self._write_run(runs, "history", 18, 0.11, 0.045, seed)
                self._write_run(runs, "history", 24, 0.10, 0.04, seed)
                self._write_run(runs, "history", 36, 0.105, 0.042, seed)
                self._write_run(runs, "history", 48, 0.12, 0.048, seed)
            # bandwidth: center=30 rmse=0.11；单调降到 60（边界）最优 → at_boundary
            for seed in (1, 2):
                self._write_run(runs, "bandwidth", 10.0, 0.15, 0.06, seed)
                self._write_run(runs, "bandwidth", 20.0, 0.13, 0.055, seed)
                self._write_run(runs, "bandwidth", 30.0, 0.11, 0.05, seed)
                self._write_run(runs, "bandwidth", 45.0, 0.10, 0.048, seed)
                self._write_run(runs, "bandwidth", 60.0, 0.09, 0.045, seed)

            report = aggregate_report(CONFIG, runs)
            summary = {row["group"]: row for row in report["summary"]}

            # 字段齐全
            for row in report["summary"]:
                self.assertIn("max_degradation", row)
                self.assertIn("at_boundary", row)
                self.assertIn("rmse", row["max_degradation"])
                self.assertIn("crps", row["max_degradation"])

            history = summary["history"]
            self.assertAlmostEqual(history["max_degradation_rmse"], 0.03, places=6)
            self.assertEqual(history["best_level"], 24)  # 最优=中心点
            self.assertFalse(history["at_boundary"])

            bandwidth = summary["bandwidth"]
            self.assertEqual(bandwidth["best_level"], 60.0)  # 落在右端点
            self.assertTrue(bandwidth["at_boundary"])

            # 明细行：5+5+4+5+4+5+5 = 33 水平行（每组含中心点）
            self.assertEqual(len(report["detail"]), 33)

    def test_cli_end_to_end_with_runs_dir(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = root / "runs"
            runs.mkdir()
            self._write_run(runs, "history", 24, 0.10, 0.04, 1)
            self._write_run(runs, "history", 12, 0.13, 0.05, 1)
            output = root / "sensitivity.json"
            argv = [
                "aggregate_sensitivity",
                "--config", str(DEFAULT_CONFIG),
                "--runs-dir", str(runs),
                "--output", str(output),
            ]
            with patch.object(sys, "argv", argv):
                main()
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["method"], "one-factor-at-a-time (OFAT)")
            self.assertIn("交互", payload["ofat_disclaimer"])
            self.assertTrue(payload["contract_count"] > 0)
            hist = [s for s in payload["report"]["summary"]
                    if s["group"] == "history"][0]
            self.assertAlmostEqual(hist["max_degradation_rmse"], 0.03, places=6)


class GateWidthContractTests(unittest.TestCase):
    """④ model.gate_hidden_dim 覆盖必须出现在 run 契约中。"""

    def test_gate_hidden_dim_override_emitted(self):
        contracts = build_contracts(CONFIG, seeds=(2025,))
        gate = [j for j in contracts
                if j["sensitivity_group"] == "gate_width"]
        self.assertTrue(gate)
        perturbed = [j for j in gate if not j["is_center"]]
        self.assertTrue(perturbed)
        for job in perturbed:
            self.assertIn("model.gate_hidden_dim", job["overrides"])
        # 水平序列：32/48/64/96/128
        levels = sorted({j["level"] for j in gate})
        self.assertEqual(levels, [32, 48, 64, 96, 128])

    def test_backbone_consumes_gate_hidden_dim_after_integration(self):
        """集成后验证：backbone 已实现 gate_hidden_dim 形参（gap3 落地）。

        gate = Sequential(Linear(hidden_dim, gate_inner), GELU,
                          Linear(gate_inner, n_static_graphs+2))。
        传 gate_hidden_dim=32 时 gate 中间层宽度=32，且主干 forecast_head 仍为
        hidden_dim=64（宽度解耦）；不传（None）时回退 = hidden_dim。
        """
        import torch  # noqa: PLC0415  延迟导入，避免未装 torch 时拖垮整文件
        from prwarn.models.backbone import (  # noqa: PLC0415
            DynamicMultiGraphResidualForecaster,
        )

        # 显式传 gate_hidden_dim=32：gate 中间层解耦为 32
        narrow = DynamicMultiGraphResidualForecaster(
            n_nodes=10, n_features=12, forecast_steps=6, hidden_dim=64,
            gate_hidden_dim=32,
        )
        self.assertEqual(narrow.gate[0].out_features, 32)
        self.assertEqual(narrow.gate[2].in_features, 32)
        # 主干（forecast_head）宽度不受 gate 扰动影响，仍为 64
        self.assertEqual(narrow.forecast_head[0].out_features, 64)

        # 不传（None）：回退 hidden_dim=64，与原始耦合实现一致
        coupled = DynamicMultiGraphResidualForecaster(
            n_nodes=10, n_features=12, forecast_steps=6, hidden_dim=64,
        )
        self.assertEqual(coupled.gate[0].out_features, 64)


if __name__ == "__main__":
    unittest.main()
