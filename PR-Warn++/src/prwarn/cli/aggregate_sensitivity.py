"""One-factor-at-a-time (OFAT) parameter-sensitivity sweep contract + report.

对应约束 C3-7（IEEE TSTE 投稿 PR-Warn++ P0 硬缺口）：围绕默认配置中心点，
对 history / Top-K / distance scale / bandwidth / lag / bin width / gate width
共七组参数逐一生成「中心点 + N 个水平」的可执行 run 契约，并聚合多 seed 指标，
报告每个参数相对基线的最大绝对退化（max_degradation）与最优水平是否落在
搜索边界（at_boundary）。

方法学声明（产物中固化，论文敏感性节须照写）：
    OFAT 每次只扰动一个因子、其余全部保持默认配置中心点；它**不捕捉因子间
    交互作用**（Czitrom 1999, The American Statistician 53(2):126-131；
    Saltelli 2002, CPC 145(2):280-297 为全局敏感性分析对照方法学）。本文用 OFAT
    做稳健性筛查，而非全局敏感性排序。所有敏感性指标只在 validation split 上
    评估，绝不触碰测试集。

真实训练（7 组参数 × 多 seed 的 GPU 指标采集）在服务器完成；本模块只产出
「sweep 契约」与「聚合报告」两类纯 CPU 产物，不触发任何训练。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from prwarn.experiments.matrix import make_experiment_id

# ---------------------------------------------------------------------------
# 仓库与默认配置定位
# ---------------------------------------------------------------------------
_THIS = Path(__file__).resolve()
# src/prwarn/cli/aggregate_sensitivity.py -> parents[3] = PR-Warn++ 仓库根
REPO_ROOT = _THIS.parents[3]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "sdwpf_v3_2.yaml"

# 敏感性实验只在验证集评估（C3-7 / C3-5：超参只在验证集选择）
EVAL_SPLIT = "val"

OFAT_DISCLAIMER = (
    "One-factor-at-a-time (OFAT): 每次只扰动一个参数，其余全部保持默认配置中心点。"
    "OFAT 不捕捉参数间交互作用（Czitrom 1999; Saltelli 2002）——本文用于稳健性筛查，"
    "而非全局敏感性排序；所有指标仅在 validation split 评估。"
)

# ---------------------------------------------------------------------------
# 七组参数水平表（唯一契约来源，逐条对照 gap4_param_sensitivity.md §4 表）
#   levels 已包含加粗的中心点；center 运行 = overrides 为空（不扰动）。
# ---------------------------------------------------------------------------
PRIMARY_METRICS = ("rmse", "crps")


@dataclass(frozen=True)
class SensitivityGroup:
    logical_id: str          # S1..S7
    report_group: str        # 报告分组名
    key: str                 # 被扰动的点号配置键（dotted）
    levels: tuple[float, ...]  # 含中心点的水平序列
    question: str
    center_fallback: str | None = field(default=None)
    # center_fallback: 若主键在 config 中缺失/为 null，回退读取的点号键
    #   （仅 model.gate_hidden_dim：null/缺失时回退 = model.hidden_dim）


SENSITIVITY_GROUPS: tuple[SensitivityGroup, ...] = (
    SensitivityGroup(
        "S1", "history", "data.history_steps", (12, 18, 24, 36, 48),
        "回看窗口（10min 步）变化时，验证集 RMSE/CRPS 是否稳健？",
    ),
    SensitivityGroup(
        "S2", "top_k", "graphs.correlation_k", (4, 8, 12, 16, 24),
        "相关图 Top-K 稀疏度变化时，验证集指标是否稳健？",
    ),
    SensitivityGroup(
        "S3", "distance_scale", "graphs.direction_distance_scale",
        (500.0, 1000.0, 2000.0, 4000.0),
        "方向图距离衰减尺度（米）变化时，验证集指标是否稳健？",
    ),
    SensitivityGroup(
        "S4", "bandwidth", "graphs.direction_sigma_degrees",
        (10.0, 20.0, 30.0, 45.0, 60.0),
        "方向图高斯角带宽（度）变化时，验证集指标是否稳健？",
    ),
    SensitivityGroup(
        "S5", "lag", "model.temporal_kernel", (1, 3, 5, 7),
        "时间卷积核（局部感受野/lag，须为奇数）变化时，验证集指标是否稳健？",
    ),
    SensitivityGroup(
        "S6", "bin_width", "physics.power_curve_bins", (20, 35, 50, 75, 100),
        "功率曲线分箱数（bin 宽度 ∝ 风速量程/n_bins）变化时是否稳健？",
    ),
    SensitivityGroup(
        "S7", "gate_width", "model.gate_hidden_dim", (32, 48, 64, 96, 128),
        "gate MLP 隐藏宽度（独立于全局 hidden_dim）变化时是否稳健？",
        center_fallback="model.hidden_dim",
    ),
)


# ---------------------------------------------------------------------------
# 配置读取与中心点解析
# ---------------------------------------------------------------------------
def _dotted_get(mapping: Mapping[str, Any], dotted: str) -> Any:
    node: Any = mapping
    for part in dotted.split("."):
        if not isinstance(node, Mapping) or part not in node:
            raise KeyError(f"config 中找不到键 {dotted!r}（缺少段 {part!r}）")
        node = node[part]
    return node


def load_config(config_path: Path) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_center_value(group: SensitivityGroup, config: Mapping[str, Any]) -> float:
    """中心点 = 默认 config 现值；gate_hidden_dim 缺失/null 时回退 hidden_dim。"""
    try:
        value = _dotted_get(config, group.key)
    except KeyError:
        value = None
    if value is None and group.center_fallback is not None:
        value = _dotted_get(config, group.center_fallback)
    if value is None:
        raise KeyError(
            f"无法为 {group.key} 解析中心点：键缺失且无可用回退键"
        )
    return float(value)


def resolve_seeds(config: Mapping[str, Any]) -> tuple[int, ...]:
    seeds = _dotted_get(config, "experiment.seeds")
    return tuple(int(s) for s in seeds)


# ---------------------------------------------------------------------------
# OFAT run 契约生成（每次只覆盖一个键；中心点覆盖为空）
# ---------------------------------------------------------------------------
def _coerce_level(group: SensitivityGroup, raw: float) -> float:
    """把配置中的中心值对齐到本组水平表的同值水平（int/float 容忍）。"""
    for level in group.levels:
        if float(level) == float(raw):
            return level
    raise ValueError(
        f"{group.key} 默认中心点 {raw!r} 不在水平表 {list(group.levels)} 内；"
        "OFAT 要求中心点必须落在扰动水平序列中"
    )


def build_ofat_plan(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """从基准 config 出发，生成 7 组参数的「中心点 + N 水平」计划。

    返回每组：{group, key, center, levels, runs:[{level, overrides, is_center}]}。
    每次只覆盖一个键；中心点 run 的 overrides 为空 dict（= 默认配置本身）。
    """
    plan: list[dict[str, Any]] = []
    for group in SENSITIVITY_GROUPS:
        center = resolve_center_value(group, config)
        center_level = _coerce_level(group, center)
        runs: list[dict[str, Any]] = []
        for level in group.levels:
            is_center = float(level) == float(center_level)
            overrides: dict[str, Any] = {} if is_center else {group.key: level}
            runs.append(
                {"level": level, "overrides": overrides, "is_center": is_center}
            )
        plan.append(
            {
                "logical_id": group.logical_id,
                "group": group.report_group,
                "key": group.key,
                "question": group.question,
                "center": center_level,
                "levels": list(group.levels),
                "runs": runs,
            }
        )
    return plan


def build_contracts(
    config: Mapping[str, Any],
    *,
    seeds: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    """展开为 seed 级可执行 run 契约（对齐 outputs/plans/v3_2_matrix.json 结构）。"""
    plan = build_ofat_plan(config)
    seed_list = tuple(seeds) if seeds is not None else resolve_seeds(config)
    contracts: list[dict[str, Any]] = []
    for entry in plan:
        for run in entry["runs"]:
            level = run["level"]
            overrides = dict(run["overrides"])
            variant = f"{entry['group']}_{level}"
            for seed in seed_list:
                job = {
                    "logical_id": entry["logical_id"],
                    "group": "sensitivity",
                    "comparison": f"OFAT sweep: {entry['key']}",
                    "question": entry["question"],
                    "primary_metrics": list(PRIMARY_METRICS),
                    "server_required": True,
                    "eval_split": EVAL_SPLIT,
                    "sensitivity_group": entry["group"],
                    "key": entry["key"],
                    "level": level,
                    "is_center": run["is_center"],
                    "variant": variant,
                    "seed": int(seed),
                    "overrides": overrides,
                    "experiment_id": make_experiment_id(
                        entry["logical_id"], variant, int(seed), overrides
                    ),
                    "evidence_status": "planned_not_run",
                }
                contracts.append(job)
    return contracts


# ---------------------------------------------------------------------------
# 聚合报告：输入各 run 的指标 JSON，输出 明细表 + 每参数汇总
# ---------------------------------------------------------------------------
def _extract_metric(result: Mapping[str, Any], metric: str) -> float | None:
    metrics = result.get("metrics")
    if isinstance(metrics, Mapping) and metric in metrics:
        value = metrics[metric]
    elif metric in result:
        value = result[metric]
    else:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value


def _mean_std(values: list[float]) -> dict[str, float | int] | None:
    finite = [v for v in values if v == v]  # NaN 过滤
    if not finite:
        return None
    mean = sum(finite) / len(finite)
    if len(finite) > 1:
        var = sum((v - mean) ** 2 for v in finite) / (len(finite) - 1)
        sd = var ** 0.5
    else:
        sd = 0.0
    return {"mean": mean, "sd": sd, "n_seeds": len(finite)}


def aggregate_report(
    config: Mapping[str, Any],
    runs_dir: Path | None,
) -> dict[str, Any]:
    """读取 runs_dir 下各 run 的指标 JSON，产出 OFAT 报告。

    每个结果 JSON 应含 sensitivity_group / level / seed，以及 metrics.rmse 与
    metrics.crps（或顶层 rmse/crps）。runs_dir 为空或缺失时，仍输出完整计划
    骨架，指标字段为 null（供服务器运行前先行校验契约结构）。
    """
    plan = build_ofat_plan(config)

    # (group, level_value) -> {metric: [per-seed values]}
    collected: dict[tuple[str, float], dict[str, list[float]]] = {}
    split_seen: set[str] = set()
    if runs_dir is not None and Path(runs_dir).exists():
        for path in sorted(Path(runs_dir).glob("*.json")):
            try:
                result = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            group = result.get("sensitivity_group")
            level = result.get("level")
            if group is None or level is None:
                continue
            split_seen.add(str(result.get("split", "unknown")))
            bucket = collected.setdefault(
                (str(group), float(level)), {m: [] for m in PRIMARY_METRICS}
            )
            for metric in PRIMARY_METRICS:
                value = _extract_metric(result, metric)
                if value is not None:
                    bucket[metric].append(value)

    detail_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for entry in plan:
        group = entry["group"]
        center = entry["center"]
        center_stats = {
            m: _mean_std(collected.get((group, float(center)), {}).get(m, []))
            for m in PRIMARY_METRICS
        }
        per_level_degradation: dict[str, float] = {m: 0.0 for m in PRIMARY_METRICS}
        best_level: float | None = None
        best_mean: float | None = None
        for level in entry["levels"]:
            stats = {
                m: _mean_std(collected.get((group, float(level)), {}).get(m, []))
                for m in PRIMARY_METRICS
            }
            row: dict[str, Any] = {
                "group": group,
                "key": entry["key"],
                "level": level,
                "is_center": float(level) == float(center),
            }
            degradation: dict[str, float | None] = {}
            for m in PRIMARY_METRICS:
                cell = stats[m]
                row[m] = cell
                if cell is not None and center_stats[m] is not None:
                    degradation[m] = abs(cell["mean"] - center_stats[m]["mean"])
                    per_level_degradation[m] = max(
                        per_level_degradation[m], degradation[m]
                    )
                else:
                    degradation[m] = None
            row["abs_degradation"] = degradation
            detail_rows.append(row)

            # 最优水平 = 最小 mean RMSE（主指标）；只在有数据时更新
            rmse_cell = stats["rmse"]
            if rmse_cell is not None:
                if best_mean is None or rmse_cell["mean"] < best_mean:
                    best_mean = rmse_cell["mean"]
                    best_level = level

        levels = entry["levels"]
        at_boundary: bool | None
        if best_level is None:
            at_boundary = None
        else:
            at_boundary = best_level in (levels[0], levels[-1])

        summary_rows.append(
            {
                "group": group,
                "key": entry["key"],
                "center": center,
                "levels": list(levels),
                "best_level": best_level,
                "best_metric": "rmse",
                "max_degradation": dict(per_level_degradation),
                "max_degradation_rmse": per_level_degradation["rmse"],
                "max_degradation_crps": per_level_degradation["crps"],
                "at_boundary": at_boundary,
            }
        )

    return {
        "eval_split": EVAL_SPLIT,
        "split_seen": sorted(split_seen),
        "center_values": {
            e["key"]: e["center"] for e in plan
        },
        "detail": detail_rows,
        "summary": summary_rows,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG,
        help="基准（中心点）配置 YAML，默认 configs/sdwpf_v3_2.yaml",
    )
    parser.add_argument(
        "--runs-dir", type=Path, default=None,
        help="S1–S7 各 run 指标 JSON 所在目录；缺省则只产出契约骨架",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=None,
        help="覆盖 experiment.seeds（默认读 config 内 seeds）",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_config(args.config)
    contracts = build_contracts(config, seeds=args.seeds)
    report = aggregate_report(config, args.runs_dir)

    payload = {
        "schema_version": 1,
        "method": "one-factor-at-a-time (OFAT)",
        "ofat_disclaimer": OFAT_DISCLAIMER,
        "eval_split": EVAL_SPLIT,
        "center_config": str(args.config),
        "group_count": len(SENSITIVITY_GROUPS),
        "contract_count": len(contracts),
        "contracts": contracts,
        "report": report,
        "evidence_status": "planned_not_run",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "contract_count": len(contracts),
                "group_count": len(SENSITIVITY_GROUPS),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
