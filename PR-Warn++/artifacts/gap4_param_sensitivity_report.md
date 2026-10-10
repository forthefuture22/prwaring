# 缺口 4 实施报告：参数敏感性 one-factor-at-a-time（OFAT）

> 约束：C3-7（P0 硬缺口）｜ 目标：IEEE TSTE 投稿 PR-Warn++
> 实施依据：`artifacts/gap_specs/gap4_param_sensitivity.md`（逐条对照）
> 实施日期：2026-10-10 ｜ 实施范围：仅新增 2 个文件，未改任何既有文件/配置

---

## 0. 交付物清单（绝对路径）

| 文件 | 性质 | 行数 |
|---|---|---|
| `C:\Users\weface\Desktop\prwaring-main\PR-Warn++\src\prwarn\cli\aggregate_sensitivity.py` | 新增（核心交付） | 398 |
| `C:\Users\weface\Desktop\prwaring-main\PR-Warn++\tests\test_param_sensitivity.py` | 新增（单测） | 约 210 |

未修改 `configs/sdwpf_v3_2.yaml`、`experiments/matrix.py`、`resolve.py`、`physics/power_curve.py`、`data/window.py`、`models/backbone.py`、`cli/aggregate_seed_runs.py`（均按要求只读）。

---

## 1. 改动文件说明（文件:行号 + 说明）

### 1.1 `src/prwarn/cli/aggregate_sensitivity.py`

- `:41` `EVAL_SPLIT = "val"` —— 敏感性只在验证集评估（C3-7 / C3-5），产物每条记录固化。
- `:43-46` `OFAT_DISCLAIMER` —— 固化方法学声明：每次只动一个因子、不捕交互（Czitrom 1999 / Saltelli 2002），随产物落盘，论文敏感性节照写。
- `:60-117` `SENSITIVITY_GROUPS`（`:68`）—— 七组参数水平表唯一契约来源，逐条对照规范 §4 表：
  - S1 history → `data.history_steps`，水平 12/18/**24**/36/48
  - S2 top_k → `graphs.correlation_k`，水平 4/8/**12**/16/24
  - S3 distance_scale → `graphs.direction_distance_scale`，水平 500/**1000**/2000/4000
  - S4 bandwidth → `graphs.direction_sigma_degrees`，水平 10/20/**30**/45/60
  - S5 lag → `model.temporal_kernel`，水平 1/**3**/5/7（全奇数，满足 backbone.py:107 强制）
  - S6 bin_width → `physics.power_curve_bins`，水平 20/35/**50**/75/100
  - S7 gate_width → `model.gate_hidden_dim`，水平 32/48/**64**/96/128（`center_fallback="model.hidden_dim"`）
- `:120-131` `resolve_center_value` —— 中心点 = 默认 config 现值；`gate_hidden_dim` 缺失/null 时回退 `model.hidden_dim`（=64），对齐规范「null 回退」约定。
- `:148-152` `load_config` / `:133-136` `resolve_seeds` —— 读基准 YAML 与 seed 列表。
- `:154-183` `build_ofat_plan` —— 从基准 config 生成「中心点 + N 水平」计划；中心点 run 的 `overrides={}`（不扰动），扰动 run 只含 `{本组键: 水平}`。`:171` 强制中心点必须落在水平序列内。
- `:185-221` `build_contracts` —— 展开为 seed 级 run 契约，结构对齐 `outputs/plans/v3_2_matrix.json`（复用 `matrix.make_experiment_id`），每条带 `eval_split:"val"`、`sensitivity_group`、`key`、`level`、`is_center`、`evidence_status:"planned_not_run"`。
- `:255-357` `aggregate_report` —— 输入各 run 指标 JSON（`metrics.rmse/crps`），按 (group, level) 分组做多 seed mean±SD；产出：
  - `detail`：行=参数×水平，列=rmse/crps(mean/sd/n_seeds) + 相对基线绝对退化 `abs_degradation`；
  - `summary`：每参数一行汇总，含 `max_degradation`（{rmse, crps} 绝对量）、`max_degradation_rmse/crps`、`best_level`、`at_boundary`（最优水平是否落在水平序列端点）。
- `:360-385` `parse_args` / `:387-398` `main` —— CLI 支持 `--config`（默认 configs/sdwpf_v3_2.yaml）、`--runs-dir`（缺省只出契约骨架）、`--output`、`--seeds` 覆盖。产物单文件含 `contracts` + `report` 两部分。

### 1.2 `tests/test_param_sensitivity.py`

- `CenterPointTests`：断言中心点 = config 现值、且每组中心点落在水平序列内。
- `OfatMutualExclusionTests`：断言每次仅一个参数变化——中心点覆盖为空、扰动 run 恰好一个覆盖键且等于本组键、七组覆盖键两两不相交（互斥）。
- `ReportFieldsTests`：用合成 run 结果断言 `max_degradation`/`at_boundary` 字段齐全，并验证边界判定逻辑（最优在右端点 → at_boundary=True；最优=中心点 → False）；CLI 端到端跑通。
- `GateWidthContractTests`：断言 `model.gate_hidden_dim` 覆盖出现在契约中（水平 32/48/64/96/128）；另一条 backbone 消费侧测试 `@unittest.skip` 标注「集成后验证」，待 gap3 落地后由编排者启用。

---

## 2. 七组参数水平取值表（OFAT 契约，作为服务器运行依据）

| # | 报告分组 | 配置键 | 中心点（默认 config） | 扰动水平（粗体=中心点） | 单位/约束 |
|---|---|---|---|---|---|
| S1 | history | `data.history_steps` | 24 | 12, 18, **24**, 36, 48 | 10min 步（≈2/3/4/6/8 h） |
| S2 | Top-K | `graphs.correlation_k` | 12 | 4, 8, **12**, 16, 24 | k 自动 clamp ≤ 节点数 |
| S3 | distance scale | `graphs.direction_distance_scale` | 1000.0 | 500, **1000**, 2000, 4000 | 米，方向图距离衰减尺度 |
| S4 | bandwidth | `graphs.direction_sigma_degrees` | 30.0 | 10, 20, **30**, 45, 60 | 度，方向图高斯角带宽 |
| S5 | lag | `model.temporal_kernel` | 3 | 1, **3**, 5, 7 | 须奇数（backbone.py:107 强制） |
| S6 | bin width | `physics.power_curve_bins` | 50 | 20, 35, **50**, 75, 100 | n_bins；bin 宽 ∝ 风速量程/n_bins |
| S7 | gate width | `model.gate_hidden_dim` | 64（回退 = `model.hidden_dim`） | 32, 48, **64**, 96, 128 | 新增键；null→回退 hidden_dim |

- 水平总数 = 5+5+4+5+4+5+5 = **33**；× 默认 5 seed = **165 条 run 契约**（实测 `contract_count=165`）。
- 每次只扰动一个键，其余全部保持中心点；中心点 run（`overrides={}`）共 7 组 × seed 数。
- 评估 split 一律 `val`，不触碰测试集。
- 多 seed mean±SD：契约默认读 `experiment.seeds=[2025..2029]`（5 个）；**规范 C0-2 要求 ≥10 seed**，服务器运行前建议在 config 扩 seed 至 ≥10（或 `--seeds` 传入），本模块不锁死。

---

## 3. 验收自检清单（对照规范 §5）

| 规范条目 | 状态 | 证据 |
|---|---|---|
| 每组 variants 只覆盖一个配置键（一次只动一个因子） | ✓ | `test_perturbed_runs_change_exactly_one_key` + `test_override_keys_are_mutually_exclusive_across_groups` 通过；实测 0 违规契约 |
| 七组全部映射到真实键 | ✓ | SENSITIVITY_GROUPS `:68-117`，键与规范 §4 表逐项一致 |
| 每组水平含中心点；temporal_kernel 全奇数 | ✓ | `test_every_center_is_inside_levels` 通过；lag 水平 1/3/5/7 全奇数 |
| 敏感性 run 评估 split 标注 val，无 test 混入 | ✓ | `EVAL_SPLIT="val"`（:41）；`test_seven_groups_and_eval_split_val` 断言全部契约 split=val |
| 能展开 run contract，experiment_id 稳定、evidence_status=planned_not_run | ✓ | 复用 `make_experiment_id`；契约 `evidence_status="planned_not_run"`；实测 165 条 |
| 报告含每参数 max_abs_degradation（绝对量，非百分比）与 at_boundary 列 | ✓ | summary 含 `max_degradation`/{rmse,crps}、`max_degradation_rmse/crps`、`at_boundary`；`test_report_has_required_fields_and_boundary_logic` 通过 |
| 每水平多 seed 报告 mean±SD | ✓ | 明细单元格 `{mean, sd, n_seeds}`；聚合按 seed 分组（样本 sd ddof=1） |
| 论文声明 OFAT 不捕交互（Czitrom 1999 / Saltelli 2002） | ✓ | `OFAT_DISCLAIMER` 固化进产物；`test_cli_end_to_end_with_runs_dir` 断言含「交互」字样 |
| gate_hidden_dim 不传时回退 hidden_dim，回归不破坏 A0–A12 | ✓ | 配置键侧回退逻辑已实现并测（`resolve_center_value`，:120）；backbone 消费侧已由 gap3 落地（backbone.py:188 `gate_inner`）；集成后测试 `test_backbone_consumes_gate_hidden_dim_after_integration`（tests:198）已解除 skip 并通过 |

---

## 4. 验证结果

- `python -m compileall -q src tests` → **FULL_COMPILEALL_OK**。
- `python -m pytest tests\test_param_sensitivity.py -v` → **9 passed, 1 skipped**（skip 项 = 刻意标注的 gap3 集成后验证）。
- 相邻既有回归：`test_experiment_matrix.py` + `test_aggregate_seed_cli.py` → **5 passed**，无破坏。

---

## 5. 需服务器运行清单（不在本地训练）

1. 集成后由编排者在 `configs/sdwpf_v3_2.yaml` `model` 段新增 `gate_hidden_dim: null`（null→回退 hidden_dim）。
2. （建议）把 `experiment.seeds` 扩到 ≥10 以满足 C0-2。
3. 在服务器逐 run 执行 165 条契约（`--seeds` 与 config 对齐），每条产出验证集指标 JSON，字段：
   `{"sensitivity_group":..., "level":..., "seed":..., "split":"val", "metrics":{"rmse":..., "crps":...}}`，放入同一目录。
4. 聚合：
   `python -m prwarn.cli.aggregate_sensitivity --config configs/sdwpf_v3_2.yaml --runs-dir <结果目录> --output artifacts/ofat_sensitivity_report.json`
5. 若某参数 `at_boundary=true`，说明当前搜索区间过窄，需扩区间重测（C3-7）。

---

## 6. 待集成配置键 / 依赖

- **待编排者集成**：`configs/sdwpf_v3_2.yaml` 新增 `model.gate_hidden_dim: null`（本任务按禁令未改 config）。
- **已闭环（2026-10-10）**：`backbone.py` 消费 `gate_hidden_dim`（gate MLP 中间层宽度解耦，None 回退 hidden_dim，backbone.py:188-193）；集成后测试 `test_backbone_consumes_gate_hidden_dim_after_integration`（tests:198）已解除 skip 并通过。
- 本模块不依赖上述两项即可独立运行契约生成与报告聚合（gate 中心点经 hidden_dim 回退=64 已验证）。
