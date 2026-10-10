# 缺口 3 实施报告：门控干预实验（learned / uniform / parameter-matched + frozen_mean / shuffled 后验）

- 任务：PR-Warn++ 缺口 3（C2-6 / C2-1）门控干预对照
- 实施日期：2026-10-10
- 范围：仅修改所有权清单内文件；未做任何真实训练（GPU 训练与真实 checkpoint 后验运行属服务器工作）。
- 环境：Windows / Python 3.13 / torch 2.14.1+cpu；`prwarn` 已 `pip install -e`。

---

## 1. 改动清单（绝对路径 + 行号 + 说明）

### 1.1 `src/prwarn/models/backbone.py`（核心实现）
绝对路径：`C:\Users\weface\Desktop\prwaring-main\PR-Warn++\src\prwarn\models\backbone.py`

| 位置 | 说明 |
|---|---|
| L80–110 | 类 docstring 增补 `gate_mode` / `gate_hidden_dim` 语义说明；明确 frozen_mean/shuffled 为后验干预、不进 backbone。 |
| L115 | 新增类常量 `GATE_MODES = ("learned", "uniform", "parameter_matched")`。 |
| L135–136 | `__init__` 新增形参 `gate_mode: str = "learned"`、`gate_hidden_dim: int \| None = None`。 |
| L141–147 | 校验 `gate_mode` 合法性；`gate_hidden_dim` 为正整数校验；保存 `self.gate_mode`。 |
| L188–192 | gate MLP 首隐层宽度改为 `gate_inner`（`gate_hidden_dim`，None 时回退 `hidden_dim`）；第二 Linear 输出维度不变（`n_static_graphs + 2`）。 |
| L194–197 | `parameter_matched` 模式下冻结 gate 全部参数（`requires_grad_(False)`），保持与 learned 相同参数量但不可训。 |
| L229–236 | `_mix_graphs` 权重分支：`learned` 走原 `softmax(logits)` 逐位路径；`uniform`/`parameter_matched` 在启用图上广播等权 `1/#enabled`（未启用图为 0，仍受 `enabled_graph_mask` 屏蔽语义约束）。 |
| L240–244 | 后验干预钩子：`getattr(self, "_posthoc_weights", None)`，仅由后验 CLI 设置；默认 None 时 learned 路径完全不变。 |

**回归保证**：`gate_mode="learned"` 且 `gate_hidden_dim=None` 时，`_mix_graphs` 执行与改动前完全相同的三行（gate→masked_fill→softmax），钩子默认缺席。单测 `test_learned_default_is_bitwise_regression` 用同一种子构造默认模型与显式 `gate_mode="learned", gate_hidden_dim=None` 模型，断言 `y_det` 与 `graph_weights` 逐位 `torch.equal`。

### 1.2 `src/prwarn/cli/train_deterministic.py`（接线，最小改动）
绝对路径：`C:\Users\weface\Desktop\prwaring-main\PR-Warn++\src\prwarn\cli\train_deterministic.py`

| 位置 | 说明 |
|---|---|
| L279–284 | multigraph 分支内把 `config["graphs"].get("gate_mode","learned")` 与 `config["model"].get("gate_hidden_dim", None)` 写入 `model_config`；键缺失时回退默认，不影响既有行为。这两个键随 `best.pt` 的 `model_config` 一起保存，使 checkpoint 可被后验 CLI 原样重建。非 multigraph 分支（TemporalDeterministicBaseline）不受影响。 |

### 1.3 `src/prwarn/cli/evaluate_gate_intervention.py`（新增后验 CLI）
绝对路径：`C:\Users\weface\Desktop\prwaring-main\PR-Warn++\src\prwarn\cli\evaluate_gate_intervention.py`

- 入参：`--checkpoint`（learned `best.pt`）、`--data-dir`、`--config`、`--output-dir`、`--mode {frozen_mean,shuffled}`、`--split val`（仅允许 val）、`--batch-size`、`--shuffle-seed 2025`、`--device`。
- 流程：加载 checkpoint → 用 `model_config` 重建 `DynamicMultiGraphResidualForecaster` 并 load_state → 仅读 `val.npz` 前向一遍收集 learned 的 `graph_weights` 与 RMSE/MAE → 构造干预权重矩阵（frozen_mean = 验证集逐图均值广播；shuffled = 固定种子对样本轴置换）→ 第二遍经 `model._posthoc_weights` 钩子重评。
- 产物：`gate_intervention_{mode}.json`，含 `learned` 与 `intervention` 两段的 `rmse/mae/crps(null)/graph_weight_mean/graph_weight_std`、`paired_delta_rmse/mae`、`gate_names`、`shuffle_seed`、拟合 split 标注。
- 诚实声明：确定性 gate-3 只出点预测，`crps=null`（CRPS 属 gate-4 概率层，不在本后验诊断范围）。

### 1.4 `src/prwarn/experiments/matrix.py`（新增 A12）
绝对路径：`C:\Users\weface\Desktop\prwaring-main\PR-Warn++\src\prwarn\experiments\matrix.py`

| 位置 | 说明 |
|---|---|
| L149–166 | 接 A11 后新增 `ExperimentSpec("A12", "ablation", ...)`：comparison 为「learned vs uniform vs parameter-matched (+frozen_mean/shuffled post-hoc)」；question 表述动态门控净贡献；`primary_metrics=("rmse","crps","graph_weights")`；variants = `learned({})` / `uniform({graphs.gate_mode:"uniform"})` / `parameter_matched({graphs.gate_mode:"parameter_matched"})`。frozen_mean/shuffled 在 question 中声明为后验干预、不重训。 |

### 1.5 `src/prwarn/experiments/resolve.py`（阶段与命令接线）
绝对路径：`C:\Users\weface\Desktop\prwaring-main\PR-Warn++\src\prwarn\experiments\resolve.py`

| 位置 | 说明 |
|---|---|
| L33 | `execution_stage` 把 `A12` 并入 `"deterministic"`（learned/uniform/parameter_matched 走 train_deterministic）。 |
| L171–184 | `post_command_templates`：仅当 logical_id==A12 且 variant==learned 时，追加两条后验命令（frozen_mean、shuffled），消费 `outputs/{experiment_id}/best.pt`，`--data-dir <PROCESSED_DATA_DIR>`，`--split val`。 |

### 1.6 `tests/test_gate_intervention.py`（新增单测）
绝对路径：`C:\Users\weface\Desktop\prwaring-main\PR-Warn++\tests\test_gate_intervention.py`（8 个用例）

1. `test_learned_default_is_bitwise_regression` — learned 回归逐位一致。
2. `test_uniform_weights_equal_and_normalised` — uniform 启用图严格等权且行和为 1。
3. `test_gate_hidden_dim_decouples_width` — gate_hidden_dim=32 时 `gate[0].out_features==32`，None 时等于 hidden_dim=8。
4. `test_parameter_matched_keeps_param_count_and_freezes_gate` — 参数量与 learned 相等、gate 参数冻结。
5. `test_invalid_gate_mode_rejected` — frozen_mean 等非法 mode 被拒。
6. `test_a12_contract_stable_and_planned` — A12 三变体展开、experiment_id 稳定、`evidence_status=planned_not_run`。
7. `test_a12_stage_and_post_commands` — A12 阶段=deterministic、post-command 含两模式。
8. `test_evaluate_gate_intervention_runs` — 随机初始化 checkpoint 端到端 CLI（临时目录伪造最小 processed 数据）退出码 0、产物含每图权重均值/标准差。

---

## 2. 验证结果

- `python -m compileall -q src tests`：通过。
- `python -m pytest tests\test_gate_intervention.py -v`：**8 passed**。
- 全量 `python -m pytest tests -q`：**74 passed, 1 skipped**（skipped 为 gap4 子代理预留用例 `test_param_sensitivity.py:198`，其注释明确等待本任务落地 `gate_hidden_dim`——现已具备，gap4 子代理可后续解除 skip）。无任何既有用例回归。

## 3. 验收自检清单（对照 gap3 规范 §5）

| 项 | 结果 | 证据 |
|---|---|---|
| config 出现 `graphs.gate_mode` 默认 learned，切 learned 时行为与改动前逐位一致 | ◐（代码侧就绪，配置键待编排者集成） | backbone L229 learned 路径原样；单测 1 逐位相等。yaml 集成后由编排者终核（见 §5）。 |
| uniform 模式启用图严格等权 | ✓ | 单测 2；内联输出 `[0.25,0.25,0.25,0.25]`。 |
| frozen_mean 常数仅在验证集求得、跨样本权重相同、产物可追溯 split | ✓（代码与产物字段） | CLI 仅 `--split val`；frozen_mean=验证集均值广播；JSON 含 `split:"val"`、`shuffle_seed`。真实数值待服务器运行。 |
| shuffled 固定种子置换、逐图边缘均值与 learned 一致、标注后验不重训 | ✓（代码） | CLI `torch.randperm(N, generator=seed)`；post_command 仅消费 learned run，无训练。 |
| parameter_matched 参数量与 learned 逐位相等、gate 参数不更新 | ✓ | 单测 4：`sum(p.numel())` 相等（1070==1070）、`requires_grad` 全 False。 |
| matrix 新增 A12、可展开 run contract、experiment_id 稳定、evidence_status=planned_not_run | ✓ | 单测 6/7；内联展开 `A12-learned-s2025-fcf8a208b3` 稳定。 |
| 后验 CLI 可加载 learned checkpoint 并在验证集输出对比表（RMSE/各图权重均值与标准差） | ✓（smoke 通过） | 单测 8 退出码 0、产物含 `graph_weight_mean/std`；CRPS 字段为 null 并注明原因。 |
| 结论措辞限定「动态门控相对对照的改善」、不声称尾流因果 | ✓（实现层无越界 claim） | CLI notes 与 A12 question 均限定为「门控机制净贡献」，未做因果/物理声明。 |
| 多 seed（≥10）报告 mean±SD | ◐（机制就绪，服务器执行） | matrix 经 seeds 展开；C0-2 的 ≥10 seed 由服务器编排时配置 seeds 参数完成。 |

图例：✓ 已本地验证；◐ 代码机制就绪、依赖服务器/编排者终核。

## 4. 需服务器运行清单（本地不跑真实训练）

1. 在 GPU 上为 A12 三个可训练变体（learned / uniform / parameter_matched）按 `seeds`（建议 ≥10）跑 `train_deterministic`，产出各 run `best.pt` 与 `deterministic_val.h5`。
2. 对 learned run 执行 `post_command_templates` 生成的两条后验命令：
   - `python -m prwarn.cli.evaluate_gate_intervention --config <cfg> --checkpoint outputs/<A12-learned-run>/best.pt --data-dir <PROCESSED> --split val --mode frozen_mean --output-dir outputs/<run>/gate_intervention`
   - 同上 `--mode shuffled`。
3. 收集五模式（learned/uniform/parameter_matched/frozen_mean/shuffled）的 val RMSE/MAE 与各图 `graph_weight_mean/std`，按多 seed 汇总 mean±SD；`paired_delta_rmse` 即相对 learned 的干预净效应。
4. 论文结果表成套报告四类对照，措辞限定为「动态门控相对静态/均匀/打乱/参数匹配对照的改善」，不做尾流/因果断言。

## 5. 待编排者集成的配置键（未改 `configs/sdwpf_v3_2.yaml`）

请在 `configs/sdwpf_v3_2.yaml` 中加入（代码已按此读取）：

```yaml
graphs:
  gate_mode: learned        # learned | uniform | parameter_matched
model:
  gate_hidden_dim: null     # null -> 回退 model.hidden_dim；OFAT 时置 32/48/96/128
```

说明：
- `graphs.gate_mode` 缺失时代码回退 `"learned"`；`model.gate_hidden_dim` 缺失/null 时回退 `hidden_dim`，因此**不加这两个键也不会破坏 A0–A11**（已由全量测试 74 passed 佐证）。
- frozen_mean/shuffled 不进 yaml，由后验 CLI 与 post_command 驱动，无需配置键。
- gap4 的 OFAT gate-width 组直接复用 `model.gate_hidden_dim`（backbone 消费已在本任务落地，gap4 子代理预留的 skip 用例可解除）。

## 6. 文件所有权合规自查

- 仅新增/修改：`backbone.py`、`train_deterministic.py`、`evaluate_gate_intervention.py`(新)、`matrix.py`、`resolve.py`、`test_gate_intervention.py`(新)。
- 未触碰 `configs/sdwpf_v3_2.yaml`、`stress.py`、`statistics.py`、`deterministic_baselines.py` 及其他任何文件；未做任何 git 操作；未删除/覆盖文件。
