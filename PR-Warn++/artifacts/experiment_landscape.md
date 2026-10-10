# PR-Warn++ 现有实验全景（面向 IEEE TSTE 投稿）

> 范围：纯代码阅读 + 表格整理，未改任何代码/配置、未跑实验。
> 依据文件：`src/prwarn/experiments/matrix.py`（行号均对应该文件）、`configs/sdwpf_v3_2.yaml`、`docs/投稿约束清单.md`（C0–C6 共 39 条）、`docs/约束符合性核查报告.md`，并以 `src/prwarn/eval/metrics.py`、`src/prwarn/cli/evaluate*.py`、`benchmark_inference.py` 核对指标实现。
> 所有矩阵作业在 `build_experiment_matrix` 中统一打 `evidence_status="planned_not_run"`（matrix.py:258），故运行状态整体为 **planned**；下表「代码有无」区分的是 spec 是否已登记、配套 CLI 是否已就绪。

---

## 1. 现有实验全景表（主表）

### 1.1 A 族 — 核心消融 / 协议（A0–A11，共 12 项）

| 实验 ID · 名称 | 目的 / 检验问题 | 卖点归类 | 关联约束 ID | 实现状态（代码｜指标｜运行） |
|---|---|---|---|---|
| **A0** Direct power vs P_pc+residual | 绕弱物理中心做残差预测是否必要？（direct_power ↔ physics_residual） | 物理先验 | C2-1、C1-1、C5-1 | 代码：matrix.py:28 spec；无专属 CLI ｜ 指标：rmse 族未列入，**extreme_rmse/extreme_mae/ramp_f1 未实现**（仅声明）｜ planned |
| **A1** 密度修正 off/on | 等效风速/密度修正是否改善物理中心？（density_correction False↔True） | 物理先验 | C5-1、C2-1、C1-1 | 代码：matrix.py:37 spec；物理模块 `physics.reference_density_kg_m3=1.225` 已在 config ｜ 指标：**power_curve_error/seasonal_rmse/extreme_rmse 均未实现**（仅声明）｜ planned |
| **A2** GEO→+CORR→+DIR→+ADP 图叠加 | 每类图是否带来独立增益？（geo / geo+corr / +dir / all） | 动态多图 | C2-1、C2-6、C2-4 | 代码：matrix.py:46 spec；`graphs.enabled` 逐项可控、`graph_weights` 由 backbone.py:244 输出并落盘 ｜ 指标：rmse✓、energy_score✓、graph_weights✓（作为张量记录，非标量指标）｜ planned |
| **A3** 原始相关 vs 差分相关 | difference correlation 是否值得保留？（correlation_mode raw↔difference） | 动态多图 | C2-1 | 代码：matrix.py:60 spec；config `correlation_mode: difference` ｜ 指标：rmse✓、missing_stress_rmse△（stress CLI 有 rmse_degradation，非同名）｜ planned |
| **A4** Direct CFM vs VAE-CFM | VAE 阶段带来的增益是否值回成本？（flow.variant） | 概率生成 | C3-6、C2-1 | 代码：matrix.py:69 spec；`generative_baselines.vae_cfm` 已配 ｜ 指标：energy_score✓、variogram_score✓、ramp_ks✓、latency✓（benchmark_inference.py:103）、peak_vram✓（:112）｜ planned |
| **A5** CFM vs 真·OT-CFM | 最优传输耦合是否提升质量或效率？（coupling independent↔optimal_transport）`ablation_optional` | 概率生成 | C3-6、C2-1 | 代码：matrix.py:78 spec（optional）｜ 指标：energy_score✓、ode_steps△（train_flow.py:169 仅作输出属性记录，非评分）、**training_time 未实现**（benchmark 只测推理）｜ planned |
| **A6** ODE 步数 4/8/16/32 | 采样速度–质量 Pareto 前沿在哪？（flow.steps sweep） | 概率生成 / 效率 | C3-6、C3-7 | 代码：matrix.py:87 spec；config `flow.steps: 8` ｜ 指标：energy_score✓、variogram_score✓、latency✓、throughput✓（benchmark_inference.py:109）｜ planned |
| **A7** 静态 conformal vs ACI | 时序自适应在分布漂移下是否有用？（calibration.primary split_per_horizon↔aci） | 校准 | C3-3、C2-5 | 代码：matrix.py:93 spec；`calibration.aci_gamma=0.01` 已配、`calibration/adaptive.py` 存在 ｜ 指标：**rolling_picp 未实现**（仅有静态 interval_metrics）、interval_width△（metrics.py:295 返回 mean_width/winkler，非同名）｜ planned |
| **A8** Context/fallback off/on | 经验 OOD fallback 是否改善尾部覆盖？（context_fallback） | 校准 / 风险 | C3-3、C1-1、C1-3 | 代码：matrix.py:102 spec；evaluate.py:274 已接 `fallback_trigger_rate`、adaptive.py:164 `fallback_triggered` ｜ 指标：**extreme_picp 未实现**（无事件条件 PICP）、interval_width△、fallback_trigger_rate✓｜ planned |
| **A9** X-only vs X+M vs X+M+ΔT | 缺失模式本身是否含预测信息？（missing_inputs []/[mask]/[mask,delta_t]） | 鲁棒 / 缺失 | C3-4、C3-5 | 代码：matrix.py:117 spec；config `missing_inputs: [mask, delta_t]` ｜ 指标：missing_stress_crps△（evaluate_probabilistic_stress.py）、missing_stress_rmse△（evaluate_missing_stress.py）｜ planned |
| **A10** 风险无辅助 vs 有 OOD/数据项 | 辅助诊断是否改善事件技巧？（risk.auxiliary_terms []↔[ood,data_quality]） | 风险 | C1-1、C1-3、C6-6 | 代码：matrix.py:130 spec；`risk/evaluate_risk_ablation.py`、`risk/proxies.py:260 stylized_event_cost` 已就绪 ｜ 指标：auprc✓（average_precision, metrics.py:229）、brier✓、stylized_cost✓｜ planned |
| **A11** history-only vs oracle ERA5 vs 真预报 | 未来气象信息可得性造成的 gap 有多大？（weather_protocol） | 鲁棒 / 协议 | C2-5、C3-8 | 代码：matrix.py:139 spec；config `weather_protocol: history_only`，availability 已标注 future_era5=oracle_only / issue_time_nwp=unavailable ｜ 指标：rmse✓、crps✓、energy_score✓、ramp_metrics✓（evaluate.py:46 `_ramp_metrics` 块已实现，含 count/brier/auprc/ramp_ks/event_f1/lead_time）｜ planned |

### 1.2 G 族 — 联合概率方法横向对比（G1–G6，共 6 项）

统一指标 `(crps, energy_score, variogram_score, ramp_brier, ramp_ks)`（matrix.py:153），单 variant：`scenario.method` 切换。

| 实验 ID · 名称 | 目的 / 检验问题 | 卖点归类 | 关联约束 ID | 实现状态（代码｜指标｜运行） |
|---|---|---|---|---|
| **G1** Gaussian residual | 参数化残差参考基线 | 概率生成 / 基线 | C3-3、C3-5 | 代码：matrix.py:155 spec；scenario.method=gaussian_residual ｜ 指标：crps✓、energy_score✓、variogram_score✓、ramp_brier✓（evaluate.py:118 ramp 块内 brier）、ramp_ks✓｜ planned |
| **G2** Residual bootstrap | 非参场重采样参考 | 概率生成 / 基线 | C3-3、C3-5 | 代码：matrix.py:156 spec；residual_bootstrap ｜ 指标：同 G1 全套✓｜ planned |
| **G3** Gaussian copula | 经验边缘 + 高斯依赖 | 概率生成 / 基线 | C3-3、C3-5 | 代码：matrix.py:157 spec；gaussian_copula ｜ 指标：同 G1 全套✓｜ planned |
| **G4** CVAE | 隐变量联合场景基线 | 概率生成 / 基线 | C3-3、C3-5 | 代码：matrix.py:163 spec；`generative_baselines.cvae` 已配 ｜ 指标：同 G1 全套✓｜ planned |
| **G5** Diffusion/DDIM | 迭代 score-based 场景基线 | 概率生成 / 基线 | C3-3、C3-5、C3-6 | 代码：matrix.py:164 spec；`generative_baselines.ddim`（diffusion 100 / sampling 20）｜ 指标：同 G1 全套✓｜ planned |
| **G6** Direct CFM（所提方法） | 所提直接条件流匹配 vs 上述生成参考 | 概率生成（主卖点） | C3-3、C3-5、C2-1 | 代码：matrix.py:165 spec；flow 模块主路径 ｜ 指标：同 G1 全套✓｜ planned |

### 1.3 B 族 — 点/边缘预报基线（B0–B9，共 10 项）

统一指标 `(mae, rmse, r2_nse, picp, pinaw, pinball)`（matrix.py:183），单 variant。

| 实验 ID · 名称 | 目的 / 检验问题 | 卖点归类 | 关联约束 ID | 实现状态（代码｜指标｜运行） |
|---|---|---|---|---|
| **B0** Persistence | 与朴素持续预报比 | 基线 | C3-5 | 代码：matrix.py:185；`evaluate_point_references.py` ｜ 指标：mae✓、rmse✓、r2_nse✓（:19）、picp△（点参考无区间，picp/pinaw/pinball 对 B0/B1 多为 N/A）｜ planned |
| **B1** 物理功率曲线 P_pc | 与物理中心本身比 | 基线 / 物理先验 | C3-5、C5-1 | 代码：matrix.py:186；p_pc 点参考 ｜ 指标：mae✓、rmse✓、r2_nse✓；区间类△｜ planned |
| **B2** GRU | 与确定性 RNN 骨干比 | 基线 | C3-5、C5-5 | 代码：matrix.py:187；`deterministic_baselines.py` gru ｜ 指标：mae✓、rmse✓、r2_nse✓、picp/pinaw/pinball 走 evaluate_marginal.py ｜ planned |
| **B3** TCN | 与确定性卷积骨干比 | 基线 | C3-5、C5-5 | 代码：matrix.py:188；tcn ｜ 指标：同 B2✓｜ planned |
| **B4** Marginal Gaussian | 边缘高斯分位数参考 | 基线 / 校准 | C3-3、C3-5 | 代码：matrix.py:189；`evaluate_marginal.py` ｜ 指标：picp✓、pinaw✓（metrics.py:305）、pinball✓（:83）｜ planned |
| **B5** Marginal Student-t | 重尾边缘参考 | 基线 / 校准 | C3-3、C3-5 | 代码：matrix.py:190；student_t ｜ 指标：同 B4✓｜ planned |
| **B6** Quantile + CQR | 分位数 + 共形校准参考 | 基线 / 校准 | C3-3、C3-5 | 代码：matrix.py:191；quantile ｜ 指标：同 B4✓｜ planned |
| **B7** Deep ensemble | 深度集成场景参考 | 基线 / 概率生成 | C3-3、C3-5 | 代码：matrix.py:192；scenario.method=deep_ensemble ｜ 指标：点/边缘指标✓；联合概率指标不在本族 primary_metrics ｜ planned |
| **B8** Graph WaveNet style | 与图时空 SOTA 确定性骨干比 | 基线 / 动态多图 | C5-5、C3-5 | 代码：matrix.py:193；graphwavenet ｜ 指标：同 B2✓｜ planned |
| **B9** AGCRN style | 与图自适应 SOTA 确定性骨干比 | 基线 / 动态多图 | C5-5、C3-5 | 代码：matrix.py:197；agcrn ｜ 指标：同 B2✓｜ planned |

> 图例：✓ = src 有实现且已被对应 CLI 接线；△ = 有近似实现但非同名/需扩展；**粗体未实现** = matrix.py 声明但 src 无对应函数。所有项运行状态均为 planned（`evidence_status="planned_not_run"`）。

---

## 2. 已声明但 src 未实现的指标清单

| 指标名 | 在哪些实验 primary_metrics 中被声明（实验:matrix.py 行号） | src 现状 |
|---|---|---|
| `extreme_rmse` | A0:31、A1:40 | 无事件级 RMSE 函数；`eval/metrics.py` 仅有全量 `masked_rmse`。核查报告 C1-1/C1-3 判定 ✗，需新增事件窗口链接 + 事件级指标 CLI |
| `extreme_mae` | A0:31 | 同上，无事件条件 MAE；仅有全量 `masked_mae` |
| `extreme_picp` | A8:105 | `interval_metrics`(metrics.py:265) 只给全量 PICP，无「极端事件条件下」的 PICP；事件级评估 CLI 缺失 |
| `power_curve_error` | A1:40 | 无功率曲线拟合误差指标函数；physics 模块只做密度修正，未产出该评分 |
| `seasonal_rmse` | A1:40 | 无分季节/分时段 RMSE 切分逻辑 |
| `rolling_picp` | A7:96 | `calibration/adaptive.py` 有 ACI/fallback 逻辑，但未输出滚动 PICP 时间序列；只有静态 `interval_metrics.picp` |
| `training_time` | A5:81 | `benchmark_inference.py` 只测推理时延/吞吐/峰值显存，无训练时长记录（核查报告 C3-6 列为缺口） |

**近似/命名不一致（建议落地时对齐命名，非完全缺失）：**

| 指标名 | 声明处 | src 近似实现 |
|---|---|---|
| `ramp_f1` | A0:31 | evaluate.py:121 ramp 块内有 `event_f1`（binary_event_metrics.f1），未以 `ramp_f1` 命名 |
| `interval_width` | A7:96、A8:105 | metrics.py:295 返回 `mean_width`、:299 `winkler`，未以 `interval_width` 命名 |
| `ode_steps` | A5:81 | train_flow.py:169 仅作输出属性 echo（采样步数本身），非质量评分 |

---

## 3. 配置与运行基线事实（`configs/sdwpf_v3_2.yaml`）

| 配置键 | 取值 | 说明 |
|---|---|---|
| `experiment.seed` / `seeds` | 2025 / `[2025,2026,2027,2028,2029]` | **默认 5 个 seed**；矩阵 `build_experiment_matrix(seeds=(2025..2029))`（matrix.py:233）。距 C0-2/C3-2「≥10 seed」差 5 个，改 config 即可，无需改码 |
| `data.split` | train 0.60 / val 0.15 / calib 0.10 / test 0.15 | 校准与测试时间分离（对应 C3-3 ✓）；`data/window.py:28` 目标不跨 split 边界（C1-4 ✓） |
| `data.weather_protocol` | `history_only` | A11 的基线档位；`future_era5=oracle_only`、`future_issue_time_nwp=unavailable_until_archive_added`（issue-time 档数据未就绪） |
| `data.resolution / history / forecast` | 10 min / 24 步 / 6 步 | 4h 历史 → 1h 前瞻（10min×6） |
| `graphs.enabled` | `[geo, corr, directional, adaptive]` | A2 全图档；`correlation_mode: difference`（A3 差分档） |
| `flow.steps` / `solver` / `coupling` | 8 / euler / independent | A6 主档；A5 的 OT 档靠 override `flow.coupling=optimal_transport` |
| `flow.scenarios_eval` | 100 | 评估场景数（G1–G6 联合概率采样） |
| `calibration.primary` / `alpha` / `aggregate_level` | `split_per_horizon` / 0.10 / farm | A7/A8 的 fallback 基线档；α=0.10；farm 级聚合（C4-2 ✓） |
| `calibration.aci_gamma` / `context_neighbors` / `context_ood_quantile` | 0.01 / 100 / 0.95 | A7 ACI 档、A8 context fallback 档参数 |
| `risk.auxiliary_terms` / weights / FN:FP cost | `[ood, data_quality]`，各 0.25 / 5.0 : 1.0 | A10 有辅助项档；`ramp_durations=[1,3,6]`、`ramp_threshold_fractions=[0.05,0.10,0.20]`、`cvar_levels=[0.90,0.95,0.99]` |

**实验矩阵机制（一句话）：** `build_experiment_matrix` 把每个 `ExperimentSpec` 的 `variants × seeds` 笛卡尔展开为可执行 run contract——每条带稳定 `experiment_id`（logical_id-variant-seed-sha256[:10]）、`overrides` 字典，并统一打 `evidence_status="planned_not_run"`（matrix.py:231–262）。当前规模 = 28 logical specs（A12 + G6 + B10）× variants × 5 seeds，全部处于「已登记、未运行」状态。

---

## 4. 全景结论

1. **覆盖情况**：A/G/B 三族共 **28 项**（A 族 12、G 族 6、B 族 10），已把「物理先验（A0/A1/B1）→ 动态多图（A2/A3/B8/B9）→ 概率生成主链路（G1–G6、A4–A6）→ 校准/风险（A7/A8/A10）→ 缺失与协议鲁棒（A9/A11）→ 点/边缘基线（B0–B9）」的卖点证据骨架搭全，且多数概率/点/边缘指标（crps、energy、variogram、ramp_ks/brier、mae/rmse/r2/picp/pinaw/pinball、auprc/brier/stylized_cost、latency/throughput/peak_vram）在 src 已有实现。
2. **最大空白——事件级证据链**：`extreme_rmse/extreme_mae/extreme_picp` 三项被 A0/A1/A8 声明却完全未实现，叠加 C1-1/C1-3 判定 ✗，意味着「极端天气/鲁棒」这一主打卖点目前**没有独立事件级实验证据**，只能靠全量平均——这是投稿被批的最高风险点（P0）。
3. **机制与对照缺口**：A2 虽做了图叠加消融，但**缺门控干预对照**（frozen/uniform/shuffled/parameter-matched，C2-6 ✗），无法把「门控」从相关图里分离成独立证据；**无参数敏感性实验**（C3-7 ✗，A6 仅扫 ODE 步数，未覆盖 history/Top-K/distance scale/gate width 等）。
4. **统计与基线缺口**：统计层 HAC/DM、聚类 CI 已有，但**缺效应量与 Holm 校正**（C2-2 △）；基线侧 GRU/TCN/GWN/AGCRN 已就位，**缺 Transformer/PatchTST/iTransformer 等常用对照**（C5-5 △）；数据侧仅 SDWPF 单数据集（C0-1/C3-1 ✗），seed 仅 5 个（C0-2 △，距 ≥10 差 5）。
5. **配置即改即得**：扩 seed 到 ≥10、补 issue-time NWP 数据适配器、把 `rolling_picp/training_time/extreme_*` 指标落地，均为增量工作而非重构；矩阵构造已参数化，新增实验只需在 `_core_ablation_specs` 等三处追加 `ExperimentSpec`。
