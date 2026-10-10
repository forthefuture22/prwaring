# 缺口 2 实施报告：极端天气事件定义 + 事件级评估 CLI（extreme_rmse / extreme_mae / extreme_picp）

- 日期：2026-10-10
- 依据：`artifacts/gap_specs/gap2_event_eval.md`（逐条对照）
- 约束：C1-1（事件级独立证据）、C1-2（train-only 冻结阈值）、C1-3（事件单独成表 + 公开计数）、C1-4（事件时间轴不跨 split）
- 所有权：仅新增 3 个文件；未改动 matrix.py / statistics.py / proxies.py / evaluate.py / evaluate_missing_stress.py / train_flow.py / configs/sdwpf_v3_2.yaml；未做任何 git 操作。

---

## 1. 改动文件清单（绝对路径 + 行号范围 + 说明）

### 1.1 新增 `C:\Users\weface\Desktop\prwaring-main\PR-Warn++\src\prwarn\eval\event_definition.py`（277 行）

| 行号 | 内容 |
|---|---|
| 1–37 | 模块 docstring：协议说明 + 重要口径声明（h5 只导出 farm 级功率，无未来天气序列，故 episode 真值=观测 ramp 并集；风速/温度阈值 train 拟合后入产物报告）。 |
| 39–50 | `_column_index(metadata, column_key)`：由 `physical_columns` 键（wind_speed/temperature）映射到 `feature_index` 列号，KeyError 兜底。 |
| 52–62 | `_inverse(split, index, scaler)`：`x[...,idx]*std+mean` 反标准化（复用 evaluate_missing_stress.py:47–58 口径），返回 (raw, valid)。 |
| 64–132 | `fit_extreme_thresholds(train, metadata, *, wind_quantile=0.95, temperature_quantile=0.95) -> {"wind_speed_mps","temperature_c"}`。要点：① 分位数越界抛错；② `feature_scaler.fit_split=="train"` 否则拒绝（C1-2）；③ 若 metadata 带 `split_manifest.train.{start,end}`，校验传入 split 的 origin_time 落在训练窗内，否则拒绝（val/calib/test 会被拒）；④ 风速取 train mask 有效样本分位数；⑤ 温度按 `physical_units.temperature_input` 换算到摄氏度（celsius 原值 / kelvin−273.15）后取分位数。爬坡幅度阈值**不在此拟合**。 |
| 135–149 | `future_timestamp_grid(origin_time, horizon, resolution_minutes)`：所有 origin 的 future 时间步并集，升序 `datetime64[ns]`；future[h] 对齐 origin+(h+1) 步，与 observed_ramp_event 语义一致。 |
| 151–277 | `link_event_episodes(origin_time, y_farm, y_farm_mask, current_y_farm, current_y_farm_mask, thresholds, *, farm_capacity, resolution_minutes, link_max_gap_steps=1)`。要点：① 逐 (fraction, duration, direction) 调 `risk.proxies.observed_ramp_event` 生成 ramp 真值与有效 mask；② 用 searchsorted 把 (sample,horizon) 事件标志投影到并集时间轴；③ 用 `eval.statistics.contiguous_event_ids(..., maximum_gap=link_max_gap_steps*resolution)` 链接 episode，非事件位 −1；④ 事件表每行 event_id/start/end/direction/max_magnitude_pct/duration_steps（方向与幅度取"进入步"参与的最大变幅窗口符号，保证单步 ramp 也有方向）。 |

### 1.2 新增 `C:\Users\weface\Desktop\prwaring-main\PR-Warn++\src\prwarn\cli\evaluate_events.py`（253 行）

| 行号 | 内容 |
|---|---|
| 1–33 | docstring + 用法。 |
| 36–46 | `_resolve_calibrated`：优先 `flow_run/evaluation/calibrated_test.h5`（evaluate.py:332 默认输出位），回退 `flow_run/calibrated_test.h5`。 |
| 49–60 | `_ci_section`：BootstrapCI → JSON dict（estimate/lower/upper/confidence/n_boot/n_events）。 |
| 62–115 | `_conditional_summary`：在事件步上算 extreme_rmse/mae/picp；用 `event_bootstrap_ci`（按完整 episode 聚类重采样）给 CI（RMSE 的 statistic=sqrt(mean)）；事件步<2 或 episode<2 时 extreme_* 置 null 并写 note。 |
| 118–131 | CLI 参数：`--flow-run`、`--data-dir`（train.npz+metadata.json，train-only 阈值来源）、`--config`、`--output`、`--wind-quantile 0.95`、`--temperature-quantile 0.95`、`--link-max-gap-steps 1`、`--bootstrap-n 2000`、`--bootstrap-seed 0`。 |
| 132–181 | 主流程：读 config → 读 metadata/train 拟合并冻结阈值 → 读 calibrated_test.h5（lower/upper/y_farm/y_farm_mask/origin_time，区间为已冻结校准值，不在事件子集重算 conformal）→ 读 scenarios_test.h5（y_det_farm/current_y_farm/current_y_farm_mask，attrs rated_power×n_nodes=farm_capacity）→ link_event_episodes → 把每个 (origin,horizon) 步投影到时间轴取 episode id。 |
| 182–238 | 计算 overall_*（全量有效步）与 extreme_*（事件步）；组装 payload：`thresholds`（含 wind_speed_mps/wind_quantile/temperature_c/temperature_quantile/ramp_threshold_fractions/farm_capacity/fitted_on=train_only）、`event_counts`（episodes/linked_event_timesteps/event_steps/in_event_origins/overall_origins）、`event_table`、顶层数值键 `extreme_rmse/extreme_mae/extreme_picp`（与 matrix.py:31/40/105 同名，供 statistics.aggregate_seed_metrics 跨 seed 聚合）+ `overall_rmse/overall_mae/overall_picp` + `*_ci` 嵌套段。 |
| 239–253 | 写 events_metrics.json（拒绝覆盖）+ 摘要 stdout。 |

### 1.3 新增 `C:\Users\weface\Desktop\prwaring-main\PR-Warn++\tests\test_event_eval.py`（293 行）

| 用例 | 验证点 |
|---|---|
| `FitThresholdTests::test_thresholds_match_hand_computed_quantile` | 小 train split，函数输出与手算分位数一致（按全部 node/history 展平复算）。 |
| `FitThresholdTests::test_rejects_split_outside_train_window` | train 窗内 split 接受；窗外（模拟 test）split 抛 ValueError。 |
| `LinkEpisodeTests::test_links_ramp_episodes_and_marks_non_event_minus_one` | 已知 down-ramp(100→40)+up-ramp(40→100)：链接出 2 个 episode，非事件位为 −1，方向集合 {down,up}，幅度>0。 |
| `EventCliTests::test_cli_end_to_end_values_and_keys` | 合成 calibrated_test.h5+scenarios_test.h5+train.npz+metadata.json，CLI 子进程退出码 0；extreme_rmse/mae=2.0（det=truth+2 手算）、extreme_picp=1.0（区间恒含真值）；JSON 含字面量 extreme_* 键、事件表列齐全、CI 的 n_events=2。 |
| `EventCliTests::test_bootstrap_ci_is_reproducible` | 同 seed 两次 event_bootstrap_ci 结果完全相等。 |

---

## 2. 验收自检（对照规范第 5 节）

- [x] CLI 退出码 0 —— 证据：`tests/test_event_eval.py` 中子进程 `run_cli()` 断言 returncode==0（stderr 为空）；独立 smoke 打印 `exit 0`。
- [x] `events_metrics.json` 含 `thresholds` 段：`wind_speed_mps`（数值）、`wind_quantile=0.95`、`temperature_c`、`ramp_threshold_fractions`；阈值仅由 train/metadata 反标准化得到 —— 证据：smoke 输出 thresholds 段齐全；`fitted_on:"train_only"`；单测①手算一致 + 拒绝窗外 split。
- [x] 含 `event_table`（event_id/start/end/direction/max_magnitude_pct/duration_steps）与 `event_counts`（episodes/linked_event_timesteps/event_steps/in_event_origins/overall_origins）—— 证据：smoke JSON：episodes=2、event_steps=6、in_event_origins=6、overall_origins=10。
- [x] 含 extreme_rmse/extreme_mae/extreme_picp（带 bootstrap_ci lower/upper，n_events=episode 数），与 overall_* 并列；子集过小时置 null 并注明 —— 证据：smoke 中 extreme_rmse_ci.n_events=2；`_conditional_summary` 小样本分支返回 null+note。
- [x] 事件归属不跨 split —— 证据：in_event 仅在 test 的 origin_time 上判定，复用 evaluate.py 导出的 origin_time，不拼接其他 split；C1-4 由 window.py:27 final_origin 既有事实保证。
- [x] 校准区间在事件级不重算 —— 证据：extreme_picp 直接使用 calibrated_test.h5 已冻结的 lower/upper（calib 拟合、test 应用），代码中无任何在事件子集上重新 conformal 的步骤。

附加验证：
- `python -m compileall` 三个新文件：通过。
- `python -m pytest tests\test_event_eval.py -v`：**5 passed**。
- 全量回归 `python -m pytest tests\`：**79 passed, 1 skipped**（skip 为 gap3 预存标记，与本次无关），无回归。

---

## 3. 需要服务器/真实数据运行清单（本机无真实 h5，仅合成 smoke）

1. 用真实 `train_flow.py` 产出的 flow run（含 `scenarios_test.h5` 与 `evaluation/calibrated_test.h5`）执行：
   `python -m prwarn.cli.evaluate_events --flow-run outputs/<run> --data-dir data/<processed> --config configs/sdwpf_v3_2.yaml --output outputs/<run>/evaluation/events_metrics.json --bootstrap-n 2000 --bootstrap-seed 0`
2. 多 seed 聚合：对每个 seed 产出的 events_metrics.json，把含数值的顶层键（extreme_rmse/extreme_mae/extreme_picp/overall_*）喂给 `prwarn.eval.statistics.aggregate_seed_metrics`，确认 extreme_* 出现在聚合输出（key 名已对齐 matrix.py 声明）。
3. 真实数据下检查事件计数量级（episodes / in_event_origins / overall_origins）；若 episode<2，extreme_* 会如实输出 null 并在 subset_note 说明，需在论文/报告中披露事件数。
4. 核对 `metadata["physical_units"].temperature_input` 在真实数据上的取值（celsius/kelvin），确认 temperature_c 阈值单位正确。
5. 确认 calibrated_test.h5 位于 evaluation/ 子目录（evaluate.py 默认）；若放在别处，`_resolve_calibrated` 已回退根目录。

---

## 4. 待编排者集成的配置键（本次未改 configs/sdwpf_v3_2.yaml）

在现有 `risk:` 段（第 88–97 行）之后插入：

```yaml
event:
  wind_quantile: 0.95
  temperature_quantile: 0.95
  link_max_gap_steps: 1
  bootstrap_n: 2000
```

说明：CLI 当前以同名命令行参数为默认值；集成后可让 CLI 读取 `config["event"]` 作为默认（--link-max-gap-steps/--bootstrap-n 等），与 risk.ramp_threshold_fractions=[0.05,0.10,0.20]、ramp_durations_steps=[1,3,6] 协同（爬坡阈值复用 risk 段，不重复定义）。

---

## 5. 已知设计口径（写入论文/核查报告时需一致）

- 事件 episode 真值 = 观测 farm 功率 ramp（observed_ramp_event 遍历 fraction×duration×direction 后取并集）。风速/温度阈值 train 拟合后作为冻结阈值入产物报告；当前 test h5 不含未来天气序列，故未在事件真值上叠加高风速/温度掩码（event_definition.py docstring 已声明此口径）。
- extreme_* 条件样本 = 落入事件 episode 的有效 (origin,horizon) 步；overall_* = 全部有效步；两者并列报告。
- bootstrap CI 按完整 episode 聚类重采样（event_bootstrap_ci），非逐点。
