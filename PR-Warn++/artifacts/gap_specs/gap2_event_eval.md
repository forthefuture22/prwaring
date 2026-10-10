## 缺口 2：极端天气/极端事件级评估（事件定义→窗口链接→事件级指标）（优先级 P0）

> 对照约束：**C1-1**（主打"极端天气"必须有独立事件级证据，禁止用全量平均代替）、**C1-2**（特殊条件需可操作化定义：训练集拟合阈值并冻结，给出阈值数值与分位数）、**C1-3**（事件结果单独成表、公开事件/样本计数）。
> 本规范只做调研与实现设计，不在本机运行训练/实验，不改动现有代码；以下"落点文件/函数/配置键"均已 Read 核实行号。

### 1. 缺口定位

**现状（真实文件/函数/配置键）：**

- `src/prwarn/experiments/matrix.py` 把 `extreme_rmse`/`extreme_mae`/`extreme_picp`（A0 第 31 行、A1 第 40 行、A8 第 105 行）、`ramp_f1`（A0 第 31 行）、`ramp_metrics`（A11 第 142 行）声明为 `primary_metrics`，但 **src 内无任何计算这些"事件级"指标的实现**——它们只是待跑契约里的名字。
- `src/prwarn/cli/evaluate_missing_stress.py:39 _train_extreme_threshold` 是唯一"训练集拟合阈值"的函数：第 47–58 行从 `metadata.feature_index` 取风速列、用 `feature_scaler` 反标准化回原始风速、在 train mask 有效样本上取 `np.quantile(..., quantile)`，默认 `extreme_quantile=0.95`（第 215 行）。但它**只用于"极端条件缺失压力"（`extreme_conditioned_missingness`，第 92–109 行），不产出任何事件级 RMSE/MAE/PICP**；且只覆盖风速单变量，无降水/温度/功率爬坡阈值。
- `src/prwarn/cli/evaluate.py:46 _ramp_metrics` 已能从 `scenarios_test.h5` 算爬坡**概率/检测**指标（brier/auprc/event_f1/lead_time，第 115–131 行），其阈值来自 `config.risk.ramp_threshold_fractions=[0.05,0.10,0.20]` × `farm_capacity`（第 321–329 行），时长来自 `risk.ramp_durations_steps=[1,3,6]`，方向 up/down。这是"事件是否被预测到"的分类证据，**不是"事件发生期间预测误差有多大"的条件误差证据**。
- `src/prwarn/eval/statistics.py:139 contiguous_event_ids(event, timestamps, *, maximum_gap)` 已实现：把连续正样本按 `maximum_gap` 容忍链接成事件 episode（非事件位给 -1）；`statistics.py:94 event_bootstrap_ci(values, event_ids, ...)` 已实现：按完整 episode 聚类重采样做 percentile CI。**两个可复用原语都在，但没有任何 CLI 调用它们来产出事件级指标。**
- `evaluate.py:332–340` 写出 `calibrated_test.h5`，含 `lower/upper/y_farm/y_farm_mask/origin_time`（+ attrs `alpha/calibration_method`）——事件级 PICP 所需的校准区间与 origin 时间已具备，只差"按事件 origin 筛选"这一步。
- C1-4（窗口按完整视野归属不跨 split）已由 `window.py:27 final_origin = len − forecast_steps` 保证，事件归属复用同一 origin 即可。

**缺口是什么：** 缺一条完整链路——(a) 在 train 上拟合并冻结多变量极端阈值（风速分位数 + 爬坡幅度分数 × 额定容量 + 可选温度分位数），把阈值数值/分位数写进产物；(b) 用已有的 `observed_ramp_event` + 高风速掩码生成逐时间步事件真值，再用 `contiguous_event_ids` 链接成 episode 事件表；(c) 只在"预测视野落入事件 episode"的 origin 子集上计算 `extreme_rmse/extreme_mae/extreme_picp`，并用 `event_bootstrap_ci` 给聚类 CI，与全量结果分离成表、公开事件计数。

**为什么必须补（P0 理由）：** 标题/摘要主打"extreme weather"（EPSR 拒稿稿名即如此），审稿人（R1.1/R2.2/R5.1）明确要求事件级独立证据；当前全量平均会被事件内/事件外误差稀释，无法证明模型在极端段真的更准/区间真的更可靠。C1-2 还要求阈值不得触碰验证/测试观测——现有 `_train_extreme_threshold` 已是 train-only，扩展它即可合规。

### 2. 文献依据表

| 文献（作者-年份） | 出处/年份 | DOI/arXiv | 核验状态 | 关键做法 | 可借鉴的协议要素 |
|---|---|---|---|---|---|
| Valldecabres–von Bremen–Kühn 2020 | Wind Energy 23(12):2202–2224 | 10.1002/we.2553 | VERIFIED（Wiley 全文页核对三位作者 Laura Valldecabres / Lueder von Bremen / Martin Kühn、Vol.23 Iss.12 pp.2202–2224、2020-09-14 在线；14 个事件、Westermost Rough） | 爬坡事件由起始时间 ts、时长 Δt、幅度 ΔP 三参数化；阈值以**额定容量百分比**表示，文献区间 10%–75%、时间窗 5 min–6 h；作者用 5 min 功率变化分布的 3σ（≈10% 额定）定阈值；事件按 up/down 分类并按事件计数成表（Table 2：每段事件数/最大幅度/时长）；在观测到的事件期间评估密度预测 | 阈值=额定容量百分比（与现有 `ramp_threshold_fractions×farm_capacity` 一致）；事件按 episode 计数成表；**事件期间单独评估**概率预测 |
| Cui–Feng–Wang–Zhang 2018 | IEEE Trans. Sustainable Energy 9(1):261–272 | 10.1109/TSTE.2017.2727321 | VERIFIED（IEEE Xplore doc 7981390 + Scilit/SciSpace 核对卷期页/DOI/作者） | 把爬坡抽取为一组 ramping 特征（幅度、时长、变化率、起始时间、方向），用 GGMM 刻画其非高斯/多峰分布 | 事件特征（幅度/时长/方向）作为事件表的列；极端段是厚尾分布，事件级误差不能假设正态 |
| Cui–Zhang–Feng–Florita–Sun–Hodge 2017 | Renewable Energy 111:227–244 | 10.1016/j.renene.2017.04.005 | VERIFIED（ORCID 作者页 + Google Scholar 核对期刊/卷页/作者） | 在风/光/负荷/净负荷上统一刻画与统计 ramp 事件，跨序列报告事件发生率与特征分布 | 跨数据集同一事件定义口径；事件计数/发生率需与全量结果并列报告 |

> 说明：三篇均为顶刊（Wind Energy / TSTE=目标期刊 / Renewable Energy），均逐篇核验标题-作者-期刊-年份-DOI 一致。未找到可核验的"事件内 PICP"专文，故事件内 PICP 只在现有 `interval_metrics`（evaluate.py:313）基础上加 `in_event` origin 掩码实现，不额外引用未核验文献。

### 3. 标准做法要点

1. **阈值以额定容量百分比 + 分布分位数双轨，且 train-only 冻结**：爬坡幅度阈值用 `ramp_fraction × farm_rated_capacity`（Valldecabres–von Bremen–Kühn 2020 的 10%–75% 区间内，现有 config 0.05/0.10/0.20）；高风速/温度极端用 train 分布分位数（现有 `_train_extreme_threshold` 的 train-quantile 做法），阈值数值与分位数全部写入 artifact。（Valldecabres–von Bremen–Kühn 2020；C1-2）
2. **事件 = 连续 episode，需窗口链接而非逐点**：逐时间步事件真值用 `observed_ramp_event`（risk/proxies.py:68，已含 up/down 方向与 mask）生成，再用 `contiguous_event_ids`（statistics.py:139）以 `maximum_gap` 容忍链接成 episode；相邻正步若间隔 ≤ `link_max_gap_steps` 视为同一事件。（Cui et al. 2018；复用已有原语）
3. **事件级指标只在"预测视野落入事件 episode"的 origin 上计算**：对每个 origin 的 6 步预测窗口，若其 future horizon 命中任一 episode 的时间范围，则 `in_event=True`；在该子集上算 RMSE/MAE/PICP，与全量 `metrics["probabilistic"]/["interval"]` 分离。（Valldecabres–von Bremen–Kühn 2020：在观测到的 ramp 期间评估）
4. **事件级 CI 按 episode 聚类 bootstrap，不按逐点**：事件内样本时间相关，必须用 `event_bootstrap_ci`（statistics.py:94）对完整 episode 重采样，`n_events` 为链接后的 episode 数；事件过少时如实报告。（Cui et al. 2017；C1-3）
5. **事件表与计数单独成表**：输出每个 episode 的 start/end/direction/max_magnitude/duration，以及"链接窗口数 / episode 数 / 有效 origin 数"，与全量指标分开。（C1-3；Valldecabres Table 2 惯例）

### 4. 实现规范（映射到本代码库）

| 改动点 | 落点文件/函数/配置键 | 新增参数与默认值 | 协议要点 | 验收方式 |
|---|---|---|---|---|
| train-only 多变量极端阈值拟合 | 新建 `src/prwarn/eval/event_definition.py` 的 `fit_extreme_thresholds(train: ProcessedSplit, metadata, *, wind_quantile=0.95, temperature_quantile=0.95) -> dict`（复用 `evaluate_missing_stress.py:47–58` 的反标准化逻辑） | `wind_quantile=0.95`、`temperature_quantile=0.95`；返回 `{"wind_speed_mps": float, "temperature_c": float}`；爬坡幅度不由训练拟合，而由 `risk.ramp_threshold_fractions × farm_capacity`（冻结自 config） | 只从 train + metadata 反标准化取值，绝不读 val/test；阈值数值与所用分位数写进产物 JSON（满足 C1-2） | 单测：构造小 train split，返回阈值与手算分位数一致；断言函数不接受 test split |
| 逐时间步事件真值 + episode 链接 | `src/prwarn/eval/event_definition.py` 的 `link_event_episodes(origin_time, y_farm, y_farm_mask, current_y_farm, current_y_farm_mask, thresholds, *, farm_capacity, resolution_minutes) -> tuple[np.ndarray, list[dict]]` | 内部调用 `risk.proxies.observed_ramp_event`（已有，proxies.py:68）逐 duration/direction 生成 ramp 真值，叠加高风速/温度掩码；调用 `statistics.contiguous_event_ids(event, timestamps, maximum_gap=link_max_gap_steps*resolution_minutes)`（已有，statistics.py:139） | C1-4：事件时间轴为校准后的 origin/future 时间，不跨 split；输出事件表每行：event_id/start/end/direction/max_magnitude_pct/duration_steps | 对已知 ramp 段能链接出正确 episode 数；非事件位为 -1 |
| 事件级指标 CLI（extreme_rmse/mae/picp） | 新建 `src/prwarn/cli/evaluate_events.py`（读 `calibrated_test.h5`：lower/upper/y_farm/y_farm_mask/origin_time；读 `scenarios_test.h5`：y_det_farm/current_y_farm） | `--wind-quantile 0.95`、`--link-max-gap-steps 1`、`--bootstrap-n 2000`、`--bootstrap-seed 0` | 用 `link_event_episodes` 得 episode 表；按"origin 视野是否命中 episode"建 `in_event [sample]`；子集上算 `extreme_rmse/extreme_mae`（用 y_det_farm 残差）与 `extreme_picp = mean(y_farm∈[lower,upper])`；用 `event_bootstrap_ci`（statistics.py:94）给 CI | 产出 `events_metrics.json`，含 extreme_rmse/extreme_mae/extreme_picp 及 bootstrap CI，与 overall 对照 |
| 配置键 | `configs/sdwpf_v3_2.yaml` 新增 `event:` 段（紧邻现有 `risk:` 段，第 88–97 行） | `event.wind_quantile: 0.95`、`event.temperature_quantile: 0.95`、`event.link_max_gap_steps: 1`、`event.bootstrap_n: 2000` | 与 `risk.ramp_threshold_fractions=[0.05,0.10,0.20]`、`risk.ramp_durations_steps=[1,3,6]` 协同（复用现有，不重复定义爬坡阈值） | `yaml.safe_load` 后 `config["event"]` 四键齐全 |
| matrix.py 指标接线 | `src/prwarn/experiments/matrix.py`（声明已存在，不改名） | 无新增；要求 `evaluate_events.py` 输出的 JSON key 名为 `extreme_rmse/extreme_mae/extreme_picp`，使 `statistics.aggregate_seed_metrics`（statistics.py:189）能按同名 key 跨 seed 聚合 | 不新造指标名，避免与 matrix.py:31/40/105 已声明名漂移 | 种子聚合后 events_metrics 各 key 出现在 `aggregate_seed_metrics` 输出里 |

### 5. 验收自检清单

- [ ] `python -m prwarn.cli.evaluate_events --flow-run <dir> --config configs/sdwpf_v3_2.yaml --output <dir>/events_metrics.json` 退出码 0。
- [ ] `events_metrics.json` 含 `thresholds` 段：`wind_speed_mps`（数值）、`wind_quantile=0.95`、`temperature_c`、`ramp_threshold_fractions`；且阈值仅由 train/metadata 反标准化得到（可复算一致）。
- [ ] 含 `event_table`：每行 event_id/start/end/direction/max_magnitude_pct/duration_steps；含 `event_counts`：`episodes`、`linked_windows`、`in_event_origins`、`overall_origins`。
- [ ] 含 `extreme_rmse/extreme_mae/extreme_picp`（带 `bootstrap_ci` 下界/上界，`n_events`=episode 数），并与 `overall_rmse/overall_mae/overall_picp` 并列；子集过小时 `extreme_*` 给 `null` 并注明。
- [ ] 事件归属不跨 split：`in_event` 仅在 test 的 origin 上判定，复用 `window.py:27` 的不跨边界事实。
- [ ] 校准区间在事件级不重算：`extreme_picp` 用 `calibrated_test.h5` 已冻结的 lower/upper（calib 拟合、test 应用），不在事件子集上重新 conformal。

### 6. 引用来源列表

1. Valldecabres, L., von Bremen, L., Kühn, M. Minute-scale detection and probabilistic prediction of offshore wind turbine power ramps using dual-Doppler radar. *Wind Energy* 23(12):2202–2224, 2020. DOI: 10.1002/we.2553. URL: https://onlinelibrary.wiley.com/doi/10.1002/we.2553
2. Cui, M., Feng, C., Wang, Z., Zhang, J. Statistical Representation of Wind Power Ramps Using a Generalized Gaussian Mixture Model. *IEEE Transactions on Sustainable Energy* 9(1):261–272, 2018. DOI: 10.1109/TSTE.2017.2727321. URL: https://doi.org/10.1109/TSTE.2017.2727321
3. Cui, M., Zhang, J., Feng, C., Florita, A.R., Sun, Y., Hodge, B.-M. Characterizing and analyzing ramping events in wind power, solar power, load, and netload. *Renewable Energy* 111:227–244, 2017. DOI: 10.1016/j.renene.2017.04.005. URL: https://doi.org/10.1016/j.renene.2017.04.005
