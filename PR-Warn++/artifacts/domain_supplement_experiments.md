# 领域常规实验盘点 + 方法针对性补充实验建议（PR-Warn++ / IEEE TSTE）

> 任务范围：纯调研 + 规范设计。**未运行任何实验、未改任何代码/配置。**
> 对照基准：①实验全景表 `artifacts/experiment_landscape.md`（A0–A11 / G1–G6 / B0–B9 共 28 项，含实现状态与 7 项声明未实现指标清单）。
> 代码行号均经本次 Read 核对（`src/prwarn/` 下 `experiments/matrix.py`、`models/backbone.py`、`models/deterministic_baselines.py`、`cli/evaluate.py`、`cli/evaluate_missing_stress.py`、`cli/benchmark_inference.py`、`eval/statistics.py`、`risk/proxies.py`、`data/stress.py`，及 `configs/sdwpf_v3_2.yaml`）。
> 文献核验协议：general_search + web_fetch 权威记录页（arXiv abs / 作者主页 / 出版社页）逐条交叉核验；标 **VERIFIED**。凡记忆中无法落到权威记录页的 arXiv/DOI 一律不用（本轮已剔除两条错误记忆 ID：arXiv:1806.00142、arXiv:2007.02846）。
> 版本：v1.0 · 2026-10-10

---

## 一、领域常规实验盘点表

> 依据 IEEE TSTE / TSG / TPWRS、Applied Energy、Renewable Energy、Energy、Wind Energy、NeurIPS/ICLR/KDD 等顶刊顶会中风电功率时序与极端天气预测的「常规出现 / 审稿人常要求」实验类型。
> 对照列判定口径：**已覆盖** = ①全景表已有完整同类实验且指标已接线；**部分覆盖** = 有同类骨架/近似指标，但关键分解或对比维度缺失；**未设计** = 矩阵无对应 spec 或指标声明了却未实现。

| # | 实验类型 | 领域普遍性依据（可核验文献） | 常见做法要点 | 对照①全景表（判定 + 依据实验 ID / 文件） |
|---|---|---|---|---|
| 1 | **多步误差传播（逐 horizon 误差曲线）** | Gneiting & Raftery 2007（proper scoring 按预测步长评估）[V4]；Godahewa et al. 2021 Monash Archive（按数据集/horizon 分报误差协议）[V8]；Zhou et al. 2024 SDWPF（空间动态多步风功率）[V6] | 画 RMSE/MAE/CRPS/PINAW 随 lead time 上升的曲线，标注误差加速段；区分训练-部署 horizon 差距；报告每步显著性 | **部分覆盖**。区间指标已逐 horizon：`evaluate.py:316-320` 对 `h1..hH` 调 `interval_metrics`（PICP/PINAW/Winkler），爬坡指标逐 horizon（`evaluate.py:87-98`）。**缺口**：点预测 RMSE/MAE 未按 horizon 分解——`evaluate_missing_stress.py:172-181` `error.square()` 在 horizon 维聚合；CRPS 也只在 `evaluate.py:289-291` 跨步聚合（`crps_total/crps_count`）。即区间逐步✓、点/概率逐步曲线✗ |
| 2 | **跨季节 / 跨风况 / 跨场址分段评估** | Godahewa et al. 2021（跨 20 异构数据集同协议分报）[V8]；Islam et al. 2024（跨气候区域自适应）[V3]；Draxl et al. 2015 / Plumley 2022（多场址公开数据源）[V7][V9] | 按季节、风速分箱、风向扇区、湍流强度分段报指标；不同风场分行报告，不跨池平均；协议（窗口/掩码/调参预算）跨场一致 | **未设计**。`seasonal_rmse` 在 `matrix.py:40`（A1）声明但 **src 未实现**；`configs/sdwpf_v3_2.yaml` 仅 SDWPF 单数据集、单时序 split（train/val/calib/test=0.60/0.15/0.10/0.15）；无风速/风向扇区分段逻辑。对应缺口1（多数据集适配器） |
| 3 | **可靠性图与校准诊断（reliability diagram / CRPS 分解）** | Gneiting & Raftery 2007（calibration + sharpness 联合评分）[V4]；Gneiting et al. 2005（EMOS/minimum-CRPS 校准）[V5]；Wen, Pinson et al. 2024（CRPS 风功率概率预测）[V1] | 画 PIT/名义覆盖率-经验可靠性图；把 CRPS/BS 分解为可靠性项 + 尖锐度项；报告 sharpness（区间宽度）固定后的校准增益 | **部分覆盖**。事件可靠性图已有：`evaluate.py:130` `reliability_bins(event, probability)`（爬坡概率 vs 观测事件）；`farm_crps`✓、`dependence_diagnostics`✓。**缺口**：连续预测分布的 PIT / 名义覆盖率分层可靠性图未做；CRPS 的 reliability-sharpness 分解未做；`rolling_picp` 在 `matrix.py:71`（A7:96）声明未实现 |
| 4 | **异常 / 极端事件检测与条件技能** | Wen, Pinson et al. 2024（极端/不完备数据下的概率风功率）[V1]；Zhou et al. 2024 SDWPF（机组阵列极端/爬坡）[V6]；Gneiting & Raftery 2007（事件概率 proper scoring）[V4] | 定义事件（爬坡幅度×额定容量、高分位功率）；报 Brier/AUPRC/F1/检测提前量；**并单独报事件期间的条件误差**（事件表与全量表分离） | **部分覆盖**。检测指标已接线：`evaluate.py:46 _ramp_metrics` 输出 brier/auprc/event_f1/lead_time（:118-127）。**缺口**：事件条件误差表缺失——`extreme_rmse/extreme_mae/extreme_picp` 在 `matrix.py:31/40/105`（A0/A1/A8）声明却完全未实现（全景表 §2）。即"检出事件"✓、"事件期间准不准"✗。对应缺口2 |
| 5 | **新场址迁移 / 跨场泛化** | Islam et al. 2024（跨位置深度域自适应，源站预训练+目标站微调）[V3]；Plumley 2022 Kelmarsh（外部 6 机真实风场）[V9]；Draxl et al. 2015 WIND Toolkit（格点 NWP 第三数据）[V7] | 源站预训练 → 目标站少样本微调/冻结主干；同输入契约/同 split/同指标；报告零样本 vs 微调曲线；区分 within-farm retrain 与跨场迁移 | **未设计**（缺口1 已规划）。`configs/sdwpf_v3_2.yaml` 仅单数据集；`docs/约束符合性核查报告.md:15` 明确 Kelmarsh「需先编写数据适配器」。缺口1（Kelmarsh + WIND Toolkit 同协议适配器）即补此项 |
| 6 | **长程 / 多日预报** | Nie et al. 2023 PatchTST（长时序 Transformer 基准）[V10]；Liu et al. 2024 iTransformer（逆范式长程预测）[V11]；Godahewa et al. 2021（多 horizon 协议）[V8] | 拉长 horizon（24h–7d）；评估长程误差饱和/漂移；与短程模型公平对比；气象预报可用性随 horizon 衰减 | **未设计**。`configs/sdwpf_v3_2.yaml` `history_steps:24`（4h 历史）、`forecast_steps:6`（10min×6 = **仅 1h 前瞻**）；无多日 horizon 配置。本方法定位短时极端预警，长程属能力边界，需在文中显式声明 |
| 7 | **输入特征敏感性与因果分析** | Chen & Guestrin 2016 XGBoost（特征重要性/置换评估）[V12]；Godahewa et al. 2021（统一评估协议）[V8]；Zhou et al. 2024 SDWPF（多气象通道）[V6] | 置换重要性 / permutation importance；逐通道消融；混淆因素（限电、共享控制）分层；区分相关与因果 | **部分覆盖**。组件级消融已有：A2（图叠加 `matrix.py:46`）、A3（raw vs difference 相关 `:60`）、A9（缺失掩码信息 `:117`）。**缺口**：无系统化输入特征重要性/permutation；`graph_weights` 仅作张量记录非标量解读。注：缺口4（OFAT S1–S7）是**超参**敏感性（history/Top-K/distance scale/gate width），不含输入通道重要性 |
| 8 | **经济成本 / 收益评估** | Genoese et al. 2016 Energy Policy（COIF 不平衡清算成本、偏差-协方差分解）[V14]；Wen, Pinson et al. 2024（决策感知概率预测）[V1] | 把预测误差映射到不平衡/调峰/备用成本；报成本随预测精度曲线；区分偏差成本与方差成本；与完美预报基准对比 | **部分覆盖**。风险决策代理已有：`risk/proxies.py:260 stylized_event_cost`（非对称 FN:FP）、`:133 cvar_shortfall`、`:288 select_cost_threshold`。**缺口**：均为 stylized 代理（`proxies.py:268` 自述"this is a proxy, not grid economics"），无实际电力市场结算/不平衡电价/电网调度成本建模 |
| 9 | **缺失与插补策略对比** | Wen, Pinson et al. 2024（Marginalize rather than Impute：impute-then-predict vs 边际化）[V1]；Zhou et al. 2024 SDWPF（SCADA 缺失/质量码）[V6] | 在相同缺失机制（MCAR/块/空间）下对比插补算法（前向填充/线性/KNN/MICE/GRUD）；报插补后下游误差；区分"对缺失输入鲁棒"与"插补算法优劣" | **部分覆盖**。缺失压力协议已有：`data/stress.py:56/72/99/125`（mcar/block/spatial_outage/extreme_conditioned）+ `:159 apply_missingness_stress`（constant/forward_fill）；A9 消融缺失信息。**缺口**：现有是"模型对缺失输入的鲁棒性"，**未做插补算法横向对比**（缺 KNN/MICE/GRUD 等基线）。缺口5（加性噪声+特征级/关键气象通道缺失）补鲁棒维度，不补插补算法对比 |
| 10 | **训练数据量敏感性（learning curve）** | Godahewa et al. 2021（多数据量协议）[V8]；Islam et al. 2024（目标站少样本/微调数据量）[V3]；Chen & Guestrin 2016（数据量-精度曲线）[V12] | 扫训练数据比例（10%–100%）画 learning curve；判数据饱和点；评估小样本场景泛化 | **未设计**。`matrix.py` 无数据量相关 spec；`configs/sdwpf_v3_2.yaml` 无数据比例 knob；`build_experiment_matrix` 仅按 variant×seed 展开，无 data-size 轴 |

**盘点计数：部分覆盖 6 类（#1/#3/#4/#7/#8/#9）｜ 未设计 4 类（#2/#5/#6/#10）｜ 已覆盖 0 类。**
（说明：0 类"已覆盖"是诚实结论——全景表 §4 自陈最大空白为事件级证据链，上述 10 类均只搭了骨架或做了代理，尚无一类达到顶刊常规要求的完整分解/对比。）

---

## 二、方法针对性补充实验建议表（10 条）

> 视角：站在 IEEE TSTE 审稿人立场，针对本方法的**弱物理先验 → 动态多图门控融合 → 残差 Direct CFM 概率生成 → conformal 校准 → 风险代理**五段，挑「最可能被追问、最能证明方法价值」且**①全景表与 8 缺口均未覆盖**的实验。
> 与缺口重叠者标注「缺口 N 已覆盖」并写明差异/补强点，不重复设计。

| # | 建议实验名称 | 动机（对应审稿质疑点） | 顶刊依据（VERIFIED） | 在本代码库落地位置（文件 / 函数 / 配置键，行号） | 与现有实验 / 8 缺口的关系 |
|---|---|---|---|---|---|
| **S1** | **门控权重的工况条件分布诊断**（gating-weight distribution by wind regime） | 「你说动态门控，它到底动态了什么？是否随风向/风速工况合理选图，还是退化为固定权重？」C2-4 要求机制解释有界——需证明门控在响应物理工况而非随机 | Zhou et al. 2024 SDWPF（风向/空间关系）[V6]；Gneiting & Raftery 2007（诊断性评分）[V4] | `models/backbone.py:244` 输出 `graph_weights`（[B,4]，来自 `:180-182 softmax`）；按 `data/stress.py:304 wind_from`（风向）与风速分箱聚合权重分布；新增 `eval/diagnostics.py`（或扩 `evaluate.py`）；配置复用 `graphs.enabled`，新增诊断分箱键 | **缺口3（门控干预 learned/uniform/frozen/shuffled/parameter-matched）已覆盖"替换权重→性能"**；S1 是"描述权重在工况下如何分布"，证明门控**行为可解释**。互补不重复：缺口3 证有用性，S1 证合理性 |
| **S2** | **共形区间的条件覆盖分层审计**（conditional coverage by regime） | 「split/ACI conformal 只保证边际覆盖；在极端/漂移工况下条件 PICP 是否达标？」这是共形预测最常见的审稿攻击点 | Gibbs & Candès 2021 ACI under shift [V2]；Gneiting & Raftery 2007（calibration+sharpness）[V4] | `cli/evaluate.py:316-320` 已有逐 horizon `interval_metrics`；扩展按风速分箱 / OOD 分数（`proxies.py:157 MahalanobisOOD`）分层算条件 PICP vs 名义 α=0.10；落地全景表 §2 声明未实现的 `rolling_picp`（A7:96） | **A7/A8 已覆盖"方法间对比"（static vs ACI vs context_fallback）**；S2 是选定方法内的**条件覆盖审计**，并补 `rolling_picp` 未实现项。不重复 |
| **S3** | **残差物理中心的误差传播与偏差-尖锐度分解** | 「两阶段 P_pc + 残差 CFM 会不会把物理中心的系统偏差带进最终场景？限电/极端时中心不准怎么办？」 | Gneiting & Raftery 2007（CRPS calibration-sharpness 分解）[V4]；Wen, Pinson et al. 2024（残差概率预测）[V1] | `physics/power_curve.py EmpiricalPowerCurve.predict`（物理中心）；`backbone.py:234` `centre = p_pc + residual`；把中心偏差（y_det vs 真值）与场景离散度（尖锐度）分解，归因总 CRPS/区间宽度来源；配置复用 `physics` 段 | **A0 已覆盖"direct vs physics_residual 结构选择"**；S3 是残差路径**内部**的误差传播诊断（中心偏差如何流到最终场景）。不重复 |
| **S4** | **场内空间留一泛化**（leave-one-turbine / spatial-block-out） | 「多图融合学到的是真空间相关还是邻接过拟合？屏蔽一组机组连接/留一节点，泛化还成立吗？」 | Islam et al. 2024 跨位置域自适应 [V3]；Zhou et al. 2024 SDWPF 空间机组阵列 [V6] | `backbone.py` 静态图 `a_geo.npy/a_corr.npy`（`evaluate_missing_stress.py:234-236` 加载）；训练时对 adjacency 加 node dropout / 留一机组子图；评估 held-out 节点预测；配置新增 `training.spatial_dropout` 比例 | **缺口1（跨场址迁移）已覆盖"跨农场"**；S4 是**同农场内节点级空间留一**（不同泛化轴：空间结构 vs 场址分布）。不重复 |
| **S5** | **风险代理的决策曲线与成本比敏感性**（FN:FP cost sweep） | 「风险代理给个概率，运营方怎么用？不同误报/漏报代价比下阈值最优吗？比气候态基线好多少？」 | Genoese et al. 2016 Energy Policy（COIF 不平衡成本）[V14]；Wen, Pinson et al. 2024（决策感知）[V1] | `risk/proxies.py:260 stylized_event_cost`、`:288 select_cost_threshold`；扫描 FN:FP 成本比（1:1/5:1/10:1/20:1），画成本-阈值曲线 vs climatology 基线；配置复用 `risk.false_negative_cost/false_positive_cost`（yaml:93-94） | **A10 已覆盖"有无 OOD/数据辅助项消融"**；S5 是**决策阈值/成本比敏感性**，证明风险代理的决策可用性。补强 A10，不重复 |
| **S6** | **生成场景的尾部分位联合依赖诊断**（variogram/energy by event severity） | 「CFM 生成的场景保留了空间/爬坡联合依赖，还是只匹配边缘？极端尾部的联合结构对不对？」 | Gneiting & Raftery 2007（variogram/energy proper scores）[V4]；Wen, Pinson et al. 2024（场景评估）[V1] | `evaluate.py:311 dependence_diagnostics`、`:297 variogram_score_values`；按事件严重度（爬坡幅度分位 / 真值功率分位）分层输出 variogram/energy，对比 G1–G6；配置复用 `risk.ramp_threshold_fractions`（yaml:96） | **G1–G6 已报全量 variogram/energy**；S6 按**事件严重度分层**（尾部）。**缺口2 已覆盖"事件条件功率误差"**；S6 是"事件条件联合依赖质量"，互补。不重复 |
| **S7** | **密度修正在温湿/空气密度工况下的增益分层** | 「弱物理密度修正在什么工况下有用？跨季节（冬夏温差/密度差）稳定吗？会不会某季节反而有害？」 | Gneiting & Raftery 2007（物理中心诊断）[V4]；Zhou et al. 2024 SDWPF（多气象通道）[V6] | A1（`matrix.py:37` density off/on）按空气密度分箱 / 季节分层评估；落地全景表 §2 声明未实现的 `seasonal_rmse`（A1:40）；`data/stress.py:289-293` `equivalent_wind_speed` 密度修正路径 | **A1 已覆盖"全局 density on/off"**；S7 按密度工况分层，落地未实现的 `seasonal_rmse`。补强，不重复 |
| **S8** | **场景数与蒙特卡洛收敛性扫描**（n_scenarios sweep） | 「100 个场景够吗？CRPS/时延随场景数收敛在哪？工程部署该取多少？」 | Gneiting & Raftery 2007（MC 评分稳定性）[V4]；Godahewa et al. 2021（评估协议）[V8] | `cli/benchmark_inference.py:35 --scenarios`；评估侧 `flow.scenarios_eval`（yaml:61）扫 [10,25,50,100,200]，画 CRPS/能量分数/时延 vs n_scenarios 收敛曲线 | **A6 已扫 ODE 步数（采样质量-速度）**；S8 扫**场景数（MC 收敛）**。缺口4（OFAT S1–S7）不含 n_scenarios。不同效率轴，不重复 |
| **S9** | **OOD/fallback 触发的真实性校准**（trigger vs realized error） | 「context fallback 的 OOD 触发是真 OOD 还是误触发？触发时 realized 误差真的更大吗？Mahalanobis 分数与误差退化相关吗？」 | Gibbs & Candès 2021（漂移下自适应）[V2]；Gneiting & Raftery 2007（校准）[V4] | `risk/proxies.py:157 MahalanobisOOD`、`:189 data_quality_delta`；`evaluate.py:274 fallback_trigger_rate`；把触发/OOD 分数与 realized `rmse_degradation`（`evaluate_missing_stress.py:262`）做相关/分层 | **A8/A10 已触及 fallback/OOD**；S9 审计"触发-误差"一致性，证明 context fallback 的可信度。补强，不重复 |
| **S10** | **事件检测的提前量-误报权衡分布**（lead-time distribution vs FAR by horizon） | 「爬坡预报提前多久有用？误报率多少？按 horizon 的精度-召回曲线如何？」 | Wen, Pinson et al. 2024（事件预报）[V1]；Gneiting & Raftery 2007（事件评分）[V4] | `evaluate.py:125 lead_time_steps_mean`、`:121 event_f1`；扩展为 lead-time 分布、按 horizon 的 precision-recall / 误报率（FAR）曲线；复用 `observed_ramp_event`（`proxies.py:68`） | **`_ramp_metrics` 已有 lead_time_mean/auprc（部分）**；S10 补全**提前量分布 + FAR 权衡**。**缺口2 已覆盖事件条件功率误差**；S10 是检测提前量权衡。不重复 |

---

## 三、盘点结论

1. **整体覆盖度**：①全景表 28 项实验已搭起「物理先验→动态多图→概率生成→校准/风险→缺失/协议→基线」的证据骨架，主流概率/点/边缘指标（crps/energy/variogram/ramp_ks/brier、mae/rmse/r2/picp/pinaw/pinball、auprc/brier/stylized_cost、latency/throughput/peak_vram）在 src 已实现；但对照顶刊常规 10 类实验，**无一类达到"完整分解/对比"标准——6 类部分覆盖、4 类未设计**，最高风险空白仍是事件级证据链（`extreme_rmse/extreme_mae/extreme_picp` 三指标声明却零实现）。

2. **领域常规中最该补的 3 类**（优先级从高到低）：
   - **#4 异常/极端事件检测的条件技能表**（已被缺口2 覆盖，但务必落地——这是标题主打"extreme weather"的唯一独立证据来源，全量平均会被事件内外误差稀释，投稿被批概率最高）；
   - **#2 跨季节/风况/场址分段评估**（被缺口1 覆盖，单数据集+单时序 split 无法支撑"空间图/物理先验可迁移"claim）；
   - **#1 逐 horizon 点/概率误差曲线**（区间已逐步、点与 CRPS 却跨步聚合，补这一步成本极低但能显著强化多步预测证据）。

3. **方法专属最该补的 3 个补充实验**（在 S1–S10 中，若时间有限只做三个）：
   - **S2 共形条件覆盖审计**——直接回应"marginal ≠ conditional coverage"这一共形预测头号审稿质疑，且能顺手落地未实现的 `rolling_picp`；
   - **S1 门控权重工况分布诊断**——零新模型、只读已落盘的 `graph_weights`（`backbone.py:244`），用极低成本证明"动态门控"名副其实，与缺口3（干预）形成"有用 + 合理"双证据；
   - **S3 物理中心误差传播分解**——直接回应"两阶段会不会把物理中心偏差带进最终场景"，把弱物理先验从"结构性卖点"落成"可归因证据"。

---

## 四、引用来源列表（本文件新引用 + 核验状态）

> 核验状态：**VERIFIED-本会话** = 本次以 web_fetch 打开权威记录页（arXiv abs / 作者主页 / 出版社页）逐条确认；**VERIFIED-主文档** = 引自并行调研主文档 `docs/顶刊文献调研与代码补全规范.html`（该文档自述 26 条引用经独立第三方交叉核验，24 VERIFIED / 2 DOUBTFUL 已修正 / 0 FAIL）。

| 编号 | 文献（作者-年份-标题-出处） | DOI / URL | 核验状态 |
|---|---|---|---|
| V1 | Wen, H., Pinson, P., Gu, J., Jin, Z. (2024). *Marginalize, Rather than Impute: Probabilistic Wind Power Forecasting with Incomplete Data.*（投 INFORMS J. Data Science） | https://arxiv.org/abs/2403.03631 | **VERIFIED-本会话**（arXiv abs 页确认标题/作者/摘要） |
| V2 | Gibbs, I., Candès, E. (2021). *Adaptive Conformal Inference Under Distribution Shift.* NeurIPS 2021 | https://arxiv.org/abs/2106.00170 | **VERIFIED-本会话**（arXiv abs 页确认） |
| V3 | Islam, M.S., Hasan, A.S.M.J., Rahman, M.S., Yusuf, J. (2024). *Wind Power Prediction across Different Locations using Deep Domain Adaptive Learning.* | https://arxiv.org/abs/2405.11188 | **VERIFIED-本会话**（arXiv abs 页确认） |
| V4 | Gneiting, T., Raftery, A.E. (2007). *Strictly Proper Scoring Rules, Prediction, and Estimation.* JASA 102(477):359–378 | DOI: 10.1198/016214506000001437（作者主页 sites.stat.washington.edu/raftery 与 ASA 期刊页双重确认卷期页码） | **VERIFIED-本会话**（作者主页 + amstat 页交叉确认） |
| V5 | Gneiting, T., Raftery, A.E., Westveld, A.H., Goldman, T. (2005). *Calibrated Probabilistic Forecasting Using Ensemble Model Output Statistics and Minimum CRPS Estimation.* Monthly Weather Review 133:1098–1118 | （经 SIAM 出版物页 + 作者主页列目确认卷期页码） | **VERIFIED-本会话**（检索记录交叉确认） |
| V6 | Zhou, J. et al. (2024). *SDWPF: A Dataset for Spatial Dynamic Wind Power Forecasting over a Large Turbine Array.* Scientific Data 11:649 | https://arxiv.org/abs/2208.04360 | **VERIFIED-主文档** |
| V7 | Draxl, C., Clifton, A., Hodge, B.-M., McCaa, J. (2015). *The Wind Integration National Dataset (WIND) Toolkit.* Applied Energy 151:355–366 | DOI: 10.1016/j.apenergy.2015.03.121（OSTI 1250028） | **VERIFIED-主文档** |
| V8 | Godahewa, R.W., Bergmeir, C., Webb, G.I., Hyndman, R.J., Montero-Manso, P. (2021). *Monash Time Series Forecasting Archive.* NeurIPS 2021 D&B | https://arxiv.org/abs/2105.06643 | **VERIFIED-主文档** |
| V9 | Plumley, C. (2022). *Kelmarsh Wind Farm Data (v0.0.3).* Zenodo（当前最新 v4 为 record 16807551，2025-08；投稿前固定版本） | 概念 DOI: 10.5281/zenodo.5841833 | **VERIFIED-主文档** |
| V10 | Nie, Y., Nguyen, N.H., Sinthong, P., Kalagnanam, J. (2023). *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers (PatchTST).* ICLR 2023 | https://arxiv.org/abs/2211.14730 | **VERIFIED-主文档** |
| V11 | Liu, Y., Hu, T., Zhang, H., Wu, H., Wang, S., Ma, L., Long, M. (2024). *iTransformer: Inverted Transformers Are Effective for Time Series Forecasting.* ICLR 2024 | https://arxiv.org/abs/2310.06625 | **VERIFIED-主文档** |
| V12 | Chen, T., Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System.* KDD 2016, pp.785–794 | https://arxiv.org/abs/1603.02754 · DOI: 10.1145/2939672.2939785 | **VERIFIED-主文档** |
| V13 | Vaswani, A. et al. (2017). *Attention Is All You Need.* NeurIPS 2017 | https://arxiv.org/abs/1706.03762 | **VERIFIED-主文档** |
| V14 | Genoese, M., Slednev, V., Fichtner, W. (2016). *Analysis of drivers affecting the use of market premium for renewables in Germany.* Energy Policy 97:494–506（内含 Obersteiner et al. 2010 的 COIF / 不平衡成本分解） | DOI: 10.1016/j.enpol.2016.07.043（KIT IIP 作者单位出版列表 + RePEc + Semantic Scholar 三方交叉确认卷期页码） | **VERIFIED-本会话**（检索交叉确认：标题/作者/卷页/DOI 一致；摘要确认其研究风电预测误差的清算成本） |

**剔除记录（防幻觉）**：arXiv:1806.00142（实为引力量子引力论文 *Gravitational collapse...*，非 Graph WaveNet）、arXiv:2007.02846（实为计算机视觉 *Point-Set Anchors...*，非 AGCRN）。二者均因记忆 ID 错误在本会话核验中被剔除，未进入上表，也未在正文引用。
