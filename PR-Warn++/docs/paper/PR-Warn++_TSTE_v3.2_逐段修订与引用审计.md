# PR-Warn++ TSTE v3.2 逐段修订与引用审计

> 路径说明（目录重组后）：本文档现位于 `docs/paper/`；文中提到的旧 `3.0/`
> 已移至 `legacy/3.0/`，旧 `3.2/` 即当前目录。

## 1. 审计范围与判定规则

本审计覆盖旧版 `3.0/PR-Warn++_TSTE_叙事冻结版_v2.tex` 与新版
`3.2/PR-Warn++_TSTE_v3.2.tex` 的 Abstract、Section I--III，以及新版正文实际引用的全部文献。

每处修改按以下问题判定：

1. 相邻论断是否得到该文献题名、摘要或方法定义的直接支持；
2. 文献是否仅支持一个较弱结论，正文是否把结论写得过强；
3. 题名、作者、期刊、年份、卷期、页码/文章号和 DOI 是否一致；
4. 公式是否与 v3.2 代码中的输入、目标、张量层级和训练/推理边界一致；
5. 不能由当前证据支持的内容是否被降级为待验证假设或协议边界。

判定标签：`直接支持` 表示引用可支撑相邻论断；`限定后支持` 表示正文必须采用更窄表述；
`方法来源` 表示只用于公式或通用方法定义；`数据来源` 表示只用于数据内容和协议说明。

## 2. 旧稿到 v3.2 的逐段修订追踪

### 2.1 Abstract

| 段落 | 旧稿问题 | v3.2 修订 | 修订依据 |
|---|---|---|---|
| A-1 研究动机 | 以模块清单开头，预测对象不够集中 | 先区分点轨迹、联合场景、边际区间和 ramp 风险 | 三类不确定性对象不能用同一指标证明 |
| A-2 确定性中心 | 使用 `graph-recurrent`，与当前实现不一致 | 改为 mask-aware temporal multi-graph network | 当前代码是 temporal Conv2d 与 graph residual blocks |
| A-3 生成模块 | 容易把 VAE/OT 写成主线 | 明确 Direct CFM 直接生成最终误差场；VAE 与 OT 只作消融 | 当前 A4/A5 实现边界 |
| A-4 校准与风险 | 容易暗示场景被 conformal 整体校准 | 明确 conformal 仅对应 farm-level per-horizon marginal intervals | 当前评估代码与理论保证边界 |
| A-5 结果陈述 | 旧结构容易诱发先写结果 | 只保留服务器结果占位符 | 不虚构五种子、显著性、延迟或显存结果 |

### 2.2 Section I -- Introduction

| 段落 | 旧稿内容/问题 | v3.2 段落功能 | 关键修改 |
|---|---|---|---|
| I-1 | 泛化描述风电不确定性 | 定义联合 turbine-by-horizon 预测对象 | 用 SDWPF 的 134 台、10 min 事实落地问题 |
| I-2 | 动态图被写成主要新颖性来源 | 建立图预测已成熟的竞品背景 | 明确动态图本身不足以构成贡献 |
| I-3 | 概率预测文献混在一起 | 区分区间、分布适配与联合生成 | 提出“确定性中心之后究竟生成什么”的具体缺口 |
| I-4 | NWP/ERA5 可用性边界不够醒目 | 将 retrospective ERA5 与 issued forecast 分开 | 防止 oracle weather 被写成在线输入 |
| I-5 | 缺口数量多且重复 | 收敛成最终误差场、三类可靠性对象、信息可用性三项缺口 | 与 3+1 贡献一一对应 |
| I-6 | 两种残差在不同位置出现 | 并列定义 `Y=P_pc+R` 与 `Y=Y_det+E` | 明确 `R` 是确定性残差，`E` 是最终预测误差 |
| I-7 | 贡献点按模块堆叠 | 改为三项方法贡献加一项协议贡献 | 不把 power curve、graph、CFM、conformal 各自虚构为独立创新 |

### 2.3 Section II -- Related Work

| 段落 | 旧稿问题 | v3.2 修订 | 审计结论 |
|---|---|---|---|
| II-A1 | 静态图、动态图和缺失处理混述 | 按 static/adaptive/time-varying/missing-aware 组织 | 引用均与相邻方法类别对应 |
| II-A2 | “direction-aware graph”尺度含混 | 区分 wind-farm-cluster 的风速/风向相关图与 turbine-scale wake-directed graph | 修复 `wang2025clustergraph` 与 `hou2026wake` 被并列概括过强的问题 |
| II-B1 | 边际概率方法与联合场景方法混用 | 单列 prediction interval、meta、quantile、conformal | 不再用边际文献支持联合依赖结论 |
| II-B2 | 生成式工作的目标差异不清 | 区分 wind speed/wind power、diffusion/flow matching | 把本文差异限定为生成目标 `E`，而不是“首次使用生成模型” |
| II-C1 | conformal 保证表述偏泛 | 明确 exchangeability/chronological protocol 下的 marginal coverage | 不声称 simultaneous 或任意条件覆盖 |
| II-C2 | 风险容易越权为电网安全 | 仅保留 wind-side ramp/deviation/tail proxies | 无网络、潮流、备用模型时不称 grid-security probability |
| II-C3 | 小节结尾缺少碰撞结论 | 用 claim-separated design 连接 Section III | 文献线与方法线闭合 |

### 2.4 Section III -- Problem Formulation and Proposed Framework

| 段落/公式块 | 旧稿问题 | v3.2 修订 | 与代码/协议的对应 |
|---|---|---|---|
| III-1 任务定义 | Problem Formulation 与 Method 分成两个主体 Section | 合并为一个完整 Section III | 满足“前三章形成闭合方法定义”的稿件结构 |
| III-A 信息集 | 在线可见变量未完全形式化 | 定义 forecast-origin information set `I_t` | future truth 与 retrospective ERA5 不在主路径 |
| III-A 三协议 | history/oracle/forecast 容易混写 | 分为 history-only、oracle-weather、issue-time weather | A11 必须满足 `issue_time <= forecast_origin` |
| III-B 刚性网格 | 异常值、unknown、missing 容易被 drop | 保留 rigid grid，分开 `X_fill`、`M`、`Delta t`、reason sidecar | 主模型输入与审计字段分离 |
| III-C 密度修正 | 压力、温度单位与可用时间未限定 | 只在起点可用且单位已审计时计算 `rho` 与 `v_eq` | Pa/K 单位仍需服务器数据审计 |
| III-C power curve | 旧稿可能暗含未来风速 | history-only 下未来锚点保持为最后有效风速对应曲线值 | 不发生 future-weather leakage |
| III-C 残差目标 | 物理中心与最终误差混淆 | 明确定义 `R=Y-P_pc` | deterministic backbone 的监督目标 |
| III-C 图结构 | 旧稿以 graph-recurrent 描述 | 定义 `A_geo/A_corr/A_dir/A_adp` 与 gated fusion | 与当前图融合实现一致 |
| III-C 骨干 | recurrent 术语与代码不符 | 改为 temporal convolutions + residual graph blocks | 与 `backbone.py` 对齐 |
| III-D CFM 条件 | 旧稿 conditioner 边界含混 | 条件限定为冻结的 `H_t` 与 fused adjacency | 不把 reason code 或未来标签送入生成器 |
| III-D 生成目标 | VAE latent 与直接误差场混写 | 主线直接建模 `E=Y-Y_det` | VAE-CFM 降为 A4 |
| III-D pairing | 独立高斯配对可能被误称 OT | 主线称 standard independent-pair CFM | Hungarian minibatch assignment 只在 A5 称 OT |
| III-D 推理 | 场景重建对象不清 | `Y^(m)=clip(Y_det+E^(m))` | 保持确定性中心不被生成器重复学习 |
| III-E conformal | 容易把 turbine-level 和 farm-level 混用 | 主公式改为 farm-level per-horizon split conformal | 对应当前 `[sample,horizon]` 校准实现 |
| III-E event risk | 首个 ramp 差分缺少前缀 | 用预测起点观测农场功率作为 `h=0` | 与事件评估代码对齐 |
| III-E A10 | 风险组合可能被误写为学习分类器 | 仅允许冻结、预声明权重修改 event logit | 不用合成 Level 1--4 标签训练 |
| III-F 六阶段 | 训练与 Test 边界散落 | 固结为 data audit -> train-only objects -> deterministic -> frozen error -> generator -> Calib | Test 只在全部对象冻结后打开 |

## 3. 全部引用的语义与元数据审计

| 键 | 正文用途 | 语义判定 | 元数据/处理 |
|---|---|---|---|
| `zhou2024sdwpf` | 134 台、10 min、ERA5 角色、主数据集 | 数据来源，直接支持 | DOI、卷、文章号已核对 |
| `lee2020pcwg` | power curve 与密度修正实践 | 限定后支持 | 只支撑标准实践与密度影响，不声称其提出本文公式；补 issue 1 |
| `che2018grud` | mask 与 time-since-observation 有预测信息 | 方法来源，直接支持 | DOI、卷、文章号已核对；补 issue 1 |
| `tang2024timeaware` | time-aware graph probabilistic wind forecasting | 直接支持 | IEEE 卷期页码与 DOI 一致 |
| `yang2024gcn` | adaptive graph perception | 直接支持 | IEEE 卷期页码与 DOI 一致 |
| `liang2025wpformer` | graph Transformer 与 auto-correlation | 直接支持 | IEEE 卷期页码与 DOI 一致 |
| `liu2025graphcontrast` | graph contrastive robustness | 直接支持 | 摘要明确处理 noise/missing robustness |
| `li2025dynamic` | dynamic matching/online modeling | 直接支持 | IEEE 卷期页码与 DOI 一致 |
| `chen2024interval` | multi-objective prediction intervals | 直接支持 | IEEE 卷期页码与 DOI 一致 |
| `meng2024meta` | meta-learning probabilistic adaptation | 直接支持 | IEEE 卷期页码与 DOI 一致 |
| `meng2026missing` | missing-data-tolerant probabilistic forecasting | 直接支持 | IEEE 17(2), 1202--1213 已核对 |
| `zhao2025nwp` | overlapping historical NWP products | 直接支持 | IEEE 卷期页码与 DOI 一致 |
| `zhang2026transitional` | weather transition/error propagation | 直接支持 | IEEE 17(1), 295--306 已核对 |
| `ko2026ramp` | direct wind-power/ramp-rate forecasting | 直接支持 | IEEE 17(1), 338--350 已核对 |
| `liu2026diffusion` | Transformer diffusion for wind-speed distributions | 直接支持 | 不再把它概括为 wind-power generation |
| `yang2025risk` | application-centered/risk-scenario framing | 限定后支持 | 仅用于相关工作，不移植其风险标签；页码已修正 |
| `gao2025tgdpF` | power-curve distribution knowledge/noise robustness | 直接支持 | 摘要明确 TgDPF、power curve、noise robustness |
| `liu2024physicsrl` | physics-informed probabilistic extreme-event learning | 直接支持 | 摘要明确 cold-wave/extreme-event setting |
| `wang2025clustergraph` | 风速/风向相关的风场集群图 | 限定后支持 | 正文已从泛称“direction graph”改成 cluster-scale correlation graph |
| `chen2025noncrossing` | non-crossing spatio-temporal quantiles | 直接支持 | Applied Energy 卷、文章号与 DOI 一致 |
| `jonkers2024conformal` | conformalized regional probabilistic forecasting | 直接支持 | 不用它证明联合场景校准 |
| `gao2026diffusion` | conditional diffusion wind-power forecasting | 直接支持 | Part B 从 `number=B` 改为 note，避免错误期号 |
| `ma2026stgld` | graph latent diffusion for regional wind speed | 直接支持 | 不概括为 turbine power generation |
| `cao2026joint` | wide-area meteorological representation | 直接支持 | Applied Energy 卷、文章号与 DOI 一致 |
| `hou2026wake` | wake-directed turbine graph | 直接支持 | 摘要明确 physical cone directed graph；Part B 表达已修正 |
| `wu2019graphwavenet` | node-embedding adaptive dependency matrix | 方法来源，直接支持 | IJCAI DOI 与页码一致 |
| `bai2020agcrn` | data-adaptive graph generation | 方法来源，直接支持 | 增加 NeurIPS 官方论文页 URL |
| `lipman2023fm` | flow matching vector-field regression | 方法来源，直接支持 | 增加 ICLR/OpenReview 官方论文页 URL |
| `kollovieh2025tsflow` | CFM 用于概率时间序列预测 | 方法来源，直接支持 | 增加官方 URL，修正作者重音符号 |
| `romano2019cqr` | asymmetric interval nonconformity score | 方法来源，限定后支持 | 正文已明确是用该 construction 包装 scenario quantiles，不把本文方法称为训练式 CQR |
| `gibbs2021aci` | chronological adaptive miscoverage update | 方法来源，直接支持 | 增加 NeurIPS 官方论文页 URL，保留 long-run coverage 边界 |
| `gneiting2007proper` | proper scoring-rule foundation/Energy Score | 方法来源，直接支持 | DOI、卷期页码一致 |
| `scheuerer2015variogram` | Variogram Score | 方法来源，直接支持 | DOI、卷期页码一致 |
| `rockafellar2002cvar` | CVaR 定义 | 方法来源，直接支持 | DOI、卷期页码一致 |
| `draxl2015windtoolkit` | WIND Toolkit 数据资源 | 数据来源，直接支持 | 不再单独用它声称 issue-time 可审计 |
| `hodge2016windforecast` | WIND retrospective reforecast products | 数据来源，直接支持 | 新增 NREL 官方报告；A11 仍要求服务器侧时间戳审计 |
| `plumley2025kelmarsh` | Kelmarsh 6 台机组、10 min SCADA 与静态参数 | 数据来源，直接支持 | Zenodo v4、作者、DOI、URL 已核对 |

## 4. 本轮已落入正文的实质修正

1. 将 `wang2025clustergraph` 和 `hou2026wake` 的空间尺度拆开陈述，避免一篇风场集群预测论文被误写成 turbine wake solver。
2. 将 CQR 的作用限定为 asymmetric interval-score construction，避免把 scenario-quantile wrapper 错写成训练式 CQR 模型。
3. 将 WIND Toolkit 的表述从“已具备可审计 issue time”改成“包含 retrospective reforecast products，但 A11 必须先审计 issue/valid timestamps”。
4. 为没有 DOI 的 AGCRN、Flow Matching、TSFlow、CQR 和 ACI 补入官方论文页 URL。
5. 增加 NREL reforecast 报告引用；补齐三个期号；修正 Applied Energy `Part B` 的 BibTeX 表达和作者重音符号。

## 5. 审计后的剩余边界

- 文献审计确认“引用能够支持正文所写的有限论断”，不等于复现实验结论。
- 2026 年文章的具体数值结果未被移植进 PR-Warn++，因此无需对其表格数字作二次复算。
- A11 是否能作为部署型协议，最终取决于服务器侧实际下载文件中是否存在明确、无泄漏的 issue/valid timestamp。
- 所有性能、显著性、覆盖率、延迟和显存结论仍只能来自冻结服务器实验。
