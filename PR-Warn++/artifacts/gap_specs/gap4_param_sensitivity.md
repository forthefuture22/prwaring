## 缺口 4：参数敏感性 one-factor-at-a-time 协议（优先级 P0）

> 对应约束：C3-7（参数敏感性 one-factor-at-a-time 完全缺失，需 history / Top-K / distance scale / bandwidth / lag / bin width / gate width 七组；只在验证集评估；报告各参数最大退化与是否处于搜索边界）。
> 本规范只做「文献依据 + 可落地实现设计」，不运行任何训练/实验，不修改现有代码。

### 1. 缺口定位

**现状（已 Read 核实，附行号）：**

- 配置文件 `configs/sdwpf_v3_2.yaml` 中全部可调参数集中在 `data / physics / graphs / model` 段：
  - `:9` `data.history_steps: 24`；`:10` `data.forecast_steps: 6`。
  - `:30` `physics.power_curve_bins: 50`；`:31` `physics.minimum_bin_count: 20`。
  - `:35` `graphs.geographic_k: 8`；`:36` `graphs.correlation_k: 12`；`:38` `graphs.direction_distance_scale: 1000.0`；`:39` `graphs.direction_sigma_degrees: 30.0`；`:40` `graphs.direction_sector_degrees: 90.0`。
  - `:48` `model.hidden_dim: 64`；`:49` `model.temporal_kernel: 3`；`:50` `model.dropout: 0.1`。
- 这些参数的消费处：
  - `history_steps` → `src/prwarn/data/window.py:15`（`make_windows(..., history_steps: int)`），`:28` `origins = np.arange(history_steps - 1, final_origin, stride)`。
  - `correlation_k / geographic_k` → `src/prwarn/graphs/builders.py:14-22` `_top_k`；`:54` `correlation_graph(..., k: int = 12, ...)`；`:25-47` `geographic_graph(..., k: int = 8, distance_scale=None)`。
  - `direction_distance_scale / direction_sigma_degrees` → `src/prwarn/graphs/builders.py:93-137` `direction_graph(..., distance_scale, sigma_degrees=30.0, sector_degrees=90.0)`，其中 `:128` `adjacency = np.exp(-distance / distance_scale)`、`:129` `* np.exp(-0.5 * np.square(angle_error / sigma_degrees))`。
  - `temporal_kernel` → `src/prwarn/models/backbone.py:139-143`（`Conv2d(kernel_size=(1, temporal_kernel))`），且 `:107` 强制 `temporal_kernel` 必须为奇数。
  - `power_curve_bins` → `src/prwarn/physics/power_curve.py:41` `n_bins: int = 50`，`:68` `edges = np.linspace(lower, upper, self.n_bins + 1)`（bin 宽度 ∝ 风速量程 / n_bins）。
  - gate 宽度 → `src/prwarn/models/backbone.py:147-151` `self.gate = nn.Sequential(Linear(hidden_dim, hidden_dim), GELU, Linear(hidden_dim, ...))`，当前 gate 隐藏宽度与全局 `hidden_dim` 共享（=64），无独立键。
- 实验矩阵机制 `src/prwarn/experiments/matrix.py:15-22` `ExperimentSpec.variants = ((variant, overrides: dict), ...)` 已支持「覆盖键值对」；`:87-92` 的 A6（`flow.steps` 4/8/16/32）就是一个现成的「逐参数多水平」写法范式。`build_experiment_matrix`（`:231-262`）把 variants × seeds 展开为 run contract。

**缺口是什么：**

`src` 内**没有任何 one-factor-at-a-time（OFAT）敏感性实验**。审稿人 R3.20 要求：逐参数扰动、只在验证集评估、报告各参数最大退化、并判断最优值是否落在搜索边界（若落在边界说明搜索区间太窄）。当前既没有敏感性实验规格，也没有「基线值 → 单参数多水平 → 验证集指标 → 最大退化/边界判定」的报告产物。

**为什么必须补（P0 理由）：**

- C3-7 是 P0 实验设计硬缺口（核查报告结论清单第 4 条）。没有敏感性分析，读者无法判断本文报告的精度对这些关键超参（回看窗口、图稀疏度、方向图尺度、功率曲线分箱、模型宽度）是否稳健——这在 TSTE/Applied Energy 评审中是常规被质疑点。
- OFAT 不是为了调参，而是为了**稳健性声明**：证明报告结果不是「恰好卡在某个幸运超参点」。C3-7 同时要求「是否处于搜索边界」的判定，这正是 OFAT 报告区别于普通调参表的关键。

### 2. 文献依据表

| 文献（作者-年份） | 出处/年份 | DOI/arXiv | 核验状态 | 关键做法 | 可借鉴的协议要素 |
|---|---|---|---|---|---|
| Czitrom (1999) | The American Statistician, 53(2): 126–131, 1999 | DOI: 10.1080/00031305.1999.10474445（JSTOR 镜像 DOI: 10.2307/2685731） | VERIFIED（Semantic Scholar 作者 Veronica A. Czitrom / 1999-05-01 / The American Statistician；多源引用交叉确认卷期 53(2):126–131；T&F 与 JSTOR 两个 DOI 均被第三方文献引用页核对） | 以三个工程实例讲 OFAT 与设计实验（DOE）对比：OFAT 每次只动一个因子、其余固定；用**中心点（center point）**检查响应曲面曲率；明确指出 OFAT 无法估计因子间交互 | ① OFAT 的标准操作：固定其余因子、一次只变一个；② 必须有**基线中心点**（本文=默认配置）；③ 诚实声明 OFAT 不捕捉交互作用——这是论文里必须写的边界 |
| Saltelli (2002) | Computer Physics Communications, 145(2): 280–297, 2002 | DOI: 10.1016/S0010-4655(02)00280-1 | VERIFIED（HAL 存档页 hal-03679350 逐字核对标题/期刊/卷期 145(2):280–297/DOI；uqtestfuns、openturns、SALib 等多套学术软件参考文献交叉确认） | 全局敏感性分析：把输出方差分解到各输入因子（一阶/总阶 Sobol 指数）；系统讨论因子相关时的处理 | ① 作为 OFAT 的**对照方法学**引用：本文用 OFAT 做筛选/稳健性报告，而非全局 SA；② 报告时需区分「单因子主效应」与「交互作用」，呼应 C3-7 的诚实边界 |

> 说明：本缺口只选 2 篇。Czitrom (1999) 是 OFAT 协议本身的经典出处，Saltelli (2002) 是全局敏感性分析的方法学锚点——二者搭配正好支撑「用 OFAT 做稳健性筛查、并声明其局限」的写作立场。未强行凑 TSTE/Applied Energy 的逐参数扰动表实例——该类具体表格难以逐字核验标题+DOI，按「宁缺毋滥」不纳入。

### 3. 标准做法要点

从文献提炼的可执行协议要素（每条注明来源）：

1. **一次只动一个因子**：从基线配置出发，每次只改变一个参数的取值，其余所有参数保持默认；绝不同时改两个（来源：Czitrom 1999）。
2. **以默认配置为基线中心点**：把 `sdwpf_v3_2.yaml` 的默认值作为 OFAT 的中心点（center point），每个参数的扰动水平都以该中心点为参照上下展开（来源：Czitrom 1999 对 center point 的使用）。
3. **每个参数取 3–5 个水平，包含基线水平**：水平要跨越有物理意义的区间，而非仅 ±10%；基线水平必须落在水平序列中，以便直接读出相对基线的退化（来源：Czitrom 1999；与 matrix.py A6 的 4/8/16/32 多水平写法一致）。
4. **只在验证集评估，绝不触碰测试集**：敏感性的指标一律在 validation split 上计算；测试集只用于最终主结果（对齐 C3-7「只在验证集评估」与 C3-5「超参只在验证集选择」）。
5. **报告每个参数的最大退化**：对每个参数，报告其相对基线的最大绝对指标变化（RMSE/CRPS 的绝对量，对齐 C4-3 用绝对量而非百分比），并报告最优（最小）指标出现在哪个水平（来源：Czitrom 1999 对效应估计；C3-7 要求最大退化）。
6. **边界判定**：若某参数的最优水平出现在扰动区间的端点，标注「at boundary」——说明当前搜索区间可能太窄、最优值在区间外，需扩区间重测（来源：C3-7 明确要求「是否处于搜索边界」）。
7. **多 seed 报告 mean ± SD**：每个水平在多 seed（≥10，对齐 C0-2）下训练并报告 mean ± SD，避免单 seed 波动被误读为参数敏感性（来源：C0-2；与 Wiegreffe & Pinter 2019 跨种子方差校准的精神一致）。
8. **诚实声明 OFAT 局限**：在论文中写明 OFAT 无法捕捉因子间交互作用（来源：Czitrom 1999；Saltelli 2002 的全局 SA 作为补充方法学）。本文用 OFAT 做稳健性筛查，而非全局敏感性排序。

### 4. 实现规范（映射到本代码库）

> 下列均为**设计规范**，不在本任务中实施。七组参数逐组映射到真实配置键；无直接对应键处明确给出新增键名与默认值。

**七组参数与配置键映射：**

| # | 报告分组 | 映射配置键 | 基线值 | 落点文件/函数 | 建议扰动水平 | 备注 |
|---|---|---|---|---|---|---|
| 1 | history | `data.history_steps` | 24 | window.py:15 / :28；config:9 | 12, 18, **24**, 36, 48 | 单位为 10 min 步（≈2h/3h/4h/6h/8h）；改 window 长度需重训 |
| 2 | Top-K | `graphs.correlation_k` | 12 | builders.py:54 / :89 `_top_k`；config:36 | 4, 8, **12**, 16, 24 | 地理图 `graphs.geographic_k=8`（config:35）为同类键，可作姊妹组；k 须 ≤ 节点数，`_top_k` 自动 clamp |
| 3 | distance scale | `graphs.direction_distance_scale` | 1000.0 | builders.py:97 / :128；config:38 | 500, **1000**, 2000, 4000 | 方向图距离衰减尺度（米）；geographic 图距离尺度取中位距离（builders.py:40），不扰动 |
| 4 | bandwidth | `graphs.direction_sigma_degrees` | 30.0 | builders.py:98 / :129；config:39 | 10, 20, **30**, 45, 60 | 方向图高斯角带宽（度）；与 distance scale 是两个独立核，故单列 |
| 5 | lag | `model.temporal_kernel` | 3 | backbone.py:139 / :107；config:49 | 1, **3**, 5, 7 | 时间卷积核 = 局部时间感受野/lag；**必须为奇数**（backbone.py:107 已强制），不可取偶数 |
| 6 | bin width | `physics.power_curve_bins` | 50 | power_curve.py:41 / :68；config:30 | 20, 35, **50**, 75, 100 | bin 宽度 ∝ 风速量程/n_bins，扰动 n_bins 即扰动 bin width |
| 7 | gate width | 建议新增 `model.gate_hidden_dim` | 64（默认 = `model.hidden_dim`） | backbone.py:147-151 `self.gate`；config:48 | 32, 48, **64**, 96, 128 | 当前 gate 隐藏宽度与全局 `hidden_dim` 耦合；为独立扰动 gate 宽度，建议新增解耦键（见下表） |

**实现改动点：**

| 改动点 | 落点文件/函数/配置键 | 新增参数与默认值 | 协议要点 | 验收方式 |
|---|---|---|---|---|
| 解耦 gate 宽度 | `configs/sdwpf_v3_2.yaml` 新增 `model.gate_hidden_dim` | `model.gate_hidden_dim: null`（null 时回退 = `model.hidden_dim`） | 使 gate MLP 隐藏宽度可独立于全局 hidden_dim 扰动；不传则行为与现状一致 | 传 `gate_hidden_dim=32` 时 gate 中间层为 32，其余主干仍为 64 |
| 模型构造消费新键 | `DynamicMultiGraphResidualForecaster.__init__`（backbone.py:87-105）新增 `gate_hidden_dim` 形参；`:147` `self.gate = nn.Sequential(Linear(hidden_dim, gate_hidden_dim), GELU, Linear(gate_hidden_dim, n_static_graphs+2))` | `gate_hidden_dim: int \| None = None` | 仅影响 gate MLP 中间层宽度，不影响 temporal conv / graph blocks / forecast head | 参数量随 gate_hidden_dim 变化，但主干参数不变 |
| 新增敏感性实验规格 | `src/prwarn/experiments/matrix.py` 新增 group `sensitivity` 的 `ExperimentSpec`（建议 S1–S7，每组一个 spec） | variants 复用 `(variant, overrides)` 机制，逐参数生成水平 | 用与 A6（matrix.py:87-92）相同的 `tuple((f"{key}_{v}", {key: v}) for v in levels)` 写法；每组一次只覆盖一个键 | `build_experiment_matrix()` 展开出 S1–S7 的 run contract，每 run 只含一个覆盖键 |
| 验证集评估约定 | 敏感性 group 的 run contract 标注 `eval_split: val`（建议作为 ExperimentSpec 新字段或 group 约定） | `eval_split: str = "test"`（默认；sensitivity group 强制 `val`） | 敏感性指标一律在 validation split 计算，产物落盘时标注 split | 敏感性产物中每条记录 `split == "val"`，无 test 指标混入 |
| 敏感性报告产物 | 建议新增 `src/prwarn/cli/aggregate_sensitivity.py`（或复用 `aggregate_seed_runs.py` 扩展） | 无新配置键 | 输入 S1–S7 的 run 结果；输出表：行=参数×水平，列=RMSE/CRPS(mean±SD)；追加「相对基线最大退化」与「最优是否在边界」两列 | 报告表含 `max_degradation` 与 `at_boundary` 字段；每参数一行汇总 |

**新增配置键汇总（推荐键名与默认值）：**

- `model.gate_hidden_dim: null`（null → 回退 `model.hidden_dim`）。
- 其余六组均直接复用现有键（`data.history_steps` / `graphs.correlation_k` / `graphs.direction_distance_scale` / `graphs.direction_sigma_degrees` / `model.temporal_kernel` / `physics.power_curve_bins`），无需新增。

**OFAT 报告表模板（每组参数一张，建议字段）：**

| 参数 | 水平 | 覆盖键值 | Val RMSE (mean±SD) | Val CRPS (mean±SD) | 相对基线退化 |
|---|---|---|---|---|---|
| history_steps | 12 / 18 / **24** / 36 / 48 | data.history_steps=… | … | … | … |
| （每参数 5 行，基线行加粗） | | | | | |

汇总表追加列：`max_abs_degradation`（该参数跨水平最大绝对退化）、`best_level`、`at_boundary`（布尔）。

### 5. 验收自检清单

- [ ] `matrix.py` 出现 group `sensitivity` 的 S1–S7 规格，每组 variants 只覆盖一个配置键（一次只动一个因子）。
- [ ] 七组参数全部映射到真实键：history→`data.history_steps`；Top-K→`graphs.correlation_k`；distance scale→`graphs.direction_distance_scale`；bandwidth→`graphs.direction_sigma_degrees`；lag→`model.temporal_kernel`；bin width→`physics.power_curve_bins`；gate width→新增 `model.gate_hidden_dim`。
- [ ] 每组扰动水平包含基线水平，且 `model.temporal_kernel` 水平全为奇数（1/3/5/7）。
- [ ] 敏感性 run 的评估 split 标注为 `val`，产物中无 test 指标混入。
- [ ] `build_experiment_matrix()` 能展开 S1–S7 的 run contract，`experiment_id` 稳定、`evidence_status=planned_not_run`。
- [ ] 报告产物含每参数 `max_abs_degradation`（绝对量，非百分比）与 `at_boundary` 判定列。
- [ ] 每个水平在多 seed（≥10）下报告 mean ± SD。
- [ ] 论文敏感性节写明 OFAT 不捕捉因子间交互作用（引用 Czitrom 1999 / Saltelli 2002），并说明本文用 OFAT 做稳健性筛查而非全局排序。
- [ ] `model.gate_hidden_dim` 不传时回退 `model.hidden_dim`，回归不破坏 A0–A12。

### 6. 引用来源列表

1. Veronica Czitrom. One-Factor-at-a-Time Versus Designed Experiments. The American Statistician, 53(2): 126–131, 1999. DOI: 10.1080/00031305.1999.10474445. URL: https://doi.org/10.1080/00031305.1999.10474445 （JSTOR 镜像: https://doi.org/10.2307/2685731 ）
2. Andrea Saltelli. Making best use of model evaluations to compute sensitivity indices. Computer Physics Communications, 145(2): 280–297, 2002. DOI: 10.1016/S0010-4655(02)00280-1. URL: https://doi.org/10.1016/S0010-4655(02)00280-1 （HAL 存档: https://hal.science/hal-03679350 ）
