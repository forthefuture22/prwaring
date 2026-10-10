## 缺口 5：鲁棒性协议增强（加性噪声 / 特征级缺失 / 关键气象通道缺失）（优先级 P1）

> 对应约束：`docs/约束符合性核查报告.md` **C3-4**（鲁棒性协议，状态 △ 部分满足）；`docs/投稿约束清单.md` **C3-4**（扰动按训练统计量缩放、覆盖随机点/时间块/特征级/关键气象通道、派生量重算、报告绝对退化）。
> 本文档只做调研与可落地实现规范设计，不含任何训练/实验执行。

### 1. 缺口定位

**现状（已 Read 核实，行号对应当前代码）：**

| 能力 | 落点文件 / 函数 / 行号 | 现状 |
|---|---|---|
| 缺失模式选择 | `src/prwarn/data/stress.py` | 已有 `mcar_missingness`(:56) / `block_missingness`(:72) / `spatial_outage_missingness`(:99) / `extreme_conditioned_missingness`(:125) 四种 |
| 候选掩码过滤 | `stress.py:_candidates(mask, eligible)`(:45) | **已支持 `eligible` 广播掩码**，可把缺失限制在指定特征/节点子集；但 CLI 从未透出 |
| 施加缺失 + 重算时间 | `stress.py:apply_missingness_stress`(:159) | 已实现 constant/forward_fill 填充、`delta_t` 沿历史轴重算；输出 `MissingnessStressResult` |
| 派生量重算 | `stress.py:rebuild_issue_time_derived`(:241) | 已从受扰历史末时刻重算 `p_pc`（含空气密度、等效风速）、`current_y`、`wind_from` |
| CLI 组装 | `src/prwarn/cli/evaluate_missing_stress.py:_stress_selections`(:61) | 依次注册 clean/mcar/block/spatial/extreme_conditioned 五种；**调用四种缺失函数时全部不传 `eligible`** |
| 绝对退化 | `evaluate_missing_stress.py:262` | `rmse_degradation = rmse - clean_by_run[label].rmse`（绝对量，单位 kW），已符合 C4-3 |
| 特征列 | `configs/sdwpf_v3_2.yaml:18` | `feature_columns` 12 列：Wspd, Wdir, Ndir, Pab1, Pab2, Pab3, Prtv, Patv, Etmp, Itmp, Sp, T2m；物理通道 `wind_speed=Wspd`、`pressure=Sp`、`temperature=T2m`、风向 `Wdir/Ndir`（:42-43） |

**缺口是什么（C3-4 缺的三件事）：**

1. **加性 Gaussian 噪声鲁棒性测试缺失**：`stress.py` 只有"置掩码+填充"类扰动，没有对观测值叠加零均值噪声的压力算子；更没有"噪声幅度按训练集逐特征标准差缩放"的协议。
2. **特征级缺失未暴露**：`_candidates` 的 `eligible` 参数已具备把缺失限制在整列特征的能力，但 `_stress_selections` 未把"丢弃某几个特征通道"暴露成可运行的压力档位。
3. **关键气象通道缺失无专门档位**：围绕 Wspd/Wdir/Sp/T2m（密度修正、等效风速、风向、功率曲线依赖的输入）没有"专门扰动这些通道"的命名压力实验；审稿人无法判断模型对关键气象传感器失效的敏感度。

### 2. 文献依据表

> 核验口径：标题/作者/期刊/年份/DOI 经至少两个独立来源交叉一致方记 VERIFIED；具体方法细节只写可从摘要/元数据确认的部分，不补写。

| 文献（作者-年份） | 出处/年份 | DOI/arXiv | 核验状态 | 关键做法（可确认部分） | 可借鉴的协议要素 |
|---|---|---|---|---|---|
| Wen (2024) | *Energy*, 300, 131544, 2024 | 10.1016/j.energy.2024.131544 | VERIFIED（RePEc 条目 v300y2024ics0360544224013173；Google Scholar；Semantic Scholar 三方一致） | 面向风电概率预测，显式处理传感器失效/网络中断导致的缺失值，覆盖 MCAR/MAR/MNAR 三种缺失机制，提出无需预处理/重训练即可适配缺失模式的方法；以 CRPS 评估 | ①缺失压力必须区分"随机点缺失"与"特征级/传感器级缺失"；②把"缺失模式"当作与噪声并列的一类输入扰动来评估，而非只做插补预处理；③概率预测在缺失下用 CRPS/RMSE 退化度量 |
| Zhao, Zhao, Chen, Liao, Pan, Ye (2026) | *IEEE Transactions on Sustainable Energy*, 17(1), 16–29, 2026/2025 | 10.1109/TSTE.2025.3597967 | VERIFIED（ORCID 作者页；Google Scholar；Exaly 著录 17:16-29；R Discovery 四方一致） | 跨 30 个风场的可解释风电预测，摘要明确报告在"噪声与缺失数据实验"下方法鲁棒性更强，作为 TSTE 级常规评估环节 | ①"噪声 + 缺失"双扰动是 TSTE 风电预测论文的标准鲁棒性评估环节，不是可选项；②鲁棒性结论须跨多个风场/多组样本复现，不能只在单一切片上做；③退化以相对干净参考的增量报告 |

> 说明（宁缺毋滥）：本缺口只纳入上述 2 篇可逐字核验的顶刊/顶会论文。检索中另见若干候选（Scientific Reports 的 Physics-Constrained Transformer、arXiv:2602.15961 R²Energy 基准、arXiv:2503.20410 投稿 TSG 未正式录用稿、MDPI Energies 噪声敏感性），因期刊不在任务指定顶刊清单内、或尚处投稿/预印本状态、或作者-标题-年份-卷期不能与正式出版页对齐，按"灰区不使用"原则**未纳入**，仅在此如实声明。

### 3. 标准做法要点（从文献与约束提炼，逐条编号并注明来源）

1. **扰动分两类并行评估**：缺失类（置掩码）与噪声类（加性扰动）是并列的鲁棒性维度，二者都要进压力协议，不能只做缺失。〔Zhao 2026 TSTE；C3-4〕
2. **缺失覆盖四个机制维度**：随机点（MCAR）、连续时间块（block）、空间/风机联合（spatial outage）、特征级（sensor/feature drop）。特征级缺失即"整列特征不可用"，对应真实传感器失效，必须单独成档。〔Wen 2024 Energy；`stress.py` 现有四模式补 feature 维度〕
3. **噪声幅度按训练统计量缩放，且逐特征缩放**：输入已在预处理阶段按训练集均值/标准差标准化（`preprocess_sdwpf.py` 仅标准化输入、target 保留原始单位，见 C4-1）。因此在标准化空间叠加 `N(0, k²)` 等价于在原始空间叠加 `k·σ_train[d]` 的噪声。取 `k ∈ {0.10, 0.25, 0.50, 1.00}`（即 0.10–1.00× 训练 SD，与 C3-4/R2.6 区间一致），k 无量纲、对量纲不同的 Wspd(m/s)/Sp(hPa)/T2m(℃)/Patv(kW) 可直接比较。〔C3-4；标准化管线事实〕
4. **噪声只加在观测到的条目上**：对 `mask>0` 的历史观测叠加噪声，对原本已缺失（`mask==0`）的位置不再加噪，避免把"缺失"与"噪声"两种信号混叠。〔与 `apply_missingness_stress` 只在 observed 上操作的约定一致，:190-191〕
5. **派生量重算、不独立扰动**：密度修正、等效风速、功率曲线 `p_pc`、`current_y`、风向 `wind_from` 必须在受扰历史上经 `rebuild_issue_time_derived` 重算，禁止对派生量单独加噪/置缺。代码已具备该入口，只需把噪声档也接进去。〔C3-4；`rebuild_issue_time_derived`(:241)〕
6. **报告绝对退化而非百分比**：每个压力档相对各模型自身干净参考输出 `rmse_degradation`/`mae_degradation`（kW 绝对量），小负波动按数值波动处理、不宣称改进。〔C4-3；`evaluate_missing_stress.py:262` 已实现，新增档沿用〕

### 4. 实现规范（映射到本代码库）

> 原则：复用现有 `apply_missingness_stress` / `rebuild_issue_time_derived` / `_candidates`，不另起通用新模块；新增只补"噪声算子"与"eligible 透传/特征通道预设"两块。

| 改动点 | 落点文件/函数/配置键 | 新增参数与默认值 | 协议要点 | 验收方式 |
|---|---|---|---|---|
| ① 新增加性 Gaussian 噪声算子 | `src/prwarn/data/stress.py` 新增 `additive_gaussian_noise(x, mask, train_std, *, level, seed, channel_index=None) -> tuple[np.ndarray, np.ndarray, dict]` | `level: float`（由 CLI 传入 k）；`channel_index: slice\|None=None`（None=全部观测特征）；内部 `rng = np.random.default_rng(seed)` | 在标准化空间对 `mask>0` 条目加 `ε ~ N(0, level²)`；`train_std` 来自 `metadata["feature_scaler"]["std"]`（已在 `:265`/`:52` 读取）；噪声不改变 mask 与 delta_t；返回受扰 x 与 metadata（含 `noise_level=level`、`noise_channels`） | 单测：level=0 时输出与输入逐元素相等；level=1 时观测通道样本方差近似为 1+1=2 |
| ② 噪声档接入压力组装 | `evaluate_missing_stress.py:_stress_selections`(:61) 后新增 `_noise_selections` 分支；`_build_stressed_split`(:114) 扩展接收"噪声扰动"而非仅 selected 掩码 | CLI 新增 `--noise-levels` type=float nargs="+" default=`[0.10, 0.25, 0.50, 1.00]`；`--noise-channels` nargs="+" default=`null`（=全部输入特征） | 每个 k 注册一档 `gaussian_noise_{k:g}`；噪声后**必须**再走 `rebuild_issue_time_derived`（即复用 `_build_stressed_split` 现有 `derived = rebuild_issue_time_derived(...)` 一行 :130），使 p_pc/等效风速随受扰历史重算 | 输出 JSON `payload["stress"]` 出现 `gaussian_noise_0.1/0.25/0.5/1` 四档，每档含 `rmse_degradation` |
| ③ 特征级缺失（eligible 透传） | `stress.py` 新增轻量 helper `feature_eligible_mask(feature_axis, dropped, shape) -> np.ndarray`（broadcast 到 `[B,N,L,D]`，被 drop 列置 False）；`_stress_selections` 调用 `mcar_missingness(..., eligible=eligible)`（四缺失函数签名已支持，:61/:78/:105/:132） | CLI 新增 `--drop-features` nargs="+" default=`[]`；`--drop-mcar-rate` type=float default=`0.30` | 对每个被 drop 的通道注册一档 `featuredrop_{NAME}`：只在该列上以 0.30 速率 MCAR 缺失；不 drop 时不产生额外档 | 选 `--drop-features Wspd` 后输出出现 `featuredrop_Wspd` 档，其 `selection.eligible_channels=["Wspd"]` |
| ④ 关键气象通道预设 | `configs/sdwpf_v3_2.yaml` 新增 `stress:` 块（见下）；`_stress_selections` 读取该预设循环产出 `weatherdrop_{NAME}` | `stress.weather_channels: [Wspd, Wdir, Sp, T2m]`；`stress.weather_mcar_rate: 0.30`；`stress.gaussian_noise_levels: [0.10, 0.25, 0.50, 1.00]`；`stress.gaussian_noise_channels: null` | 对 Wspd/Wdir/Sp/T2m 逐一单独做特征级缺失，命名 `weatherdrop_Wspd/weatherdrop_Wdir/weatherdrop_Sp/weatherdrop_T2m`；这些通道正是 `rebuild_issue_time_derived` 依赖项，重算路径自动生效 | 输出出现 4 个 `weatherdrop_*` 档；metadata 记录每个档扰动的物理通道名 |
| ⑤ 退化与元数据归档 | `evaluate_missing_stress.py` main(:238-265) payload | 无新参数；在 `stress_result["selection"]` 中写入 `noise_level`/`dropped_channels`/`eligible_channels`/`train_std_hash`（训练 SD 指纹） | 沿用 `:262` 绝对退化公式；clean 档仍为基准；训练统计量（SD 向量）只来自 train split，不触碰 test/val（符合 C1-2 阈值冻结原则） | 产物 JSON 可被下游聚合脚本按 stress_name × model 读取，字段与现有 mcar/block 档同构 |

**推荐新增配置块（追加到 `configs/sdwpf_v3_2.yaml` 末尾，不改动既有键）：**

```yaml
stress:
  gaussian_noise_levels: [0.10, 0.25, 0.50, 1.00]   # × 逐特征训练 SD（标准化空间 N(0,k^2)）
  gaussian_noise_channels: null                        # null=全部观测输入特征；或 [Wspd, Wdir, Sp, T2m]
  weather_channels: [Wspd, Wdir, Sp, T2m]             # 关键气象/物理通道，单独做特征级缺失
  weather_mcar_rate: 0.30                             # 关键通道特征级缺失速率
  drop_features: []                                    # 额外指定要整列缺失的通道名
  drop_mcar_rate: 0.30
```

### 5. 验收自检清单

- [ ] `stress.py` 新增 `additive_gaussian_noise`：level=0 时输出与输入完全一致；level>0 时仅 `mask>0` 条目被改动。
- [ ] `evaluate_missing_stress.py` 运行后 `payload["stress"]` 在原 `clean/mcar_*/block_*/spatial_*/extreme_conditioned` 之外，新增 `gaussian_noise_0.1/0.25/0.5/1` 四档。
- [ ] 传入 `--drop-features Wspd`（或 config `stress.drop_features`）后新增 `featuredrop_Wspd` 档。
- [ ] 默认配置下新增 `weatherdrop_Wspd/Wdir/Sp/T2m` 四档。
- [ ] 每一档均相对 clean 输出 `rmse_degradation`、`mae_degradation`（kW 绝对量），字段名与现有 mcar 档一致。
- [ ] 每一档的 `selection` metadata 记录 `noise_level` 或 `dropped_channels`，且训练 SD 仅取自 train split。
- [ ] 噪声档与 featuredrop 档均经过 `rebuild_issue_time_derived`（p_pc/等效风速随受扰历史重算），而非对派生量单独扰动。
- [ ] 未改动 `apply_missingness_stress`/`rebuild_issue_time_derived` 既有签名与返回结构（A9 消融 `missing_stress_rmse/crps` 链路不破坏）。

### 6. 引用来源列表

1. Wen, H. *Probabilistic wind power forecasting resilient to missing values: An adaptive quantile regression approach.* Energy, 300, 131544, 2024. DOI: 10.1016/j.energy.2024.131544. URL: https://ideas.repec.org/a/eee/energy/v300y2024ics0360544224013173.html
2. Zhao, Y., Zhao, Y., Chen, Y., Liao, H., Pan, S., & Ye, L. *Interpretable Wind Power Forecasting With Feature and Loss Function Construction Guided by Domain Knowledge.* IEEE Transactions on Sustainable Energy, 17(1), 16–29, 2026. DOI: 10.1109/TSTE.2025.3597967. URL: https://orcid.org/0009-0009-6757-5654
