## 缺口 1：多数据集适配器（Kelmarsh / WIND Toolkit 跨风场同协议）（优先级 P0）

> 对照约束：**C0-1 / C3-1**（实验数据集 ≥ 3 个，含独立风场 within-farm retraining 验证，同一时序切分/输入契约/调参预算/多 seed 协议）。
> 本规范只做调研与实现设计，不在本机运行任何训练/实验，不改动现有代码；以下"落点文件/函数/配置键"均已 Read 核实行号。

### 1. 缺口定位

**现状（真实文件/函数/配置键）：**

- 全库唯一的预处理入口是 `src/prwarn/cli/preprocess_sdwpf.py`（`main()` 第 49–307 行），从原始 SCADA 表到 leakage-safe 窗口 npz 一气呵成。其参数化只覆盖"列名/单位"层：`--timestamp-col`(默认 `Tmstamp`)、`--turbine-col`(默认 `TurbID`)、`--target-col`(默认 `Patv`)、`--wind-speed-col`(默认 `Wspd`)、`--pressure-col`(默认 `Sp`)、`--temperature-col`(默认 `T2m`)、`--wind-direction-col`/`--nacelle-direction-col`、`--pressure-unit`/`--temperature-unit`、`--wind-direction-mode`(第 53–89 行)。
- 但"数据质量码"层是硬编码 SDWPF 的：`src/prwarn/data/grid.py` 中 `build_sdwpf_bundle()`(第 132 行) 内部固定调用 `sdwpf_reason_codes(work)`(第 152 行)，而 `sdwpf_reason_codes()`(第 67–111 行) 写死了 SDWPF 的异常区间——`Ndir∉[-720,720]`、`Wdir∉[-180,180]`、`pitch>89`、`power<=0 & speed>2.5` 判 UNKNOWN（第 95–110 行）。这些规则对 Kelmarsh（6 台 Senvion MM92、110 变量、mean/min/max/std 统计口径）不成立。
- `docs/PR-Warn++_v3.2_代码使用说明.md:154` 明确标注：Kelmarsh「当前需先编写数据适配器，不可直接喂给 SDWPF CLI」；`:155` 标注 CARE「无直接 CLI」。
- 下游契约已是数据集无关的：`src/prwarn/data/processed.py:12` `REQUIRED_ARRAYS`（11 个数组）+ `ProcessedSplit.validate()`(第 46–78 行)；`src/prwarn/data/split.py:14 chronological_split`（通用四切分 0.60/0.15/0.10/0.15）；`src/prwarn/data/window.py:8 make_windows`；`src/prwarn/data/torch_dataset.py:19 WindowTensorDataset` 只消费 `ProcessedSplit`。也就是说，**下游模型/图/校准管线天然可复用，缺口只在"上游把第 2/3 个数据集的原生表翻译成同一契约"**。
- 配置 `configs/sdwpf_v3_2.yaml` 的 `data` 段（第 6–26 行）键值：`dataset: sdwpf_full_v2`、`resolution_minutes: 10`、`history_steps: 24`、`forecast_steps: 6`、`feature_columns`(12 列)、`target_column: Patv`、`availability`(4 键)。无第二个数据集的 config。

**缺口是什么：** 缺一个能把 Kelmarsh（首选外部验证风场）原生 SCADA 表，经同一 rigid-grid → bundle → window → npz + metadata.json 契约产出的适配器；且数据质量码策略可插拔，不再硬套 SDWPF 的 pitch/nacelle 异常规则。

**为什么必须补（P0 理由）：** C0-1/C3-1 是投稿硬约束（来自 R2.3 + 用户指定）：单风场结果不足以支撑"空间图/方向图/物理先验"在不同风场的可迁移性 claim。Kelmarsh 是已公开（CC-BY-4.0）、10 min 分辨率（与 SDWPF 同分辨率，窗口参数可直接复用）、6 台真实机组的外部验证集，within-farm retraining 后与 SDWPF 主结果同协议对比，是成本最低、最有说服力的第二数据集。

### 2. 文献依据表

| 文献（作者-年份） | 出处/年份 | DOI/arXiv | 核验状态 | 关键做法 | 可借鉴的协议要素 |
|---|---|---|---|---|---|
| Zhou–Lu–Xiao et al. 2024 | Scientific Data 11:649（arXiv:2208.04360） | arXiv:2208.04360（Sci Data 11, 649） | VERIFIED（arXiv abs 页核对标题/作者/期刊参考；134 台、10 min、SCADA、含相对位置与内部状态） | 数据集论文显式区分「发布自带字段」与「作者处理」，并给出评估设置（feature 列、时间范围、采样） | 适配器必须在 metadata.json 里区分 `published_columns` vs `derived_columns`；数据集身份/license/sha256 入产物 |
| Draxl–Clifton–Hodge–McCaa 2015 | Applied Energy 151:355–366 | 10.1016/j.apenergy.2015.03.121 | VERIFIED（OSTI biblio 1250028 核对期刊/卷/页/DOI/作者） | WIND Toolkit 以格点 NWP（风速/风向/温度/气压/空气密度）为公开数据源，7 年 12.6 万站点 | 作为第三数据集候选：格点 NWP 经 `--density-correction` 等价风速物理管线接入；其字段（Sp/T2m）与现有 pressure/temperature 列契约一致 |
| Godahewa–Bergmeir–Webb–Hyndman–Montero-Manso 2021 | NeurIPS 2021 Track on Datasets and Benchmarks（arXiv:2105.06643） | arXiv:2105.06643 | VERIFIED（Monash 官方页 + robjhyndman.com 出版物列表核对作者/会议） | 跨 20 个异构数据集用**同一套 baseline 与同一套误差指标**评测，再按数据集分别报告 | 跨数据集必须同一 split 比例/窗口步长/输入契约/指标口径/多 seed；按数据集分行报告，不跨池平均 |
| Plumley 2022（Kelmarsh 数据，v0.0.3） | Zenodo 数据仓储 | 10.5281/zenodo.5841834（概念 DOI 10.5281/zenodo.5841833） | VERIFIED（Zenodo record 5841834 页 + Google Scholar 作者页核对：Plumley, Charlie；6 台 Senvion MM92、10 min、2016 起按年分组、CC-BY-4.0；本版 v0.0.3 数据至 2021-07） | 6 台机组、每变量给 mean/min/max/std、含风速/风向/功率/部件温度 | 适配器需处理"每变量多统计量"宽表；无 nacelle 角/pitch 列，方向走 `global` 模式 |

### 3. 标准做法要点

1. **同一契约、差异化输入映射**：跨数据集对比必须复用同一窗口构造（`make_windows`）、同一切分（`chronological_split`）、同一校验（`ProcessedSplit.validate`），差异只在"原生列名 → 契约列名"与"数据质量码策略"。（Godahewa et al. 2021）
2. **区分发布字段与作者派生字段**：每个数据集的 metadata 必须显式标注哪些列是发布自带、哪些是作者派生（如等价风速、密度校正），并冻结 train 拟合的 scaler/功率曲线。（Zhou et al. 2024）
3. **数据质量码随数据集策略化，不套用他站规则**：SDWPF 的 pitch>89/Ndir 异常区间来自其公开使用说明，不可平移到 Kelmarsh；新数据集需独立的 row-validity 策略（缺失优先 → 非物理值 → 本数据自定义异常）。（Zhou et al. 2024；Draxl et al. 2015）
4. **外部验证集 license 与可复现指纹入产物**：Kelmarsh 为 CC-BY-4.0，必须记录 license、输入文件 SHA-256（现有 `_sha256` 已实现，`preprocess_sdwpf.py:25`）与文件来源 URL。（Plumley 2022）
5. **分辨率对齐才复用窗口参数**：Kelmarsh 与 SDWPF 同为 10 min，故 `history_steps=24`(4h)、`forecast_steps=6`(1h)、`step_minutes=10` 可直接沿用；若引入 WIND Toolkit 格点数据（分辨率不同），需在 config 显式改 `resolution_minutes` 而非静默对齐。（Draxl et al. 2015）

### 4. 实现规范（映射到本代码库）

| 改动点 | 落点文件/函数/配置键 | 新增参数与默认值 | 协议要点 | 验收方式 |
|---|---|---|---|---|
| 数据质量码策略可插拔 | `src/prwarn/data/grid.py` `build_sdwpf_bundle()`（第 132 行） | 新增 `reason_code_fn: Callable[[pd.DataFrame], np.ndarray] \| None = None`（默认 `None`→回落现有 `sdwpf_reason_codes`，向后兼容）；新增 `generic_scada_reason_codes(frame, *, power_col, wind_speed_col, speed_max=25.0)` | Kelmarsh 无 pitch/nacelle 列，只用"缺失→非物理功率/风速超限"判 VALID/MISSING/ABNORMAL，不套 SDWPF 的 pitch>89、Ndir±720 规则 | 单测：对 Kelmarsh 风格宽表调用 `build_sdwpf_bundle(..., reason_code_fn=generic_scada_reason_codes)`，产出 `mask/delta_t` 形状与 SDWPF 路径一致 |
| Kelmarsh 预处理 CLI | 新建 `src/prwarn/cli/preprocess_kelmarsh.py`（复用 `build_rigid_grid`/`chronological_split`/`build_sdwpf_bundle`/`bundle_to_time_node`/`make_windows`，不新造窗口/切分逻辑） | `--timestamp-col` 默认 `"Date/Time"`；`--turbine-col` 默认 `"WTG"`；`--target-col` 默认 `"Active Power"`；`--wind-speed-col` 默认 `"Wind Speed"`；`--wind-direction-mode` 默认 `"global"`（Kelmarsh 无 nacelle 角）；`--density-correction` 默认 `False`（Kelmarsh 缺气压/温度列时关闭密度校正）；`--rated-power` 必给（6×2.05 MW≈12.3 MW） | 列映射后走与 SDWPF 完全相同的 rigid-grid→bundle→window→npz 管线；train 拟合 `feature_scaler`/`train_means`/功率曲线并冻结；`wind_from` 走 `global` 模式 | 运行后产出 `train/val/calib/test.npz`+`metadata.json`+`a_corr.npy`（+`a_geo.npy` 若给坐标）；`load_processed_split` 能加载且 `validate()` 通过 |
| metadata 增加数据集身份与来源 | `preprocess_sdwpf.py` `metadata` dict（第 259–303 行）+ 新 Kelmarsh CLI 的同类 dict | 新增键 `"dataset_name"`、`"dataset_license"`、`"dataset_origin_url"`、`"published_columns"`（发布自带列 list）、`"derived_columns"`（作者派生列 list） | 与 Zhou et al. 2024 的字段区分惯例一致；`input_sha256`（已有，第 261 行）保留 | `metadata.json` 可被 `json.load` 且含上述键；`published_columns` 与 `derived_columns` 不相交 |
| Kelmarsh 独立配置 | 新建 `configs/kelmarsh_v1.yaml`（仿 `configs/sdwpf_v3_2.yaml`） | `data.dataset: kelmarsh_v1`；`data.resolution_minutes: 10`；`data.history_steps: 24`；`data.forecast_steps: 6`；`data.split` 同 0.60/0.15/0.10/0.15；`data.reason_code_strategy: generic_scada`；`data.wind_direction_mode: global`；`data.dataset_license: CC-BY-4.0` | 同协议 within-farm retraining：同一 split/窗口/输入契约/多 seed；仅数据集身份与列映射不同 | `yaml.safe_load` 后 `data` 段键齐全；seeds 与主实验一致（后续扩 ≥10 时同步） |
| （第三数据集候选）WIND Toolkit 接入 | 复用 `preprocess_sdwpf.py` 既有 pressure/temperature 列契约 | 经 `--pressure-unit hpa --temperature-unit celsius`（已有 choices，第 59–60 行）+ `--density-correction` 接入格点 NWP | 分辨率不同时显式改 `resolution_minutes`，不静默重采样 | 作为后续 P1 选项；本缺口优先交付 Kelmarsh |

### 5. 验收自检清单

- [ ] `python -m prwarn.cli.preprocess_kelmarsh --input <kelmarsh.csv> --output-dir data/processed/kelmarsh --rated-power 12.3 --features <映射后12列> ...` 退出码 0，产出目录含 `train.npz/val.npz/calib.npz/test.npz/metadata.json/a_corr.npy`。
- [ ] `load_processed_split("data/processed/kelmarsh/test.npz")` 返回的 `ProcessedSplit.validate()` 不抛异常（11 个 `REQUIRED_ARRAYS` 齐全，`x` 4 维 `[sample,node,history,feature]`）。
- [ ] `metadata.json` 含 `dataset_name/dataset_license/dataset_origin_url/input_sha256/feature_scaler.fit_split=="train"/published_columns/derived_columns`。
- [ ] Kelmarsh 路径不调用 SDWPF 专属异常规则（`pitch>89`、`Ndir±720`）；`reason_code` 分布中 `UNKNOWN` 占比可打印核查。
- [ ] 切分边界不跨窗口：`final_origin = len − forecast_steps`（`window.py:27`）在 Kelmarsh npz 上同样成立。
- [ ] `configs/kelmarsh_v1.yaml` 与 `sdwpf_v3_2.yaml` 的 `data.split/history_steps/forecast_steps` 一致（同协议证据）。

### 6. 引用来源列表

1. Zhou, J., Lu, X., Xiao, Y., Su, J., Lyu, J., Ma, Y., Dou, D. SDWPF: A Dataset for Spatial Dynamic Wind Power Forecasting over a Large Turbine Array. *Scientific Data* 11:649, 2024.（arXiv 版：arXiv:2208.04360）URL: https://arxiv.org/abs/2208.04360
2. Draxl, C., Clifton, A., Hodge, B.-M., McCaa, J. The Wind Integration National Dataset (WIND) Toolkit. *Applied Energy* 151:355–366, 2015. DOI: 10.1016/j.apenergy.2015.03.121. URL: https://www.osti.gov/biblio/1250028
3. Godahewa, R.W., Bergmeir, C., Webb, G.I., Hyndman, R.J., Montero-Manso, P. Monash Time Series Forecasting Archive. *NeurIPS 2021 Track on Datasets and Benchmarks*, 2021.（arXiv:2105.06643）URL: https://arxiv.org/abs/2105.06643
4. Plumley, C. Kelmarsh Wind Farm Data (v0.0.3). *Zenodo*, 2022-02. DOI（版本记录）: 10.5281/zenodo.5841834（概念 DOI: 10.5281/zenodo.5841833）。URL: https://zenodo.org/record/5841834
