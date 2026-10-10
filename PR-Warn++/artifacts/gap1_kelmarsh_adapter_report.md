# 缺口 1：多数据集适配器（Kelmarsh）实施报告

> 目标：PR-Warn++（IEEE TSTE 投稿）P0 硬缺口 C0-1/C3-1，交付第 2 个数据集适配器。
> 实施日期：2026-10-10。环境：Windows / Python 3.13.13，prwarn 已 `pip install -e`。

## 1. 改动文件清单（绝对路径 + 行号范围 + 说明）

| # | 文件（绝对路径） | 类型 | 行号范围 | 一句话说明 |
|---|---|---|---|---|
| 1 | `C:\Users\weface\Desktop\prwaring-main\PR-Warn++\src\prwarn\data\grid.py` | 修改 | L11（import 加 `Callable`）；L114–L155（新增 `generic_scada_reason_codes`）；L173–L206（`build_sdwpf_bundle` 新增 `reason_code_fn` 参数并接入） | 数据质量码策略可插拔：默认 `None` 回落 `sdwpf_reason_codes`（向后兼容）；新函数只用「缺失→非物理功率/风速超限」判 VALID/MISSING/ABNORMAL，不套用 SDWPF 的 pitch/nacelle 规则。 |
| 2 | `C:\Users\weface\Desktop\prwaring-main\PR-Warn++\src\prwarn\cli\preprocess_sdwpf.py` | 修改 | L259–L270（metadata 前新增 published/derived 计算与不相交断言；metadata dict 新增 5 键） | SDWPF 产物 metadata 增加 `dataset_name/dataset_license/dataset_origin_url/published_columns/derived_columns`，`input_sha256` 保留；published/derived 不相交校验。 |
| 3 | `C:\Users\weface\Desktop\prwaring-main\PR-Warn++\src\prwarn\cli\preprocess_kelmarsh.py` | 新增 | L1–L348（全文） | Kelmarsh 预处理 CLI：默认列 `Date/Time/WTG/Active Power/Wind Speed`、`wind_direction_mode=global`、`--no-density-correction`、`--rated-power` 必给；复用 rigid-grid→bundle→window→npz 同一管线，挂 `generic_scada_reason_codes`，metadata 带数据集身份键与 reason_code 分布。 |
| 4 | `C:\Users\weface\Desktop\prwaring-main\PR-Warn++\configs\kelmarsh_v1.yaml` | 新增 | L1–L103（全文） | 仿 `sdwpf_v3_2.yaml`：`data.dataset=kelmarsh_v1`、`reason_code_strategy=generic_scada`、`wind_direction_mode=global`、`dataset_license=CC-BY-4.0`、split/history/forecast/resolution 与主实验一致；physics/graphs/model/flow/generative_baselines/calibration/risk 照抄。 |
| 5 | `C:\Users\weface\Desktop\prwaring-main\PR-Warn++\tests\test_preprocess_kelmarsh.py` | 新增 | L1–L167（5 个测试方法） | 覆盖 generic 质量码标签/形状、bundle 可插拔、metadata 键与不相交、yaml 一致性、合成 CSV 全管线 smoke + `load_processed_split().validate()`。 |

未触碰所有权清单外的任何文件（matrix.py / resolve.py / statistics.py / stress.py / evaluate*.py / backbone.py / train_deterministic.py / sdwpf_v3_2.yaml 均只读）。未执行任何 git 操作。

## 2. 验收自检清单（逐项 ✓/✗ + 证据）

| 规范 §5 清单项 | 结果 | 证据（命令 / 输出摘要） |
|---|---|---|
| `python -m compileall` 通过 | ✓ | `python -m compileall -q src tests` → `COMPILEALL OK`，退出码 0。 |
| 新增单测 pytest 全绿 | ✓ | `python -m pytest tests\test_preprocess_kelmarsh.py -v` → **5 passed in 1.56s**（GenericReasonCodeTests×2、ConfigTests、SmokePipelineTests×2）。 |
| CLI smoke 退出码 0，产物含 train/val/calib/test.npz + metadata.json + a_corr.npy | ✓ | 子进程 `python -m prwarn.cli.preprocess_kelmarsh ...` → `EXITCODE=0`；目录含 `train.npz val.npz calib.npz test.npz metadata.json a_corr.npy`（未给坐标故无 a_geo.npy，符合「+a_geo.npy 若给坐标」）。 |
| `load_processed_split(test.npz).validate()` 不抛异常（11 个数组齐全、x 4 维） | ✓ | `SmokePipelineTests::test_full_pipeline_loads_and_validates` 对 4 个 split 逐一 `load_processed_split(...).validate()` 通过。 |
| metadata.json 含 dataset_name/dataset_license/dataset_origin_url/input_sha256/feature_scaler.fit_split=="train"/published_columns/derived_columns | ✓ | smoke metadata 实测：`dataset_name=kelmarsh_v1`、`dataset_license=CC-BY-4.0`、`dataset_origin_url=https://zenodo.org/record/5841834`、`fit_split=train`、`has_sha256=True`、`published=['Wind Speed','Wind Direction','Active Power','Ambient Temp']`、`derived=[]`。 |
| Kelmarsh 路径不调用 SDWPF 专属异常规则（pitch>89、Ndir±720）；reason_code 分布可打印核查 | ✓ | 挂的是 `generic_scada_reason_codes`（无 pitch/nacelle 分支）；metadata 输出 `reason_code_strategy=generic_scada` 与 `reason_code_distribution_train`；单测断言 generic 路径不产出 `UNKNOWN`，且注入的缺失/超风速行分别落 MISSING/ABNORMAL。 |
| 切分边界不跨窗口：`final_origin = len − forecast_steps` 在 Kelmarsh npz 成立 | ✓ | 由 `make_windows`（window.py:27）构造保证；`validate()` 通过即等价满足（各 split 独立成窗，chronological_split 按时间戳切分）。 |
| `kelmarsh_v1.yaml` 与 `sdwpf_v3_2.yaml` 的 split/history_steps/forecast_steps 一致 | ✓ | 实测：`split match=True`、`history match=True`、`forecast match=True`、`resolution=10`；ConfigTests 断言通过。 |
| 既有测试无回归 | ✓ | `pytest tests\test_data.py test_processed.py test_preprocess_cli.py test_physics_graphs.py` → **11 passed**。 |

## 3. 需在服务器运行的实验 / 后续清单

1. **下载并核验真实 Kelmarsh 数据**：Zenodo record 5841834（v0.0.3，CC-BY-4.0），计算输入文件 SHA-256 写入 metadata（本机网络不可达，已用合成 CSV smoke 替代）。
2. **真实表预处理**：Kelmarsh 原生为「每变量 mean/min/max/std」宽表、6 台 Senvion MM92；需先映射实际列名到 CLI 参数（`--timestamp-col/--turbine-col/--target-col/--wind-speed-col/--wind-direction-col`），提供 `--rated-power 12.3`（或按实际单位声明），如有坐标则加 `--locations` 以产出 `a_geo.npy`。
3. **within-farm retraining（C3-1 核心）**：用 `configs/kelmarsh_v1.yaml` 在 Kelmarsh 上多 seed `[2025..2029]` 重训，与 SDWPF 主结果按同一 split/窗口/输入契约/多 seed 协议分行对比（不跨池平均）。
4. **第三数据集 WIND Toolkit（P1，后续）**：经 `preprocess_sdwpf.py` 既有 pressure/temperature 列契约 + `--density-correction` 接入，分辨率不同时显式改 `resolution_minutes`。

## 4. 需编排者集成 / 复核的配置键

我**未修改** `configs/sdwpf_v3_2.yaml`（按禁令）。为使两数据集配置对称，建议在 `sdwpf_v3_2.yaml` 的 `data:` 段（`dataset:` 键附近）补入同构键：

| 键 | 建议值（sdwpf_v3_2.yaml） | 插入位置 |
|---|---|---|
| `data.reason_code_strategy` | `sdwpf_official` | data 段，`dataset:` 之后 |
| `data.dataset_license` | `CC-BY-4.0` | 同上 |
| `data.dataset_origin_url` | `https://arxiv.org/abs/2208.04360` | 同上 |
| `data.wind_direction_mode` | `relative_plus_nacelle` | 同上（与现有 `graphs.wind_direction_mode` 对齐） |

另需编排者**裁决一处不一致**：`kelmarsh_v1.yaml` 的 `graphs:` 段按指令「照抄数值」保留了 `wind_direction_mode: relative_plus_nacelle` / `nacelle_direction_feature: Ndir`，但 Kelmarsh 无 nacelle 角。真实训练前建议将 Kelmarsh 的 `graphs.wind_direction_mode` 改为 `global`、方向图特征改为 `Wind Direction`（或在 Kelmarsh 上关闭 directional 图）——此为配置层决策，未擅自改动。

---
**报告路径**：`C:\Users\weface\AppData\Local\Doubao\User Data\Default\.doubao\agent_mode\workspace\.sessions\38446250299098370\agents\s_000ERpnba7s\artifacts\gap1_kelmarsh_adapter_report.md`
