# PR-Warn++ v3.2 代码使用说明

> 适用仓库：`PR-Warn++` 当前 v3.2 实现  
> 目标任务：基于 SDWPF 的 10–60 min 多风机联合概率功率预测、校准与风险代理评估  
> 当前代码状态：算法代码已完成；52 个轻量单元测试、NumPy smoke 与 Python 编译已在本地通过；全量 SDWPF、CUDA、HDF5 神经训练、五种子实验和 GPU 性能仍需服务器验证。

本文不是结果报告，而是一份从下载数据开始、能够逐步执行到论文完整实验的操作手册。凡标注“服务器待验证”的步骤，表示命令和代码路径已完成，但当前仓库尚无服务器实测结果，不能把它写成论文中的已取得结果。

## 1. 先理解整个代码链

主链如下：

```text
SDWPF 原始 SCADA 与机组坐标
  -> 原始数据审计与 SHA-256 冻结
  -> 10 min 刚性时间网格
  -> X_fill、M、delta_t、reason_code
  -> Train-only 标准化、空气密度修正、单调经验功率曲线 P_pc
  -> Train/Val/Calib/Test 滑动窗口
  -> 确定性中心 Y_det 与条件表示 H_t
  -> 最终误差 E = Y - Y_det
  -> Direct CFM 或联合概率基线
  -> Calib-only 共形校准
  -> frozen Test 上的 CRPS/ES/VS、区间、爬坡、CVaR、OOD 与数据质量代理
  -> 配对统计、五种子汇总与 GPU 效率报告
```

默认张量合同：

| 名称 | 形状 | 含义 |
|---|---:|---|
| `x` / `mask` / `delta_t` | `[B,N,L,D]` | 批次、风机、历史步、特征 |
| `p_pc` / `y` / `y_det` | `[B,N,H]` | 物理中心、真实功率、确定性预测 |
| `hidden` | `[B,N,d]` | 确定性模型提供给概率模型的条件表示 |
| 采样期误差/功率场景 | `[M,B,N,H]` | 场景数、批次、风机、预测步 |
| HDF5 中的风场场景 | `[B,M,H]` | 每个起报时刻的风场总功率场景 |

默认 `N=134`、`L=24`、`H=6`，即使用过去 4 h 预测未来 10、20、30、40、50、60 min。

## 2. 仓库目录

```text
PR-Warn++/
├─ configs/
│  └─ sdwpf_v3_2.yaml              # 主实验配置
├─ data/
│  ├─ raw/                          # 自行下载的不可变原始数据
│  └─ processed/                    # 预处理生成的 NPZ 与图
├─ outputs/                         # 训练、场景、评估、统计产物
├─ src/prwarn/
│  ├─ data/                         # 审计、网格、窗口、缺失机制、天气协议
│  ├─ physics/                      # 空气密度与经验功率曲线
│  ├─ graphs/                       # 地理、相关、方向图
│  ├─ models/                       # 确定性、边际、CFM、CVAE、DDIM、VAE-CFM
│  ├─ baselines/                    # 非神经残差场景基线
│  ├─ calibration/                  # static/ACI/context-fallback 校准
│  ├─ risk/                         # 爬坡、偏差、CVaR、OOD、风险组合
│  ├─ eval/                         # 指标与依赖样本统计
│  ├─ experiments/                  # A0–A11、B0–B9、G1–G6 实验合同
│  └─ cli/                          # 24 个可直接运行的命令行入口
├─ tests/                           # 52 个依赖较轻的合同测试
├─ README.md                        # 项目概览
├─ SERVER_RUNBOOK.md                # 按 Gate 执行的服务器清单
└─ pyproject.toml                   # Python 包和依赖声明
```

原始大数据、模型权重和最终实验结果不在仓库中。

## 3. 环境安装

### 3.1 推荐版本

- Python：3.10–3.12，论文服务器建议固定为 Python 3.11。
- PyTorch：不低于 2.2，CUDA 版本必须与服务器驱动匹配。
- 主要依赖：NumPy、Pandas、PyYAML、SciPy、scikit-learn、PyTorch、h5py。
- Parquet：安装 `data` extra 后使用 PyArrow。
- 测试：安装 `dev` extra 后使用 pytest/ruff；当前测试也可由 `unittest` 发现。

### 3.2 Windows 本地环境

在仓库根目录执行：

```powershell
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[data,dev]"
```

若系统没有 `py`，用已安装 Python 的完整路径创建虚拟环境。

### 3.3 Linux/CUDA 服务器

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

先根据服务器 CUDA/驱动，从 PyTorch 官方安装页选择固定版本的 GPU wheel，再安装本项目：

```bash
# 先执行由 PyTorch 官方选择器给出的、与本机 CUDA 匹配的安装命令。
python -m pip install -e '.[data,dev]'
```

论文实验中不要在不同种子之间静默更换 PyTorch、CUDA、驱动或 GPU 型号。

### 3.4 包导入方式

执行过 `pip install -e ...` 后可以直接运行 `python -m prwarn...`。若只想从源码临时导入：

Windows PowerShell：

```powershell
$env:PYTHONPATH = (Resolve-Path "src").Path
```

Linux：

```bash
export PYTHONPATH="$PWD/src"
```

### 3.5 安装后自检

本地轻量测试：

```powershell
python -m unittest discover -s tests -v
python -m prwarn.cli.smoke_numpy
python -m compileall -q src tests
```

当前版本的预期结果是 52 个测试全部通过，`smoke_numpy` 输出功率曲线点数、图形状、`[50,8,5,6]` 场景形状以及 CRPS/ES/校准/风险张量摘要。smoke 中的随机数值只用于检查代码链，不是论文结果。

服务器 CUDA 自检：

```bash
python -m prwarn.cli.smoke_torch --device cuda
```

这是第一项“服务器待验证”。如果失败，不要开始全量训练；先检查 `torch.cuda.is_available()`、CUDA wheel、驱动、显存和张量形状。

## 4. 数据下载与冻结

### 4.1 哪些数据是必需的

| 数据 | 是否必需 | 用途 | 当前代码入口 |
|---|---|---|---|
| SDWPF Full Version 2 | 必需 | 134 台风机主实验 | `audit_sdwpf`、`preprocess_sdwpf` |
| SDWPF 机组坐标/海拔 | 必需 | 地理图与方向图 | `preprocess_sdwpf --locations` |
| Kelmarsh | 可选 | 6 台真实风场外部验证 | 当前需先编写数据适配器，不可直接喂给 SDWPF CLI |
| CARE | 可选 | 故障/数据质量压力证据 | 不是同步多节点主图，当前无直接 CLI |
| issue-time NWP/ensemble 档案 | A11 可选但关键 | 真正可部署的未来天气协议 | `audit_weather_archive`、`attach_future_weather` |
| 未来 ERA5 | 仅 oracle 上界 | 非部署型信息上界 | `attach_future_weather --protocol oracle_era5` |

主实验先只下载 SDWPF。官方 Figshare 页面为：

- `https://figshare.com/articles/dataset/SDWPF_dataset/24798654`
- DOI：`10.6084/m9.figshare.24798654`
- 版本：Version 2
- 许可：CC BY 4.0
- 时间范围：2020-01 至 2021-12
- 采样：134 台风机、10 min、约 1140 万条记录

选择 `sdwpf_full` 中以下两个文件：

```text
sdwpf_2001_2112_full.parquet
sdwpf_turb_location_elevation.csv
```

CSV 与 Parquet 的主表内容相同，建议下载 Parquet，读取更快且体积更小。

### 4.2 Windows 下载命令

```powershell
New-Item -ItemType Directory -Force data\raw\sdwpf | Out-Null

Invoke-WebRequest `
  -Uri "https://ndownloader.figshare.com/files/46005810" `
  -OutFile "data\raw\sdwpf\sdwpf_2001_2112_full.parquet"

Invoke-WebRequest `
  -Uri "https://ndownloader.figshare.com/files/46005798" `
  -OutFile "data\raw\sdwpf\sdwpf_turb_location_elevation.csv"
```

若直链因 Figshare 版本调整而失效，从上述 Version 2 页面按精确文件名下载，不要改用 Version 1 或 245-day KDD 数据后仍称为 `sdwpf_full_v2`。

### 4.3 Linux 下载命令

```bash
mkdir -p data/raw/sdwpf
curl -L 'https://ndownloader.figshare.com/files/46005810' \
  -o data/raw/sdwpf/sdwpf_2001_2112_full.parquet
curl -L 'https://ndownloader.figshare.com/files/46005798' \
  -o data/raw/sdwpf/sdwpf_turb_location_elevation.csv
```

### 4.4 下载后必须冻结的信息

Windows：

```powershell
Get-Item data\raw\sdwpf\* | Select-Object Name,Length,LastWriteTime
Get-FileHash data\raw\sdwpf\sdwpf_2001_2112_full.parquet -Algorithm SHA256
Get-FileHash data\raw\sdwpf\sdwpf_turb_location_elevation.csv -Algorithm SHA256
```

Linux：

```bash
ls -lh data/raw/sdwpf
sha256sum data/raw/sdwpf/sdwpf_2001_2112_full.parquet
sha256sum data/raw/sdwpf/sdwpf_turb_location_elevation.csv
```

把版本、下载日期、字节数、SHA-256 和论文 DOI 写入实验日志。`audit_sdwpf` 和 `preprocess_sdwpf` 也会把主表 SHA-256 写入 JSON/metadata，但坐标文件仍应单独记录哈希。

### 4.5 先看 schema、单位和额定功率

```powershell
python -c "import pyarrow.parquet as pq; p=pq.ParquetFile(r'data/raw/sdwpf/sdwpf_2001_2112_full.parquet'); print(p.metadata.num_rows); print(p.schema)"
python -c "import pandas as pd; p=r'data/raw/sdwpf/sdwpf_2001_2112_full.parquet'; d=pd.read_parquet(p,columns=['Patv','Sp','T2m']); print(d.quantile([.01,.5,.99]))"
python -c "import pandas as pd; p=r'data/raw/sdwpf/sdwpf_turb_location_elevation.csv'; d=pd.read_csv(p); print(d.columns.tolist()); print(d.head()); print(d.shape)"
```

运行前必须确认：

- 时间列为 `Tmstamp`，风机列为 `TurbID`。
- 目标 `Patv` 和 `--rated-power` 使用同一单位。
- 官方机型是 SL1500/82，名义额定功率为 1500 kW；只有确认当前 `Patv` 以 kW 表示时，才使用 `--rated-power 1500`。
- `Sp` 若为 Pa，使用 `--pressure-unit pa`；若为 hPa，改为 `hpa`。
- `T2m` 若为 K，使用 `--temperature-unit kelvin`；若为摄氏度，改为 `celsius`。
- 坐标列默认为 `x`、`y`，官方说明单位为 m，因此建议写 `--coordinate-unit meter`。
- 坐标文件应覆盖 134 台机组且 `TurbID` 能与主表精确对齐。

不要仅凭数量级让代码自动猜单位；当前预处理器有意要求调用者声明单位。

## 5. Gate 1：原始数据审计

```powershell
python -m prwarn.cli.audit_sdwpf `
  --input data\raw\sdwpf\sdwpf_2001_2112_full.parquet `
  --output outputs\audit\sdwpf_raw.json `
  --timestamp-col Tmstamp `
  --turbine-col TurbID `
  --expected-turbines 134 `
  --frequency-minutes 10
```

作用：

- 读取 CSV/Parquet；
- 检查必需列、风机数、重复 `(TurbID,Tmstamp)`；
- 检查时间范围、10 min 网格缺口；
- 统计 official-style `valid/missing/unknown/abnormal` 原因码；
- 计算主表 SHA-256；
- 输出 `gate_ready` 和 `issues`。

只有在重复键为 0、schema/时间频率/机组数确认无误后才进入预处理。真实数据存在缺测并不等于下载损坏；关键是缺测必须被刚性网格、mask 和 reason code 显式保留。

这是第二项“服务器或全量数据待验证”。

## 6. Gate 1–2：预处理历史协议

确认 `Patv` 单位为 kW 后，主命令为：

```powershell
python -m prwarn.cli.preprocess_sdwpf `
  --input data\raw\sdwpf\sdwpf_2001_2112_full.parquet `
  --locations data\raw\sdwpf\sdwpf_turb_location_elevation.csv `
  --output-dir data\processed\sdwpf_v3_2_history_only `
  --rated-power 1500 `
  --pressure-unit pa `
  --temperature-unit kelvin `
  --coordinate-unit meter `
  --wind-direction-mode relative_plus_nacelle `
  --features Wspd Wdir Ndir Pab1 Pab2 Pab3 Prtv Patv Etmp Itmp Sp T2m
```

关键参数：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--history` | 24 | 历史窗口步数，10 min 数据下为 4 h |
| `--horizon` | 6 | 预测步数，即 10–60 min |
| `--frequency` | `10min` | 刚性时间网格频率 |
| `--step-minutes` | 10 | `delta_t` 的步长，也是 metadata 中的分辨率 |
| `--rated-power` | 必填 | 与 `Patv` 同单位的单机额定功率 |
| `--density-correction` | 开 | 用 `rho=p/(287.05T)` 和等效风速；A1 用 `--no-density-correction` |
| `--curve-bins` | 50 | 经验功率曲线分箱数 |
| `--curve-min-count` | 20 | 有效分箱最少 Train 样本数 |
| `--geographic-k` | 8 | 地理图每行保留的近邻数 |
| `--correlation-k` | 12 | 相关图每行保留的近邻数 |
| `--correlation-mode` | `difference` | Train 功率差分相关；A3 与 `raw` 比较 |
| `--wind-direction-mode` | `relative_plus_nacelle` | `Wdir + Ndir` 转成气象来向角；也可声明 `global` |

实现边界：

- scaler、缺失填充值、功率曲线和相关图只用 Train 拟合。
- `x` 被 Train mean/std 标准化；`y/current_y/p_pc` 保留物理功率单位。
- `P_pc` 在 `history_only` 协议中只使用起报时刻的等效风速，然后沿 H 复制；不读取未来真实风速或 ERA5。
- 时间按 60%/15%/10%/15% 切为 Train/Val/Calib/Test，窗口在各 split 内部构造，不跨边界。
- 当前 `preprocess_sdwpf` 不读取 YAML；修改 YAML 中的 `data.split/history_steps` 不会自动改变预处理器，必须同步修改 CLI 参数或代码。
- `--locations` 在 parser 中可省略，但当前 `train_deterministic` 会无条件加载 `coordinates.npy` 和 `a_geo.npy`，因此完整训练链中它实际上是必需的。
- 输出目录使用 `exist_ok=False`，已存在时会拒绝覆盖；更改协议时使用新目录。

预处理产物：

```text
data/processed/sdwpf_v3_2_history_only/
├─ train.npz
├─ val.npz
├─ calib.npz
├─ test.npz
├─ coordinates.npy
├─ a_geo.npy
├─ a_corr.npy
└─ metadata.json
```

每个 split NPZ：

| 键 | 形状 | 说明 |
|---|---:|---|
| `x` | `[S,N,L,D]` | Train-only 标准化历史载体 |
| `mask` | `[S,N,L,D]` | 1=有效观测，0=缺失/未知/异常 |
| `delta_t` | `[S,N,L,D]` | 距上次有效观测的分钟数 |
| `y` / `y_mask` | `[S,N,H]` | 未来目标与有效标记 |
| `p_pc` | `[S,N,H]` | 弱物理中心 |
| `current_y` / `current_y_mask` | `[S,N]` | 起报时刻功率与有效标记 |
| `wind_from` | `[S,N]` | 未标准化的起报时刻气象来向角，单位 degree |
| `origin_time` | `[S]` | 起报时间 |
| `turbines` | `[N]` | 固定节点顺序 |

`metadata.json` 保存输入 SHA-256、split 边界、Train scaler、功率曲线状态、物理单位、特征索引、方向语义和图构建参数。预处理后先人工检查它，再训练。

这是第三项“全量数据/服务器待验证”。预处理器会在内存中处理全量表和刚性网格，建议在大内存节点执行。

## 7. 配置文件如何生效

主配置为 `configs/sdwpf_v3_2.yaml`。

| 配置段 | 主要内容 | 被谁使用 |
|---|---|---|
| `experiment` | 名称、主 seed、五个固定 seeds | 所有训练/矩阵命令 |
| `data` | 协议说明、窗口、split、特征、availability | 研究合同；预处理 CLI 的对应参数仍需显式同步 |
| `physics` | 密度参考值、曲线分箱 | 研究合同；预处理当前用 CLI 默认/参数 |
| `graphs` | 图组件、k、方向距离/角度带宽 | 确定性训练和 CFM collator |
| `model` | 架构、残差/直预测、hidden、dropout、missing inputs、损失 | `train_deterministic` |
| `flow` | CFM hidden、coupling、Euler 步数、场景数 | `train_flow` 和场景评估 |
| `generative_baselines` | CVAE/VAE-CFM/DDIM 超参数 | `train_neural_baseline` |
| `calibration` | alpha、static/ACI/context 参数 | `evaluate`、`evaluate_marginal` |
| `risk` | OOD/数据质量权重、成本、爬坡阈值/持续时间、CVaR | 风险与综合评估命令 |

优先级通常是：CLI 显式覆盖 > YAML > 代码默认。训练命令会把最终配置和 CLI 执行信息写入 run 的 `config.json`，不能只保留原始 YAML。

## 8. Gate 3：确定性中心与点预测基线

### 8.1 主模型

```powershell
python -m prwarn.cli.train_deterministic `
  --data-dir data\processed\sdwpf_v3_2_history_only `
  --config configs\sdwpf_v3_2.yaml `
  --experiment-id main_det_s2025 `
  --seed 2025 `
  --epochs 100 `
  --patience 12 `
  --batch-size 32 `
  --device cuda
```

默认模型是 `DynamicMultiGraphResidualForecaster`：时间卷积编码历史，融合 `A_geo/A_corr/A_dir/A_adp`，预测残差 `R=Y-P_pc`，再得到 `Y_det=clip(P_pc+R,0,P_rated)`。

主要可控项：

```text
--architecture multigraph|gru|tcn|graphwavenet|agcrn
--target-mode physics_residual|direct_power
--graph-components geo corr directional adaptive
--missing-inputs mask delta_t
--missing-inputs                 # 选项后不跟值，表示 X-only
--learning-rate / --weight-decay / --num-workers
```

`graphwavenet` 和 `agcrn` 是共享信息/训练合同下的 style implementation，不是外部作者仓库的逐行复现。

默认输出：

```text
outputs/main_det_s2025/
├─ config.json                   # 最终配置与 execution 字段
├─ runtime.json                  # config hash、git commit、主机、Python/Torch/CUDA/GPU
├─ best.pt                       # 最佳 Val checkpoint
├─ history.json                  # 每 epoch Train/Val loss
├─ metrics.json                  # 四 split 的 MAE/RMSE/R2、P_pc、persistence
├─ deterministic_train.h5
├─ deterministic_val.h5
├─ deterministic_calib.h5
└─ deterministic_test.h5
```

确定性 HDF5 的核心键：

- `y/y_mask/p_pc/y_det`：`[S,N,H]`；
- `hidden`：`[S,N,d]`，概率模型的条件；
- `graph_weights`：每个起报时刻的图门控；
- `static_graphs/adaptive_graph/coordinates`：图状态；
- `wind_from/current_y/current_y_mask/origin_time`：评估和重建需要的起报信息。

### 8.2 persistence 与物理曲线参考

```powershell
python -m prwarn.cli.evaluate_point_references `
  --deterministic-run outputs\main_det_s2025 `
  --split test `
  --output outputs\main_det_s2025\point_references_test.json
```

输出 persistence 和 `P_pc` 的 overall/H1–H6 MAE、RMSE、R²/NSE。该命令使用同一 frozen cache，保证参考方法与主模型的样本 mask 一致。

### 8.3 受控点预测基线

```powershell
python -m prwarn.cli.train_deterministic --data-dir data\processed\sdwpf_v3_2_history_only --architecture gru --experiment-id b2_gru_s2025 --seed 2025 --device cuda
python -m prwarn.cli.train_deterministic --data-dir data\processed\sdwpf_v3_2_history_only --architecture tcn --experiment-id b3_tcn_s2025 --seed 2025 --device cuda
python -m prwarn.cli.train_deterministic --data-dir data\processed\sdwpf_v3_2_history_only --architecture graphwavenet --experiment-id b8_graphwavenet_s2025 --seed 2025 --device cuda
python -m prwarn.cli.train_deterministic --data-dir data\processed\sdwpf_v3_2_history_only --architecture agcrn --experiment-id b9_agcrn_s2025 --seed 2025 --device cuda
```

主模型和所有基线应使用同一 processed data、seed 集、训练轮数上限、early stopping、target mask 和信息可用性。

以上全部属于“服务器待验证”。先用一个 seed 和较少 epoch 完成端到端 sanity run，再冻结超参数并运行五种子。

## 9. Gate 4：概率模型与场景生成

所有概率模型都依赖确定性 run 中的 `deterministic_train/val/calib/test.h5`。不要直接从原始 NPZ 跳到 CFM。

### 9.1 Direct CFM 主模型

```powershell
python -m prwarn.cli.train_flow `
  --deterministic-run outputs\main_det_s2025 `
  --config configs\sdwpf_v3_2.yaml `
  --experiment-id main_cfm_s2025 `
  --seed 2025 `
  --epochs 100 `
  --patience 12 `
  --batch-size 16 `
  --scenarios 100 `
  --steps 8 `
  --coupling independent `
  --device cuda
```

内部步骤：

1. 从 Train cache 计算 `E=Y-Y_det` 的逐 horizon mean/std；
2. 用标准高斯源和条件 `hidden + adjacency` 训练 `DirectCFMVelocity`；
3. Val 上固定随机种子做 early stopping；
4. 用 Euler 反向 ODE 生成 Calib/Test 场景；
5. 把功率裁剪到 `[0,rated_power]`；
6. 默认保存风场总功率场景，计算 turbine CRPS 摘要。

输出：

```text
outputs/main_cfm_s2025/
├─ config.json
├─ runtime.json
├─ best.pt
├─ history.json
├─ metrics.json
├─ scenarios_calib.h5
└─ scenarios_test.h5
```

场景 HDF5 的核心键：

| 键 | 形状 | 说明 |
|---|---:|---|
| `farm_scenarios` | `[S,M,H]` | 风场总功率场景 |
| `lower/upper` | `[S,H]` | 未校准的场景分位区间 |
| `y_farm/y_farm_mask` | `[S,H]` | 风场真值与完整节点 mask |
| `y_det_farm` | `[S,H]` | 确定性风场中心 |
| `current_y_farm` | `[S]` | 起报时刻风场功率 |
| `origin_time` | `[S]` | 起报时间 |
| `turbine_scenarios` | `[S,M,N,H]` | 仅指定 `--save-full-scenarios` 时保存 |

HDF5 attrs 还保存 `n_scenarios`、`ode_steps`、`rated_power`、`n_nodes`、`interval_alpha` 和 `scenario_method`。

`--save-full-scenarios` 会显著增大文件。若只做风场 CRPS/ES/VS、区间、爬坡和 CVaR，默认风场场景已足够；只有做风机间相关性、节点级诊断时才开启。

### 9.2 A5 真正的 minibatch OT-CFM

```powershell
python -m prwarn.cli.train_flow `
  --deterministic-run outputs\main_det_s2025 `
  --experiment-id a5_ot_cfm_s2025 `
  --coupling optimal_transport `
  --device cuda
```

该模式通过 SciPy Hungarian assignment 做 minibatch 最优配对，不是把普通 CFM 改名为 OT-CFM。必须同时记录训练耗时和显存/吞吐开销；它是可选消融，不是主设置。

### 9.3 G1–G3 非神经联合残差基线

```powershell
python -m prwarn.cli.fit_probabilistic_baselines `
  --deterministic-run outputs\main_det_s2025 `
  --experiment-id joint_baseline_s2025 `
  --methods gaussian_residual residual_bootstrap gaussian_copula `
  --scenarios 100 `
  --chunk-size 128 `
  --seed 2025
```

每种方法生成独立目录，例如：

```text
outputs/joint_baseline_s2025_gaussian_copula/
├─ baseline_state.npz
├─ config.json
├─ runtime.json
├─ metrics.json
├─ scenarios_calib.h5
└─ scenarios_test.h5
```

- `gaussian_residual`：逐 horizon 均值/标准差的参数残差场景；
- `residual_bootstrap`：完整 residual field 重采样，保留节点/horizon 联合结构；
- `gaussian_copula`：经验边际 + 正则化高斯 copula 依赖；
- `--fit-max-origins K`：协方差成本过高时，只在 Train 内固定种子抽取 K 个 origin，索引与数量写入 config。所有相关方法必须用同一 K。

`baseline_state.npz` 对 A9 概率压力重生成是必需的；旧 run 如果没有它，需要用当前代码重新拟合。

### 9.4 G4/G5 与 A4 神经场景基线

```powershell
python -m prwarn.cli.train_neural_baseline --method cvae --deterministic-run outputs\main_det_s2025 --experiment-id g4_cvae_s2025 --seed 2025 --scenarios 100 --device cuda
python -m prwarn.cli.train_neural_baseline --method ddim --deterministic-run outputs\main_det_s2025 --experiment-id g5_ddim_s2025 --seed 2025 --scenarios 100 --sampling-steps 20 --device cuda
python -m prwarn.cli.train_neural_baseline --method vae_cfm --deterministic-run outputs\main_det_s2025 --experiment-id a4_vae_cfm_s2025 --seed 2025 --scenarios 100 --sampling-steps 8 --device cuda
```

- CVAE：图条件 encoder/decoder，loss 为重建项 + `beta*KL`；采样一步完成。
- DDIM：离散 VP epsilon-prediction diffusion，训练步默认 100，采样默认 20，`eta=0`。
- VAE-CFM：仅用于 Direct CFM vs VAE-CFM 的 A4 消融，包含 latent VAE 与 flow loss。
- 三者输出与 Direct CFM 相同的 `scenarios_calib/test.h5` 合同，因此可复用 `evaluate`、`compare_scenario_runs` 和压力测试。

### 9.5 B4–B6 边际概率基线

```powershell
python -m prwarn.cli.train_marginal_baseline --method gaussian --deterministic-run outputs\main_det_s2025 --experiment-id b4_gaussian_s2025 --seed 2025 --device cuda
python -m prwarn.cli.train_marginal_baseline --method student_t --deterministic-run outputs\main_det_s2025 --experiment-id b5_student_t_s2025 --seed 2025 --student-mc-samples 1000 --device cuda
python -m prwarn.cli.train_marginal_baseline --method quantile --deterministic-run outputs\main_det_s2025 --experiment-id b6_quantile_s2025 --seed 2025 --quantiles 0.05 0.1 0.5 0.9 0.95 --device cuda
```

输出 `marginal_calib.h5` 和 `marginal_test.h5`，包含 `[S,N,H,Q]` 分位数、真实值、mask 和 origin time。接着必须在独立 Calib 上做 pooled per-horizon CQR：

```powershell
python -m prwarn.cli.evaluate_marginal `
  --run outputs\b6_quantile_s2025 `
  --output-dir outputs\b6_quantile_s2025\evaluation_cqr `
  --alpha 0.10
```

输出 `calibrated_test.h5` 与 `metrics.json`，报告点误差、pinball、overall/H1–H6 区间指标。

### 9.6 B7 Deep Ensemble

先独立训练至少两个、通常五个确定性成员，然后：

```powershell
python -m prwarn.cli.build_deep_ensemble `
  --member outputs\det_seed2025 `
  --member outputs\det_seed2026 `
  --member outputs\det_seed2027 `
  --output-dir outputs\b7_deep_ensemble `
  --alpha 0.10
```

成员确定性预测被视为 ensemble 场景，命令检查成员唯一性、节点顺序、时间和真值一致性，输出统一的 `scenarios_calib/test.h5`。

## 10. Gate 5：Calib-only 校准与 frozen Test 评估

### 10.1 主 static split conformal

```powershell
python -m prwarn.cli.evaluate `
  --flow-run outputs\main_cfm_s2025 `
  --config configs\sdwpf_v3_2.yaml
```

默认输出 `outputs/main_cfm_s2025/evaluation/`。流程严格为：

1. 从 `scenarios_calib.h5` 拟合逐 horizon conformal correction；
2. 不再拟合参数地应用到 `scenarios_test.h5`；
3. 在 Test 上报告 wind-farm CRPS、Energy Score、Variogram Score、依赖诊断；
4. 报告 PICP、PINAW、Winkler overall/H1–H6；
5. 按配置的多个阈值、持续时间和方向报告 ramp Brier/AUPRC/reliability；
6. 报告 `CVaR_0.90/0.95/0.99` 摘要；
7. 写出 `calibrated_test.h5` 和 `metrics.json`。

### 10.2 A7 chronological ACI

```powershell
python -m prwarn.cli.evaluate `
  --flow-run outputs\main_cfm_s2025 `
  --calibration-method aci `
  --aci-gamma 0.01 `
  --output-dir outputs\main_cfm_s2025\evaluation_aci
```

ACI 在 Test 中执行“先预测、再看到当前真值后更新”，不能先使用当前目标修正当前区间。可用 `--aci-rolling-window` 限制历史窗口。

### 10.3 A8 context/OOD fallback

```powershell
python -m prwarn.cli.evaluate `
  --flow-run outputs\main_cfm_s2025 `
  --calibration-method context_fallback `
  --output-dir outputs\main_cfm_s2025\evaluation_context
```

它用 Calib 上的起报可用 context 建立近邻校正和 OOD 阈值，远离 Calib support 时回退到保守修正。这是经验方法，不是分布无关的条件覆盖保证。

注意：`evaluate` 的参数仍叫 `--flow-run`，但只要 run 具有统一 `scenarios_calib/test.h5`，也可以评估 residual baseline、CVAE、DDIM、VAE-CFM 或 deep ensemble。

## 11. 配对比较、多种子与效率

### 11.1 两个联合场景 run 的配对比较

```powershell
python -m prwarn.cli.compare_scenario_runs `
  --run-a outputs\main_cfm_s2025 `
  --run-b outputs\joint_baseline_s2025_gaussian_copula `
  --label-a direct_cfm `
  --label-b gaussian_copula `
  --output outputs\comparisons\cfm_vs_copula_s2025.json `
  --hac-lags 5 `
  --bootstrap 2000 `
  --confidence 0.95 `
  --ramp-threshold-fraction 0.10 `
  --ramp-duration 3
```

该命令要求两个 Test HDF5 的 origin、truth、mask 和场景数可配对。报告逐 origin CRPS/ES/VS 的 A-minus-B loss、HAC/DM、按天 cluster bootstrap；ramp 还按完整连续正事件 episode 重采样。符号约定：差值为正表示 B 更好。

### 11.2 五种子聚合

```powershell
python -m prwarn.cli.aggregate_seed_runs `
  --metrics outputs\seed2025\evaluation\metrics.json outputs\seed2026\evaluation\metrics.json outputs\seed2027\evaluation\metrics.json outputs\seed2028\evaluation\metrics.json outputs\seed2029\evaluation\metrics.json `
  --output outputs\summaries\five_seed_metrics.json
```

它只聚合所有 JSON 中路径一致的数值叶子，输出 count/mean/std 等摘要。输入文件必须唯一，输出存在时拒绝覆盖。

### 11.3 GPU 推理基准

```powershell
python -m prwarn.cli.benchmark_inference `
  --data-dir data\processed\sdwpf_v3_2_history_only `
  --scenario-run outputs\main_cfm_s2025 `
  --output outputs\main_cfm_s2025\benchmark_gpu.json `
  --device cuda `
  --batch-size 1 `
  --scenarios 100 `
  --sampling-steps 8 `
  --warmup 10 `
  --repetitions 100
```

报告确定性/场景模型参数量、P50/P95/P99/mean latency、origin/s throughput 和 peak VRAM。只应在声明的 GPU、固定 batch/M/steps 上横向比较。

## 12. Gate 6：A9 缺失压力、A10 风险组合与 A11 天气协议

### 12.1 A9 三个结构变体

```powershell
python -m prwarn.cli.train_deterministic --data-dir data\processed\sdwpf_v3_2_history_only --experiment-id a9_x_only_s2025 --missing-inputs --seed 2025 --device cuda
python -m prwarn.cli.train_deterministic --data-dir data\processed\sdwpf_v3_2_history_only --experiment-id a9_x_mask_s2025 --missing-inputs mask --seed 2025 --device cuda
python -m prwarn.cli.train_deterministic --data-dir data\processed\sdwpf_v3_2_history_only --experiment-id a9_x_mask_delta_s2025 --missing-inputs mask delta_t --seed 2025 --device cuda
```

确定性压力评估：

```powershell
python -m prwarn.cli.evaluate_missing_stress `
  --data-dir data\processed\sdwpf_v3_2_history_only `
  --run x_only=outputs\a9_x_only_s2025 `
  --run x_mask=outputs\a9_x_mask_s2025 `
  --run x_mask_delta=outputs\a9_x_mask_delta_s2025 `
  --output outputs\comparisons\a9_deterministic_s2025.json `
  --device cuda
```

默认压力包括：MCAR 10/30/50%、时间块 3/6/12 步、空间掉线 10/30% 节点、Train 95% 风速阈值定义的 extreme-conditioned missingness。选择由 seed 固定，三模型共享同一缺失位置。

代码只从仍可见历史做因果 forward fill，并重新构造 `P_pc/current_y/wind_from/delta_t`；不会用未来标签修补起报信息。

概率压力评估必须重生成受压场景，不能复用 clean scenarios：

```powershell
python -m prwarn.cli.evaluate_probabilistic_stress `
  --data-dir data\processed\sdwpf_v3_2_history_only `
  --run direct_cfm=outputs\main_cfm_s2025 `
  --run cvae=outputs\g4_cvae_s2025 `
  --run ddim=outputs\g5_ddim_s2025 `
  --run gaussian_copula=outputs\joint_baseline_s2025_gaussian_copula `
  --output outputs\comparisons\a9_probabilistic_s2025.json `
  --scenarios 100 `
  --device cuda
```

clean/stressed 使用 common random numbers，报告 point RMSE、turbine/farm CRPS、farm ES/VS、退化量及配对 HAC/DM 与 day-cluster bootstrap。

### 12.2 A10 风险代理组合

先构建 Calib/Test 输入：

```powershell
python -m prwarn.cli.build_risk_ablation_inputs `
  --scenario-run outputs\main_cfm_s2025 `
  --deterministic-run outputs\main_det_s2025 `
  --processed-data data\processed\sdwpf_v3_2_history_only `
  --threshold-fraction 0.10 `
  --duration-steps 1 `
  --output outputs\a10\risk_inputs_s2025.npz
```

该命令在 Train hidden 上拟合 Mahalanobis OOD，生成 Calib/Test 的：真实 ramp event、场景核心概率、OOD 分数、`1-history observed fraction` 数据质量分数与 origin time。

再在 Calib 拟合归一化和成本阈值，冻结到 Test：

```powershell
python -m prwarn.cli.evaluate_risk_ablation `
  --input outputs\a10\risk_inputs_s2025.npz `
  --config configs\sdwpf_v3_2.yaml `
  --output outputs\a10\metrics_s2025.json `
  --auxiliary-terms ood data_quality
```

输出 core vs augmented 的 AUPRC、Brier、stylized cost 和 Calib-fitted threshold。OOD/data quality 是诊断/敏感性代理，不是因果变量，也不是实际电网安全状态。

### 12.3 A11 issue-time 天气协议

主论文默认仍是 `history_only`。只有天气档案同时包含：

```text
TurbID, valid_time, issue_time, <weather features...>
```

且每条被选 forecast 满足 `issue_time <= origin_time`，才可以称为可部署 future forecast。

先审计已对齐的天气表：

```powershell
python -m prwarn.cli.audit_weather_archive `
  --input data\raw\weather\aligned_forecast.parquet `
  --protocol issue_time_forecast `
  --origin-column origin_time `
  --valid-column valid_time `
  --issue-column issue_time `
  --output outputs\audit\issue_time_weather.json
```

再附加到四个 processed split：

```powershell
python -m prwarn.cli.attach_future_weather `
  --input-dir data\processed\sdwpf_v3_2_history_only `
  --weather data\raw\weather\forecast_archive.parquet `
  --output-dir data\processed\sdwpf_v3_2_issue_time `
  --protocol issue_time_forecast `
  --features wind_speed temperature pressure `
  --turbine-column TurbID `
  --valid-column valid_time `
  --issue-column issue_time `
  --minimum-coverage 0.95
```

输出 NPZ 新增：

- `future_weather/future_weather_mask`：`[S,N,H,D_w]`；
- Train-only weather mean/std；
- 每 split coverage、选取与泄漏审计；
- 天气档案 SHA-256 和协议标签。

`issue_time_forecast` 对每个 origin/valid time/风机选择“不晚于 origin 的最新 issue”。数值缺失不能用未来 issue 填充。`oracle_era5` 使用相同张量路径，但必须标记为 non-deployable oracle upper bound。

当前仓库没有真实 issue-time forecast archive，因此 A11 真实运行是明确的外部数据 blocker；不得用未来 ERA5 冒充在线 NWP。

## 13. 实验矩阵 A0–A11、B0–B9、G1–G6

生成 230 个五种子计划合同：

```powershell
python -m prwarn.cli.make_experiment_matrix `
  --output outputs\plans\v3_2_matrix.json `
  --seeds 2025 2026 2027 2028 2029
```

也可仅生成部分 group：

```powershell
python -m prwarn.cli.make_experiment_matrix --output outputs\plans\joint_only.json --groups joint_probability forecast_baseline
```

group 可选值：`ablation`、`ablation_optional`、`protocol`、`joint_probability`、`forecast_baseline`。

物化为不可变 config 与 command manifest：

```powershell
python -m prwarn.cli.materialize_experiments `
  --matrix outputs\plans\v3_2_matrix.json `
  --base-config configs\sdwpf_v3_2.yaml `
  --output-dir outputs\plans\materialized_v3_2
```

可用多个 `--logical-id A0 --logical-id G6` 只物化部分实验。`manifest.json` 中的命令仍有 `<PROCESSED_DATA_DIR>`、`<DETERMINISTIC_RUN>`、`<SCENARIO_RUN>`、`<WEATHER_ARCHIVE>` 等占位符；物化表示冻结“运行合同”，不是已经执行。所有 job 初始状态均为 `planned_not_run`。

矩阵含义：

| ID | 比较 |
|---|---|
| A0 | direct power vs `P_pc + residual` |
| A1 | density correction off/on |
| A2 | geo → +corr → +directional → +adaptive |
| A3 | raw correlation vs difference correlation |
| A4 | Direct CFM vs VAE-CFM |
| A5 | independent CFM vs genuine minibatch OT-CFM |
| A6 | Euler 4/8/16/32 steps |
| A7 | static conformal vs ACI |
| A8 | context/OOD fallback off/on |
| A9 | X-only vs X+M vs X+M+delta_t |
| A10 | core risk vs +OOD/+data quality |
| A11 | history-only vs oracle ERA5 vs issue-time forecast |
| B0–B1 | persistence、物理功率曲线 |
| B2–B3 | GRU、TCN |
| B4–B6 | Gaussian、Student-t、quantile+CQR |
| B7 | deep ensemble |
| B8–B9 | Graph WaveNet-style、AGCRN-style |
| G1–G3 | Gaussian residual、bootstrap、Gaussian copula |
| G4–G6 | CVAE、DDIM、Direct CFM |

## 14. 24 个命令行文件逐一说明

| 文件/命令 | 功能 | 主要输入 | 主要输出 | 环境 |
|---|---|---|---|---|
| `audit_sdwpf.py` | 原始 SDWPF schema/重复/网格/reason code/SHA 审计 | 原始 CSV/Parquet | audit JSON | CPU/全量数据 |
| `preprocess_sdwpf.py` | 刚性网格、Train-only scaler/curve/graphs、四 split 窗口 | 主表、坐标、单位、额定功率 | NPZ、NPY、metadata | 大内存 CPU |
| `audit_weather_archive.py` | A11 history/oracle/issue-time 可用性与泄漏审计 | 已对齐天气表 | audit JSON | CPU |
| `attach_future_weather.py` | 按 valid/issue 对齐未来天气并 Train-only 标准化 | processed dir、天气档案 | 带 future weather 的新 processed dir | CPU/大内存 |
| `smoke_numpy.py` | 数据/物理/图/校准/风险的合成 CPU 冒烟 | 无 | stdout JSON | 本地 |
| `smoke_torch.py` | 神经确定性→CFM→场景的合成张量冒烟 | `--device` | stdout 摘要 | CUDA 服务器 |
| `train_deterministic.py` | 训练主多图或 GRU/TCN/GWN/AGCRN 点模型并缓存四 split | processed dir、YAML | checkpoint、HDF5、metrics | GPU |
| `evaluate_point_references.py` | 同 cache 评估 persistence 与 `P_pc` | deterministic run | JSON | CPU/HDF5 |
| `train_flow.py` | 训练 Direct CFM、反向 ODE 导出 Calib/Test 场景 | deterministic run | checkpoint、scenario HDF5 | GPU |
| `fit_probabilistic_baselines.py` | 拟合 Gaussian/bootstrap/copula residual 基线 | deterministic run | state NPZ、scenario HDF5 | CPU/内存 |
| `train_neural_baseline.py` | 训练 CVAE、DDIM、VAE-CFM | deterministic run | checkpoint、scenario HDF5 | GPU |
| `train_marginal_baseline.py` | 训练 Gaussian/Student-t/quantile 边际模型 | deterministic run | checkpoint、marginal HDF5 | GPU |
| `evaluate_marginal.py` | Calib pooled per-horizon CQR、Test 指标 | marginal run | calibrated HDF5、metrics | CPU/HDF5 |
| `build_deep_ensemble.py` | 合并多个确定性 cache 为场景 ensemble | 2+ deterministic runs | scenario HDF5 | CPU/HDF5 |
| `evaluate.py` | static/ACI/context 校准和完整 Test 概率/区间/风险评估 | scenario run | calibrated HDF5、metrics | CPU/HDF5 |
| `compare_scenario_runs.py` | 两 run 配对 CRPS/ES/VS、HAC/DM、cluster/event bootstrap | 两个 scenario runs | comparison JSON | CPU |
| `aggregate_seed_runs.py` | 聚合多个种子相同 JSON 数值路径 | metrics JSON 列表 | summary JSON | CPU |
| `benchmark_inference.py` | 端到端确定性+场景 latency/throughput/VRAM | processed dir、scenario run | benchmark JSON | 指定 GPU |
| `evaluate_missing_stress.py` | A9 确定性模型共享缺失选择的鲁棒性 | processed dir、多 deterministic runs | stress JSON | GPU |
| `evaluate_probabilistic_stress.py` | A9 受压条件下重生成场景并配对退化 | processed dir、多 scenario runs | stress/statistics JSON | GPU/耗时高 |
| `build_risk_ablation_inputs.py` | 从 frozen artifacts 生成 A10 Calib/Test 风险数组 | scenario/deterministic/processed | NPZ+metadata JSON | CPU/HDF5 |
| `evaluate_risk_ablation.py` | Calib-fitted/Test-frozen 风险组合和成本阈值 | A10 NPZ、YAML | metrics JSON | CPU |
| `make_experiment_matrix.py` | 生成 seed-expanded 计划作业 | seed/groups | matrix JSON | 本地 |
| `materialize_experiments.py` | 应用 dotted overrides，生成不可变 configs/命令清单 | matrix、base YAML | configs、manifest | 本地 |

所有命令的当前参数可随时查看：

```powershell
python -m prwarn.cli.<命令名> --help
```

## 15. 核心源码文件逐一说明

### 15.1 `data/`

| 文件 | 关键对象 | 功能 |
|---|---|---|
| `audit.py` | `audit_sdwpf_frame` | 原始表 Gate-1 审计，复用 official-style reason code |
| `grid.py` | `ReasonCode`、`build_rigid_grid`、`sdwpf_reason_codes`、`build_sdwpf_bundle`、`bundle_to_time_node` | 构造完整时间×节点载体，区分 missing/unknown/abnormal，产生 fill/mask/delta_t |
| `imputation.py` | `impute_history` | mean/forward/linear/GRU-D-style 的因果、mask-preserving 历史插补 |
| `split.py` | `chronological_split` | 时间不交叉的 60/15/10/15 切分及边界清单 |
| `window.py` | `make_windows` | 从 `[T,N,D]` 生成历史和未来窗口，保证未来 offset 正确 |
| `processed.py` | `ProcessedSplit`、`load_processed_split`、`origin_wind_from_degrees` | NPZ 数据合同、shape/mask 验证、风向语义 |
| `torch_dataset.py` | `WindowTensorDataset`、`DirectionalGraphCollator`、`DeterministicCacheDataset`、`FlowGraphCollator` | 把 NPZ/HDF5 接到 PyTorch，并在 batch 时构造方向/混合图 |
| `stress.py` | 四种 missingness、`apply_missingness_stress`、`rebuild_issue_time_derived` | A9 固定缺失选择、因果填充、受压起报量重建 |
| `weather_protocol.py` | `audit_weather_availability`、`select_latest_available_issue` | A11 availability contract 和 `issue<=origin` 选择 |

### 15.2 `physics/` 与 `graphs/`

| 文件 | 关键对象 | 功能 |
|---|---|---|
| `physics/density.py` | `air_density`、`equivalent_wind_speed` | `rho=p/(R_dT)` 与密度修正等效风速 |
| `physics/power_curve.py` | `EmpiricalPowerCurve` | Train-only 分箱、加权 PAVA 单调约束、插值、state 序列化 |
| `graphs/builders.py` | `geographic_graph`、`correlation_graph`、`directional_graph` | 构建 row-normalized 地理/Train 相关/动态风向图，支持 top-k |

### 15.3 `models/` 与 `baselines/`

| 文件 | 关键对象 | 功能 |
|---|---|---|
| `models/backbone.py` | `AdaptiveAdjacency`、`GraphResidualBlock`、`DynamicMultiGraphResidualForecaster`、`masked_huber_loss` | 主多图确定性残差模型、图混合门控、物理/直预测模式 |
| `models/deterministic_baselines.py` | `TemporalDeterministicBaseline` | 共享 cache API 的 GRU、TCN、Graph-WaveNet-style、AGCRN-style |
| `models/flow.py` | `ErrorStandardizer`、`DirectCFMVelocity`、`conditional_flow_matching_loss`、`sample_reverse_ode` | 最终误差上的 Direct CFM、independent/OT coupling、Euler 采样 |
| `models/generative_baselines.py` | `GraphConditionalVAE`、`GraphConditionalDiffusion`、`GraphVaeCFM` 及 loss/sample | CVAE、VP diffusion/DDIM、VAE-CFM 联合残差场景 |
| `models/marginal_baselines.py` | `GraphMarginalForecaster`、`marginal_loss`、`sample_marginal_errors` | Gaussian/Student-t/quantile 边际头 |
| `models/scenarios.py` | `reconstruct_power_scenarios` | 反标准化误差、加回 `Y_det`、按额定功率裁剪 |
| `models/scenario_checkpoint.py` | `LoadedScenarioCheckpoint`、两个 loader | 统一加载 residual/CVAE/DDIM/CFM/VAE-CFM checkpoint 并在压力/benchmark 中采样 |
| `baselines/probabilistic.py` | `GaussianResidual`、`ResidualBootstrap`、`GaussianCopulaResidual`、save/load | Train residual 强基线与无 pickle 状态持久化 |

### 15.4 `calibration/`、`risk/`、`eval/`

| 文件 | 关键对象 | 功能 |
|---|---|---|
| `calibration/split.py` | `PerHorizonSplitConformal`、`scenario_interval` | 有限样本逐 horizon split conformal |
| `calibration/adaptive.py` | `AdaptiveConformalIntervals`、`ContextFallbackConformal` | chronological ACI、context kernel 与 OOD 保守回退 |
| `risk/proxies.py` | ramp/deviation/CVaR、`MahalanobisOOD`、`FrozenRiskComposer`、cost threshold | 连续 wind-side 风险代理、Calib-fitted 风险组合 |
| `eval/metrics.py` | MAE/RMSE/R²、pinball、CRPS、ES、VS、Brier/BSS、AP、reliability、interval | 点、边际、联合、事件和区间指标 |
| `eval/statistics.py` | Newey-West、DM、cluster bootstrap、event IDs、seed aggregation | 对时间依赖和稀有事件友好的统计比较 |

### 15.5 `experiments/` 与训练工具

| 文件 | 关键对象 | 功能 |
|---|---|---|
| `experiments/matrix.py` | `ExperimentSpec`、`experiment_specs`、`build_experiment_matrix` | A/B/G 规范、稳定哈希 ID、种子展开 |
| `experiments/resolve.py` | `apply_dotted_overrides`、`execution_stage`、command/post templates | 把逻辑实验映射到不可变配置与命令占位符 |
| `training.py` | seed、point/flow epoch、config hash、runtime capture、run init | 训练循环、可复现元数据和不覆盖 run 目录 |
| 各目录 `__init__.py` | public imports / package marker | 组织包级 API；无独立训练逻辑 |

## 16. 测试文件分别保护什么

| 测试文件 | 保护的合同 |
|---|---|
| `test_adaptive_calibration.py` | ACI predict-before-update、无效标签不更新、OOD fallback |
| `test_aggregate_seed_cli.py` | 多种子嵌套指标聚合与文件输出 |
| `test_audit.py` | 重复键阻断和网格缺口可见性 |
| `test_calibration_risk_eval.py` | 区间只扩不缩、ramp/CVaR/OOD、冻结风险组合 |
| `test_data.py` | split 不交叉、history-only 不读未来、reason code、窗口 offset |
| `test_experiment_matrix.py` | A/B/G ID 完整、稳定唯一、无结果冒充 |
| `test_experiment_resolve.py` | dotted override、stage/command materialization |
| `test_missingness_stress.py` | MCAR/块/空间/极端缺失、因果 forward fill、起报量重建 |
| `test_physics_graphs.py` | 密度/等效风速、图归一化、单调功率曲线 roundtrip |
| `test_preprocess_cli.py` | 合成数据上的完整 history-only archive |
| `test_probabilistic_baselines.py` | Gaussian/bootstrap/copula shape、依赖与 state roundtrip |
| `test_processed.py` | NPZ shape、二值 mask、风向语义 |
| `test_statistics.py` | Newey-West、DM、cluster/event bootstrap、seed summary |
| `test_weather_imputation.py` | 插补因果性、issue-time 泄漏门和 latest-available issue |

## 17. 常见错误与处理

### `No module named prwarn`

执行 `python -m pip install -e ".[data,dev]"`，或把 `src` 放入 `PYTHONPATH`。确认当前工作目录是仓库根目录。

### 读取 Parquet 报缺少 engine

安装 data extra：

```powershell
python -m pip install -e ".[data]"
```

并检查 `python -c "import pyarrow; print(pyarrow.__version__)"`。

### `coordinates.npy` / `a_geo.npy` 不存在

重新预处理并传入 `--locations`，确认坐标的 `TurbID/x/y` 列名；当前训练链实际需要这两个文件。

### `output already exists` 或 `FileExistsError`

这是防止覆盖论文证据的设计。使用新的 `--experiment-id` / `--output-dir`；不要删除旧目录后复用同名来掩盖失败 run。

### CUDA 不可用或 device 错误

```powershell
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

若为 False，重新安装与驱动匹配的 PyTorch CUDA wheel，而不是把最终论文 run 静默改成 CPU。

### OOM

依次减小 `--batch-size`、`--scenarios`；DDIM/CFM 可降低 sampling steps 做 sanity run。不要把降低后的设置与主论文设置混在同一 experiment ID。默认不要加 `--save-full-scenarios`。

### 功率曲线异常或预测大量裁剪到 0/额定功率

优先检查 `Patv` 与 `--rated-power` 单位、`Sp/T2m` 单位、有效 mask、风速列、density correction。查看 `metadata.json` 的 power curve bins 和 Train scaler。

### weather coverage 低或 leakage audit 失败

确认天气表的 `valid_time`、`issue_time` 都有时区并与 origin 对齐；每个风机/valid time 可能有多个 issue，代码会选 `issue<=origin` 的最新一条。不要降低 coverage 阈值来隐藏系统性缺档。

### 比较两个场景 run 失败

检查两者是否来自相同 Test split、节点顺序、origin time、truth/mask 和场景数。不同信息协议的模型不能在未说明的情况下当作同信息公平比较。

### HDF5 多 worker 问题

先使用默认 `--num-workers 0` 完成 sanity run。确认文件可正常关闭后再逐步增加 worker；网络文件系统上同时打开大量 HDF5 可能变慢或失败。

## 18. 建议的实际执行顺序

### 阶段 A：本地完成

```text
[x] 安装或临时设置 PYTHONPATH
[x] 52 tests
[x] smoke_numpy
[x] compileall
[x] 生成并检查 230-job matrix
```

### 阶段 B：服务器一次性 sanity run

```text
[ ] smoke_torch --device cuda
[ ] 下载/冻结 SDWPF V2 和坐标 SHA-256
[ ] audit_sdwpf
[ ] preprocess_sdwpf
[ ] 检查 metadata、四 split、曲线和图
[ ] train_deterministic，先 1 seed/少量 epoch
[ ] train_flow，先少量 epoch/M/steps
[ ] evaluate static conformal
[ ] 检查所有 HDF5 能 roundtrip
```

### 阶段 C：冻结正式实验

```text
[ ] 固定配置、阈值、seed 和 GPU
[ ] 五种子 point baselines 与主中心
[ ] G1–G6、B4–B7
[ ] A0–A10；A11 等真实天气档案
[ ] paired HAC/DM + day/event cluster bootstrap
[ ] five-seed aggregation
[ ] P50/P95/P99/throughput/peak-VRAM
[ ] 归档 stdout/stderr、config hash、git commit、dataset hash
```

每个正式 run 最少保留：

```text
config.json + config hash
runtime.json + git commit/GPU/CUDA
dataset version + SHA-256
split boundaries + weather protocol
best checkpoint
predictions/scenarios
metrics/statistics/benchmark JSON
stdout/stderr log
```

## 19. 仍必须由服务器或外部数据给出的验证

1. CUDA 上确定性、边际、Direct CFM、VAE-CFM、CVAE、DDIM 的 forward/backward 和采样。
2. 全量 SDWPF 的 schema、重复、坐标、单位、额定功率、reason-code 数量和物理曲线诊断。
3. 四 split HDF5 cache、Calib/Test 场景导出和 checkpoint roundtrip。
4. A9 clean/stressed 场景重生成及共享缺失选择。
5. 五种子 A0–A10、B0–B9、G1–G6 的实际指标与统计显著性。
6. 真 issue-time weather archive 到位后的 A11；在此之前只能验证代码与泄漏门。
7. 声明 GPU 上的参数量、峰值显存、吞吐和 P50/P95/P99。

在这些步骤完成前，代码产物应表述为“implementation/code-complete”或“planned_not_run”，不能表述为论文已验证结论。

## 20. 结果解释边界

仓库输出的是风侧的 ramp/deviation/tail-risk/OOD/data-quality 代理。没有电网拓扑、负荷、调度、潮流和动态安全模型时，不能把它解释为真实线路过载、电压/频率稳定性、真实备用充足性或实际调度成本。
