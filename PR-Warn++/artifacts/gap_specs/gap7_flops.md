# 缺口 7：计算成本测量（FLOPs / 训练时间）

> 对应约束：C3-6（计算成本：Big-O 复杂度 + 统一硬件实测，含参数量/FLOPs/时延/吞吐/训练时间/峰值内存，注明 warm-up 与计时协议）
> 优先级：**P1**
> 版本：v1.0 · 2026-10-10

---

## 1. 缺口定位

### 现状

| 已有能力 | 文件位置 | 说明 |
|---|---|---|
| 参数量统计 | `src/prwarn/cli/benchmark_inference.py:21` `_parameter_count(model)` | 对 `torch.nn.Module` 逐参数 `numel()` 求和，返回 int |
| 推理时延测量 | `src/prwarn/cli/benchmark_inference.py:62-85` `execute(seed)` + 循环 | warm-up 10 次后重复 100 次，`perf_counter()` 计时，输出 p50/p95/p99/mean |
| 吞吐量 | `benchmark_inference.py:109-111` | `batch_size / (mean_latency_ms / 1000)` |
| 峰值显存 | `benchmark_inference.py:87-89, 112` | `torch.cuda.max_memory_allocated()`，仅 CUDA 设备 |
| 训练时间指标声明 | `src/prwarn/experiments/matrix.py:81` A5 spec | `primary_metrics` 含 `"training_time"`，但无任何训练循环计时实现 |
| latency / peak_vram 指标声明 | `matrix.py:72` A4 spec | `primary_metrics` 含 `"latency", "peak_vram"`，由 benchmark_inference.py 部分覆盖 |

### 缺口是什么

1. **缺 FLOPs 统计**：`benchmark_inference.py` 已报参数量/时延/吞吐/峰值显存，但**无 FLOPs（或 MACs）计数**。C3-6 明确要求"参数量/FLOPs/时延/吞吐/训练时间/峰值内存"六项，当前缺 FLOPs 一项。
2. **缺训练时间记录**：`matrix.py:81` A5 spec 声明了 `training_time` 指标，但 `src/prwarn/cli/train_deterministic.py` 等训练入口无任何墙钟时间（wall-clock）打点，产物（h5/json）中无 `training_time_seconds` 字段。
3. **缺计时协议元数据**：虽然 benchmark_inference.py 记录了 `warmup`/`repetitions`/`device`/`batch_size`，但未记录 CUDA 内核编译、cudnn benchmark 模式、CPU 频率锁定等环境信息，复现性不足。

---

## 2. 文献依据表

| 文献（作者-年份） | 出处/年份 | DOI/arXiv | 核验状态 | 关键做法 | 可借鉴的协议要素 |
|---|---|---|---|---|---|
| Tan & Le-2019 | ICML 2019 | arXiv:1905.11946 | VERIFIED（arXiv abs 页核验：标题/作者/ICML 2019 批注一致） | EfficientNet 系列逐模型报告 params（M）与 FLOPs（B），并对比推理加速比（× faster）；compound scaling 公式中 FLOPs 随 depth/width/resolution 缩放 | ① params + FLOPs + 推理时延三件套同时报告；② FLOPs 以 B（十亿）为单位；③ 明确区分"参数量"与"计算量"两个维度 |
| Ma, Zhang, Zheng & Sun-2018 | ECCV 2018 | arXiv:1807.11164 | VERIFIED（arXiv abs 页 + ar5iv HTML 核验：作者 Ningning Ma / Xiangyu Zhang / Hai-Tao Zheng / Jian Sun，ECCV 2018 一致） | 论证 FLOPs 是**间接指标**，实际速度还受 memory access cost (MAC)、平台特性、并行度影响；在目标平台上实测直接指标（speed）；提出四条设计准则（MAC 约束、分组卷积开销、网络碎片化、element-wise 操作） | ① FLOPs 必须配合实测时延报告，不能只报 FLOPs；② 计时需在目标硬件上进行，注明平台（CPU/GPU 型号、batch size）；③ 区分 FLOPs（理论计算量）与实际 latency（端到端墙钟时间） |
| Strubell, Ganesh & McCallum-2019 | ACL 2019 | arXiv:1906.02243 | VERIFIED（arXiv abs 页核验：标题/作者/ACL 2019 批注一致） | 量化训练一个 NLP 模型的财务成本与碳足迹；统计 GPU 小时数（training hours）、总能耗（kWh）、CO₂ 排放量；区分单次训练成本与完整研发（超参搜索+多次实验）成本 | ① 训练时间以 GPU 小时（GPU-hours）或墙钟秒报告，注明 GPU 型号与数量；② 记录完整训练流程的总时间（非单 epoch）；③ 报告"总研发成本"而非仅最终模型训练成本 |

---

## 3. 标准做法要点

1. **FLOPs 统计口径**：采用 **MACs（Multiply-Accumulate operations）** 作为 FLOPs 的统计基础——即一次乘加计为 2 FLOPs（或在论文中明确声明 MACs ≈ FLOPs 除以 2 的惯例）。EfficientNet 论文中报告的 "FLOPS" 实际为 MACs 计数（Tan & Le 2019）。需在产物中明确标注计数口径（`flops_type: "macs"` 或 `flops_type: "floating_point_ops"`）。
2. **三件套同时报告**：每个模型变体必须同时报告 ①参数量（M）、②FLOPs/MACs（单次前向，B 或 G）、③实测推理时延（ms，p50/p95）。Ma et al. (2018) 证明 FLOPs 与实际速度不等价，二者缺一不可。
3. **FLOPs 测量工具**：使用 `thop`（即 `ptflops`）或 `fvcore.nn.FlopCountAnalysis` 对模型前向传播做钩子式（hook-based）统计；输入形状必须与实际推理 batch size 一致（本代码库 batch_size=1）。手动计数仅用于交叉验证。
4. **训练时间记录机制**：在训练循环首尾用 `time.perf_counter()` 打点，记录完整训练流程（含所有 epoch、早停触发时间）的总墙钟秒数；同时记录 GPU 型号与数量，换算为 GPU-hours。Strubell et al. (2019) 建议报告完整研发成本而非仅最终训练。
5. **计时协议元数据**：benchmark 产物中必须记录 warm-up 次数、重复次数、batch size、device 类型、CUDA/cudnn 版本（如有），确保时延数字可复现。
6. **Big-O 复杂度单独给出**：C3-6 要求"Big-O 复杂度（稀疏/稠密实现分开给出）"——在论文中对注意力机制（O(L²)）、图传播（O(N²)）等给出理论复杂度，与实测 FLOPs 互补。

---

## 4. 实现规范（映射到本代码库）

| 改动点 | 落点文件/函数/配置键 | 新增参数与默认值 | 协议要点 | 验收方式 |
|---|---|---|---|---|
| **FLOPs 统计函数** | `src/prwarn/cli/benchmark_inference.py` 新增 `_flops_count(model, sample_inputs) -> dict` | 依赖：`thop`（`pip install thop`）或 `fvcore`；推荐 `thop`（轻量、API 简单） | ① 对 `deterministic` 模型和 `scenario.model` 分别统计；② 输入形状用 `batch_size=1` 的实际 sample；③ 返回 `{"macs": int, "params": int, "flops_type": "macs"}`；④ 注意：`thop` 对自定义 `einsum` 图传播算子可能漏计，需手动补写 GraphPropagation 的 MACs = batch × nodes² × hidden_dim | 运行 `benchmark_inference.py` 后 JSON 产物中出现 `flops` 字段，值 > 0 且与参数量量级合理（如 hidden_dim=64、history=24、n_nodes=10 时 MACs 应在 10⁶–10⁸ 量级） |
| **FLOPs 写入产物** | `benchmark_inference.py:90-113` `payload` dict | 新增字段：`"flops": {"deterministic": _flops_count(...), "scenario": _flops_count(...), "unit": "macs", "batch_size": args.batch_size, "input_shape": str}` | ① 与现有 `parameters` 字段并列；② 记录统计时的输入形状（[B,N,L,D]），便于复算；③ CPU 上统计 FLOPs（不依赖 CUDA） | 产物 JSON 中 `flops.deterministic.macs` 为正整数，`flops.unit == "macs"` |
| **训练时间打点** | `src/prwarn/cli/train_deterministic.py`（训练入口）在 `main()` 首尾加 `perf_counter()` | 新增配置键：`benchmark.training_time: true`（默认 `true`）；新增产物字段：`training_time_seconds: float`、`gpu_model: str`（如 `torch.cuda.get_device_name()`）、`gpu_count: int` | ① 计时范围 = 完整训练流程（含数据加载预热除外）；② 用 `perf_counter()` 而非 `time.time()`；③ 早停触发时停止计时；④ 训练完成后写入 `config.json` 或独立 `train_metadata.json` | 训练产物目录下存在 `train_metadata.json`，含 `training_time_seconds`（>0）、`gpu_model`（字符串）、`gpu_count`（≥1） |
| **训练时间纳入矩阵指标** | `src/prwarn/experiments/matrix.py:81` A5 spec | 无新增配置键；确认 `primary_metrics` 中 `"training_time"` 与训练产物字段对齐 | ① A5 spec 已有 `training_time`，需确保 `aggregate_seed_runs.py` 能从 `train_metadata.json` 读取该字段并聚合 mean±SD；② 所有可训练基线（B2/B3/B8/B9 及新增 Transformer 类）都自动继承此打点 | `aggregate_seed_runs.py` 输出中含 `training_time` 的 mean±SD 行 |
| **Benchmark 配置键** | `configs/sdwpf_v3_2.yaml` 新增 `benchmark:` 段 | `benchmark.flops: true`（默认）、`benchmark.training_time: true`（默认）、`benchmark.warmup: 10`、`benchmark.repetitions: 100` | ① 将 CLI 参数默认值收敛到 config，避免 CLI 与 config 不一致；② `flops: true` 时启用 thop 统计 | yaml 中 `benchmark.flops: true` 被 `benchmark_inference.py` 读取 |
| **环境元数据记录** | `benchmark_inference.py:90-113` `payload` dict | 新增字段：`"torch_version": torch.__version__`、`"cuda_version": torch.version.cuda or "cpu"`、`"cudnn_version": torch.backends.cudnn.version() or null` | ① 确保时延数字可复现；② CPU 模式下 cuda/cudnn 字段为 null | 产物 JSON 中 `torch_version` 字段存在且为字符串 |

---

## 5. 验收自检清单

- [ ] `benchmark_inference.py` 运行后产物 JSON 中包含 `flops.deterministic.macs`（正整数）与 `flops.scenario.macs`（正整数）
- [ ] 产物中 `flops.unit == "macs"`，且 `flops.input_shape` 记录了实际张量形状
- [ ] 训练完成后 `train_metadata.json` 中 `training_time_seconds > 0`
- [ ] `train_metadata.json` 中 `gpu_model` 与 `gpu_count` 字段存在
- [ ] 产物 JSON 中 `torch_version`、`cuda_version` 字段存在
- [ ] `configs/sdwpf_v3_2.yaml` 中 `benchmark.flops: true` 与 `benchmark.training_time: true`
- [ ] A5 spec 的 `training_time` 指标能从训练产物中读取并聚合 mean±SD
- [ ] 论文方法节给出各模型 Big-O 复杂度表（注意力 O(L²)、图传播 O(N²)、GRU O(L·d²) 等），稀疏/稠密分开标注
- [ ] thop 依赖已加入 `requirements.txt` 或 `pyproject.toml`

---

## 6. 引用来源列表

1. Tan, M. & Le, Q. V. EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. *Proceedings of the 36th International Conference on Machine Learning (ICML)*, 2019. arXiv:1905.11946. URL: https://arxiv.org/abs/1905.11946
2. Ma, N., Zhang, X., Zheng, H.-T. & Sun, J. ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design. *Proceedings of the European Conference on Computer Vision (ECCV)*, pp. 116–131, 2018. arXiv:1807.11164. URL: https://arxiv.org/abs/1807.11164
3. Strubell, E., Ganesh, A. & McCallum, A. Energy and Policy Considerations for Deep Learning in NLP. *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (ACL)*, 2019. arXiv:1906.02243. URL: https://arxiv.org/abs/1906.02243
