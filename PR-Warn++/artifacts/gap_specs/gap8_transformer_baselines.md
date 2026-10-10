# 缺口 8：SOTA 对比基线补充（Transformer / PatchTST / iTransformer 等）

> 对应约束：C5-5（与最相近 SOTA 方法的四维对比表：指导如何进入预测目标 / 空间表示 / 时序骨干 / 可 claim 的差异化范围）
> 优先级：**P2**
> 版本：v1.0 · 2026-10-10

---

## 1. 缺口定位

### 现状

| 已有能力 | 文件位置 | 说明 |
|---|---|---|
| 确定性基线框架 | `src/prwarn/models/deterministic_baselines.py:21` `TemporalDeterministicBaseline` | 统一接口：`forward(x_fill, mask, delta_t, static_graphs, directional_graph, p_pc_future, future_weather, future_weather_mask) -> dict{residual, y_det, hidden, adjacency, graph_weights}` |
| 已支持架构 | `deterministic_baselines.py:46` | `{"gru", "tcn", "graphwavenet", "agcrn"}` 四种风格；`__init__` 按 architecture 分支构建 encoder |
| 编码路径 | `deterministic_baselines.py:117` `_encode(model_input)` | GRU 取最后隐藏态；TCN 取序列最后时间步；GraphWaveNet 门控因果卷积；AGCRN 图循环 |
| 基线矩阵规格 | `src/prwarn/experiments/matrix.py:180-206` `_forecast_baseline_specs()` | B0 persistence / B1 p_pc / B2 gru / B3 tcn / B4-B6 marginal / B7 deep_ensemble / B8 graphwavenet / B9 agcrn |
| 配置覆盖键 | `matrix.py:187-197` | B2/B3 用 `{"model.architecture": "gru"/"tcn"}` 等键覆盖 config 默认值 |
| 配置默认值 | `configs/sdwpf_v3_2.yaml:45-53` model 段 | `architecture: multigraph`、`target_mode: physics_residual`、`hidden_dim: 64`、`temporal_kernel: 3`、`dropout: 0.1`、`missing_inputs: [mask, delta_t]` |
| 公平契约 | `docs/约束符合性核查报告.md:46` C3-5 | 所有基线消费同一 `deterministic_train.h5` 残差；同 alpha/掩码/单位；调参预算统一 |

### 缺口是什么

1. **缺 Transformer 类时序骨干基线**：当前 B2（GRU）、B3（TCN）为循环/卷积基线，B8/B9 为图时序基线，但**无任何 Transformer 系列对照**。C5-5 要求与 "physics-constrained Transformer、AGCRN、Graph WaveNet、MTGNN" 等 SOTA 对比——AGCRN/Graph WaveNet 已有，Transformer 类缺失。
2. **缺 PatchTST / iTransformer 等近年 SOTA**：ICLR 2023/2024 的 PatchTST 与 iTransformer 是当前时间序列预测领域最强 Transformer 基线，必须纳入对照以证明多图融合的增量价值。
3. **缺树模型基线**：XGBoost 是风电预测领域常用强基线，当前无任何树模型对照。

---

## 2. 文献依据表

| 文献（作者-年份） | 出处/年份 | DOI/arXiv | 核验状态 | 关键做法 | 可借鉴的协议要素 |
|---|---|---|---|---|---|
| Vaswani et al.-2017 | NeurIPS 2017 | arXiv:1706.03762 | VERIFIED（arXiv abs 页核验：作者 Vaswani/Shazeer/Parmar/Uszkoreit/Jones/Gomez/Kaiser/Polosukhin，标题 "Attention Is All You Need" 一致） | 纯注意力机制编码器-解码器；多头自注意力 O(L²)；位置编码；并行训练 | ① 作为 Transformer 基线的架构原型；② encoder-only 变体可直接用于时序预测；③ 位置编码（sinusoidal 或 learned）是必加组件 |
| Nie et al.-2023 | ICLR 2023 | arXiv:2211.14730 | VERIFIED（arXiv abs 页核验：作者 Yuqi Nie / Nam H. Nguyen / Phanwadee Sinthong / Jayant Kalagnanam，ICLR 2023 批注一致） | ① Patching：将时间序列切分为子序列 patch 作为 Transformer 输入 token；② Channel-independence：每个通道独立建模，共享 Transformer 权重；③ 注意力计算量随 patch 数二次方下降 | ① patch_len / stride 是核心超参；② channel-independence 意味着各涡轮机独立处理、共享权重；③ flatten patches 后接线性预测头 |
| Liu et al.-2024 | ICLR 2024 | arXiv:2310.06625 | VERIFIED（arXiv abs 页核验：作者 Yong Liu / Tengge Hu / Haoran Zhang / Haixu Wu / Shiyu Wang / Lintao Ma / Mingsheng Long，ICLR 2024 一致；OpenReview ID: JePfAI8fah） | ① Inverted：将每个变量（variate）的完整历史嵌入为一个 token，注意力跨变量计算而非跨时间步；② FFN 逐 token 处理；③ 线性投影到未来步 | ① 变量数 N_variates 决定注意力序列长度；② 与 PatchTST 的 channel-independence 形成对照（iTransformer 显式建模变量间相关性） |
| Chen & Guestrin-2016 | KDD 2016 | arXiv:1603.02754 | VERIFIED（arXiv abs 页核验：作者 Tianqi Chen / Carlos Guestrin，KDD'16 批注一致；DOI: 10.1145/2939672.2939785） | 梯度提升树系统；稀疏感知算法；加权分位数 sketch；并行训练 | ① 作为非深度学习强基线，证明深度模型的增量价值；② 特征工程：将 [L, D] 历史窗口展平为特征向量；③ 逐节点逐 horizon 训练独立模型 |

---

## 3. 标准做法要点

1. **Transformer 基线架构**：采用 encoder-only 结构（Vaswani et al. 2017 的编码器部分），输入为 [B·N, L, d_model]，经位置编码 + N 层 TransformerEncoderLayer，取最后时间步隐藏态接预测头。与原始 encoder-decoder 不同，时序预测中 encoder-only + 线性头已被证明足够（iTransformer 2024）。
2. **PatchTST 核心设计**（Nie et al. 2023）：
   - **Patching**：将长度为 L 的历史序列切分为 patch_len 长度、stride 步长的 patch；每个 patch 经线性投影为 d_model 维 token。
   - **Channel-independence**：每个特征通道（本代码库即每个涡轮机节点）独立建模，共享同一套 Transformer 权重——这与现有 `TemporalDeterministicBaseline._encode` 中 `model_input.reshape(batch * nodes, history, width)` 的处理方式一致。
   - **预测头**：将所有 patch 的输出 flatten 后接线性层，直接输出 forecast_steps 维。
3. **iTransformer 核心设计**（Liu et al. 2024）：
   - **Inverted embedding**：将每个 variate（即 D 个特征维度）的完整历史序列 [L] 嵌入为一个 d_model 维 token，得到 [B·N, D, d_model]。
   - **注意力跨变量**：自注意力在 D 个 variate token 之间计算，捕捉特征间相关性；FFN 逐 token 独立处理。
   - **预测头**：每个 variate token 经线性层映射到 forecast_steps 维。
4. **公平契约**（C3-5）：所有新增基线必须：
   - 消费同一 `deterministic_train.h5` 残差数据（`target_mode: physics_residual`，即预测 R = Y − P_pc）；
   - 接受同样的 mask / delta_t 缺失输入通道；
   - 使用同样的 `masked_huber_loss` 训练损失；
   - 超参搜索预算与 B2/B3（GRU/TCN）一致（如相同的搜索空间大小、相同的 GPU 小时数上限）；
   - 验证集选择超参，绝不触碰测试集。
5. **XGBoost 基线**（Chen & Guestrin 2016）：
   - 非神经网络，需独立训练路径：将每个节点的 [L, D] 历史窗口展平为特征向量，逐节点逐 horizon 训练 XGBoost 回归器。
   - 作为"经典机器学习上界"对照，证明深度时序模型的必要性。

---

## 4. 实现规范（映射到本代码库）

| 改动点 | 落点文件/函数/配置键 | 新增参数与默认值 | 协议要点 | 验收方式 |
|---|---|---|---|---|
| **Transformer encoder-only 分支** | `deterministic_baselines.py:46` `if architecture not in {...}` 集合加入 `"transformer"`；`__init__` 新增 `elif architecture == "transformer":` 分支 | `transformer_d_model: 64`、`transformer_nhead: 4`、`transformer_num_layers: 2`、`transformer_dim_feedforward: 128`、`transformer_dropout: 0.1`（均复用现有 `hidden_dim`/`dropout` 作为默认） | ① 输入投影：`nn.Linear(input_dim, d_model)`；② 位置编码：可学习参数 `nn.Parameter(torch.zeros(1, max_len, d_model))`；③ `nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)`；④ 取最后时间步隐藏态接现有 `forecast_head` | 实例化 `TemporalDeterministicBaseline(architecture="transformer", ...)` 不报错；forward 输出 shape 正确 |
| **PatchTST 分支** | `deterministic_baselines.py:46` 集合加入 `"patchtst"`；`__init__` 新增 `elif architecture == "patchtst":`；`_encode` 新增对应分支 | `patch_len: 4`、`patch_stride: 2`、`patchtst_d_model: 64`、`patchtst_nhead: 4`、`patchtst_num_layers: 2` | ① Patching：对 sequence [B·N, L, input_dim] 做 unfold 得到 patches [B·N, n_patches, patch_len·input_dim]；② 线性投影到 d_model；③ 加位置编码；④ TransformerEncoder；⑤ flatten 所有 patch 输出 → 线性层到 forecast_steps；⑥ channel-independence 天然满足（reshape 为 B·N 独立序列） | history_steps=24 时 n_patches = (24−4)/2 + 1 = 11；实例化不报错 |
| **iTransformer 分支** | `deterministic_baselines.py:46` 集合加入 `"itransformer"`；`__init__` 新增 `elif architecture == "itransformer":`；`_encode` 新增对应分支 | `itransformer_d_model: 64`、`itransformer_nhead: 4`、`itransformer_num_layers: 2` | ① Inverted embedding：每个 variate 的 [L] 序列线性投影到 d_model → [B·N, D, d_model]；② 自注意力在 D 个 variate token 间计算；③ FFN 逐 token；④ 每个 variate token 经线性层输出 forecast_steps 维；⑤ 注意：iTransformer 的输出需按节点聚合后接 forecast_head 或直接输出残差 | 实例化 `architecture="itransformer"` 不报错；forward 返回 dict 含 residual/y_det/hidden/adjacency/graph_weights |
| **_encode 适配** | `deterministic_baselines.py:117-147` `_encode(model_input)` | 无新增参数；新增 `elif self.architecture == "transformer":` / `"patchtst":` / `"itransformer":` 三个分支 | ① transformer：投影→位置编码→Encoder→取 last hidden；② patchtst：unfold patches→投影→Encoder→flatten→linear；③ itransformer：inverted embed→Encoder→逐 token linear；④ 返回 shape 统一为 [B, N, hidden_dim]（与现有 GRU/TCN 分支一致） | 各分支输出 shape = (batch, nodes, hidden_dim) |
| **forward 接口兼容** | `deterministic_baselines.py:149-215` `forward()` | 无改动（复用现有接口）；`graphwavenet`/`agcrn` 特有的 adaptive adjacency 逻辑保持不变 | ① Transformer/PatchTST/iTransformer 无图结构，`adaptive` 用 `_FixedAdaptive`（与 GRU/TCN 一致）；② `adjacency` 输出用 `normalize_adjacency(static_graphs[0])`（与 GRU/TCN 一致）；③ `graph_weights` 第 0 列置 1.0（与 GRU/TCN 一致） | forward 返回 dict 的 key 集合与现有 gru/tcn 分支完全一致 |
| **基线矩阵追加** | `matrix.py:184-198` `definitions` 元组 | 新增 3 条：`("B10", "Transformer encoder-only", "deterministic", {"model.architecture": "transformer"})`、`("B11", "PatchTST", "deterministic", {"model.architecture": "patchtst"})`、`("B12", "iTransformer", "deterministic", {"model.architecture": "itransformer"})` | ① metrics 复用现有 `("mae", "rmse", "r2_nse", "picp", "pinaw", "pinball")`；② group 仍为 `"forecast_baseline"`；③ question 复用 "How does the proposed stack compare with this controlled reference?" | `build_experiment_matrix()` 输出含 B10/B11/B12 条目 |
| **配置键扩展** | `configs/sdwpf_v3_2.yaml` model 段 | 新增：`transformer_d_model: 64`、`transformer_nhead: 4`、`transformer_layers: 2`、`patch_len: 4`、`patch_stride: 2`、`itransformer_d_model: 64`、`itransformer_nhead: 4`、`itransformer_layers: 2` | ① 默认值与现有 `hidden_dim: 64`、`dropout: 0.1` 对齐；② 调参时通过 overrides 覆盖（如 `{"model.patch_len": 8}`）；③ 不影响默认 `architecture: multigraph` 主模型 | yaml 中新增键被 `TemporalDeterministicBaseline.__init__` 读取 |
| **XGBoost 基线（可选）** | 新增 `src/prwarn/models/tree_baselines.py`（独立模块，不混入 TemporalDeterministicBaseline） | 依赖：`xgboost` 包；配置键 `baseline.method: "xgboost"`、`xgboost.n_estimators: 200`、`xgboost.max_depth: 6`、`xgboost.learning_rate: 0.1` | ① 非 nn.Module，不走 PyTorch 训练管线；② 将 [B, N, L, D] 展平为 [B·N, L·D] 特征；③ 逐节点逐 horizon 训练；④ 输出格式与 deterministic 基线对齐（y_det/residual）；⑤ matrix.py 追加 `("B13", "XGBoost", "tree_baseline", {"baseline.method": "xgboost"})` | `import tree_baselines` 不报错；预测输出 shape = [B, N, forecast_steps] |

---

## 5. 验收自检清单

- [ ] `TemporalDeterministicBaseline(architecture="transformer")` 可实例化，forward 输出 dict 含 `residual`/`y_det`/`hidden`/`adjacency`/`graph_weights`
- [ ] `TemporalDeterministicBaseline(architecture="patchtst")` 可实例化，history_steps=24 时 patch 数 = 11
- [ ] `TemporalDeterministicBaseline(architecture="itransformer")` 可实例化，注意力在 variate 维度上计算
- [ ] `build_experiment_matrix()` 输出含 B10（Transformer）、B11（PatchTST）、B12（iTransformer）条目
- [ ] 新增基线与 B2（GRU）、B3（TCN）使用相同的 `masked_huber_loss` 训练损失
- [ ] 新增基线消费同一 `deterministic_train.h5` 残差，`target_mode: physics_residual`
- [ ] 新增基线的 `forward()` 接受 `mask`/`delta_t` 输入通道，与现有 `missing_inputs` 配置兼容
- [ ] 配置文件中 `model.architecture: "transformer"` / `"patchtst"` / `"itransformer"` 可正确路由到对应分支
- [ ] 各新增基线的参数量与 GRU/TCN 同量级（~0.1–2M params），不因 Transformer 参数量过大而不公平
- [ ] 调参预算与现有基线一致（相同的搜索空间大小、相同的 GPU 小时上限）
- [ ] XGBoost（如实现）的特征包含历史窗口展平 + 同样的 mask/delta_t 通道

---

## 6. 引用来源列表

1. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł. & Polosukhin, I. Attention Is All You Need. *Advances in Neural Information Processing Systems 30 (NeurIPS 2017)*, pp. 5998–6008, 2017. arXiv:1706.03762. URL: https://arxiv.org/abs/1706.03762
2. Nie, Y., Nguyen, N. H., Sinthong, P. & Kalagnanam, J. A Time Series is Worth 64 Words: Long-term Forecasting with Transformers. *International Conference on Learning Representations (ICLR)*, 2023. arXiv:2211.14730. URL: https://arxiv.org/abs/2211.14730
3. Liu, Y., Hu, T., Zhang, H., Wu, H., Wang, S., Ma, L. & Long, M. iTransformer: Inverted Transformers Are Effective for Time Series Forecasting. *International Conference on Learning Representations (ICLR)*, 2024. arXiv:2310.06625. URL: https://arxiv.org/abs/2310.06625
4. Chen, T. & Guestrin, C. XGBoost: A Scalable Tree Boosting System. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD)*, pp. 785–794, 2016. arXiv:1603.02754. DOI: 10.1145/2939672.2939785. URL: https://arxiv.org/abs/1603.02754
