## 缺口 3：门控/融合机制干预实验协议（优先级 P0）

> 对应约束：C2-6（门控干预对照 frozen/uniform/shuffled/parameter-matched 缺失）、C2-1（A2 图叠加消融存在，但缺「静态 vs 动态门控」「support-matched / 参数匹配对照」）。
> 本规范只做「文献依据 + 可落地实现设计」，不运行任何训练/实验，不修改现有代码。

### 1. 缺口定位

**现状（已 Read 核实，附行号）：**

- 多图融合由一个**学习式 gate** 产生 logits 加权 4 张图（geo / corr / directional / adaptive）：
  - `src/prwarn/models/backbone.py:147-151` —— `self.gate = nn.Sequential(Linear(hidden_dim, hidden_dim), GELU, Linear(hidden_dim, n_static_graphs + 2))`。
  - `src/prwarn/models/backbone.py:180` —— `logits = self.gate(hidden.mean(dim=1))`；`:181` 用 `enabled_graph_mask` 屏蔽未启用图；`:182` `weights = torch.softmax(logits, dim=-1)`。
  - `src/prwarn/models/backbone.py:183-185` 调 `mix_graphs_from_weights(static_graphs, directional_graph, self.adaptive(), weights)`；该函数定义在 `:24-47`（含 `:40-41` 的 `weights.shape != (batch, expected_graphs)` 校验）。
  - 融合后的权重 `graph_weights` 已从 `forward` 输出（`backbone.py:244`），A2 消融把 `graph_weights` 列为 primary metric（`matrix.py:49`）。
- 现有消融矩阵 `src/prwarn/experiments/matrix.py:46-59`（A2）只做**图叠加**：`geo → geo_corr → geo_corr_dir → all_graphs`，通过覆盖 `graphs.enabled` 列表实现（`matrix.py:51-57`）。
- 配置侧 `configs/sdwpf_v3_2.yaml:33-43` 只有 `graphs.enabled / geographic_k / correlation_k / correlation_mode / direction_distance_scale / direction_sigma_degrees / direction_sector_degrees` 等**图构造参数**，**没有任何 gate 行为开关**。

**缺口是什么：**

A2 只能证明「每加一张图是否带来独立增量」，但无法回答审稿人（R1.3 / R2.1 / R2.7）真正关心的机制问题：**那个学习式 gate 到底有没有用？** 当前 `src` 内没有任何 gate intervention 实验——既不能区分「动态输入相关门控」与「静态固定权重」，也没有「参数匹配的非门控对照」来排除「gate 只是额外参数容量」的混淆。C2-6 要求的四类对照（frozen / uniform / shuffled / parameter-matched）全部缺失。

**为什么必须补（P0 理由）：**

- C2-1 要求每个核心卖点有独立证据链（matched controls + 干预实验），禁止用「E4 vs E6」这种同时改多个因素的混合对比。动态多图融合是本文核心卖点之一，没有干预对照，该卖点在 TSTE 评审下会被直接判为「claim 未被证据支持」（这正是 EPSR 被拒的教训 R1.3）。
- C2-4 同时要求「机制解释有界」：门控权重是预测混合系数，不得声称因果/尾流物理。补干预实验不是为了「证明 gate 因果」，而是为了**在论文里把 claim 收敛到证据支持的程度**——有干预对照，才能诚实地说「动态门控相对静态/均匀/打乱对照带来了 X 的 RMSE 改善，且该改善不是来自额外参数」。

### 2. 文献依据表

| 文献（作者-年份） | 出处/年份 | DOI/arXiv | 核验状态 | 关键做法 | 可借鉴的协议要素 |
|---|---|---|---|---|---|
| Jain & Wallace (2019) | NAACL-HLT 2019（Long Papers），pp. 3543–3556 | DOI: 10.18653/v1/N19-1357；arXiv:1902.10186 | VERIFIED（ACL Anthology 页面逐字核对标题/作者/页码/DOI；arXiv abs 页核对摘要） | 系统检验 attention 权重是否「可解释」：发现学习到的 attention 权重常与梯度特征重要性不相关，且能找到**与学习分布差异很大、却给出等价预测的替代分布**（权重替换/扰动实验） | ①「权重替换后性能是否下降」是检验模块必要性的直接干预；②若打乱/替换权重后性能几乎不变，说明输入→权重映射未被因果使用——直接支撑 shuffled 对照的设计与解读 |
| Wiegreffe & Pinter (2019) | EMNLP-IJCNLP 2019，pp. 11–20 | DOI: 10.18653/v1/D19-1002；arXiv:1908.04626 | VERIFIED（ACL Anthology D19-1002 页面逐字核对作者 Sarah Wiegreffe / Yuval Pinter、年份、页码 11–20、摘要） | 针对上一篇提出**四类诊断测试**：uniform-weights baseline、跨多随机种子的方差校准、用预训练模型冻结权重喂给非上下文 MLP 的诊断框架、对抗式 attention 训练 | ① **uniform 对照**（均匀权重 baseline）直接对应本文 uniform 模式；② **frozen 对照**（用预训练 gate 的固定权重、脱离输入上下文）直接对应本文 frozen_mean 模式与「静态 vs 动态」区分；③「把 attention 从主干架构中隔离出来」的思路对应 parameter-matched 对照 |
| Michel, Levy & Neubig (2019) | NeurIPS 2019（Advances in NeurIPS 32） | arXiv:1905.10650（ACM DOI: 10.5555/3454287.3455544） | VERIFIED（arXiv abs 页 1905.10650 与 alphaxiv 镜像核对标题「Are Sixteen Heads Really Better than One?」、作者 Paul Michel / Omer Levy / Graham Neubig、年份；多源引用交叉确认） | 对每个 attention head 做**移除/掩码干预**，用「loss 对 mask 的梯度 × gate 值」定义重要性分数，测量移除后性能下降；发现多数 head 可被移除而不显著掉点 | ①「移除/掩码一个模块 → 测量性能下降」是必要性检验的标准范式；②重要性必须在**受控容量**下解读——支撑 parameter-matched 对照（排除「只是参数变多了」） |

> 说明：本缺口只选了 3 篇，均为 AI 顶会（NAACL / EMNLP / NeurIPS），与任务指定的「可含可解释性/干预实验方向的 AI 顶会」一致。未强行凑 Energy/Applied Energy 的融合消融论文——检索到的该方向中文报道/二手摘要无法逐字核验标题+作者+期刊+DOI，按「宁缺毋滥」原则不纳入。

### 3. 标准做法要点

从上述文献提炼的可执行协议要素（每条注明来源）：

1. **四类干预对照必须成套出现，缺一不可**：uniform（均匀权重）、frozen（固定为预训练/验证集均值的静态权重）、shuffled（打乱 batch 内权重对应关系）、parameter-matched（参数预算相同但无学习式门控）。这是 Wiegreffe & Pinter (2019) 四类诊断测试在本代码库的直接落地（来源：Wiegreffe & Pinter 2019）。
2. **uniform 是零假设基线**：把 4 张图等权混合（1/K），检验「非均匀加权」本身是否必要（来源：Wiegreffe & Pinter 2019）。
3. **frozen 是「静态 vs 动态」的关键对照**：gate 输出不再随输入变化，而是广播一个常数权重向量（该常数 = 在验证集上对学习式 gate 求平均得到的权重）。它与 learned 模式的差，才是「动态输入相关门控」的净贡献（来源：Wiegreffe & Pinter 2019 的 frozen-weights 诊断）。
4. **shuffled 检验「输入→权重映射」是否被因果使用**：把 gate 输出的 batch 维权重向量做固定种子的随机置换，保持权重边缘分布不变、仅打破「哪个样本对应哪组权重」的对应。若置换后性能几乎不变，说明该映射未被因果利用（来源：Jain & Wallace 2019 的「替代分布仍给出等价预测」发现；Michel et al. 2019 的掩码/移除干预）。
5. **parameter-matched 排除「参数容量混淆」**：保留 gate MLP 的相同参数量，但强制其输出为固定分布并阻断梯度，使总参数量与 learned 模型一致、却无学习式门控。这样 learned vs parameter_matched 的差才是「门控机制本身」而非「额外参数」的贡献（来源：Wiegreffe & Pinter 2019 的「隔离 attention 与主干架构」思路；Michel et al. 2019 的受控重要性测量）。
6. **干预分两类执行方式**：uniform / parameter_matched 适合「从头训练的变体」（拟合现有 ExperimentSpec 矩阵机制）；frozen_mean / shuffled 本质是「在已训练 learned checkpoint 上做后验推理干预」（frozen_mean 需要先在验证集上求出均值向量，shuffled 直接在推理时置换），需单独的后验干预 CLI（来源：Jain & Wallace 2019 的后验权重替换范式）。
7. **报告必须附带门控权重统计量**：记录各图权重的均值/标准差/跨样本方差，作为「gate 是否真的在做非平凡选择」的直接证据（来源：Wiegreffe & Pinter 2019 的跨种子方差校准；本仓库 `graph_weights` 已可输出，backbone.py:244）。
8. **结论措辞要有界**：干预实验只能支持「动态门控相对对照带来 X 改善」，不得据此声称 gate 编码了尾流/因果物理（与 C2-4 一致）。

### 4. 实现规范（映射到本代码库）

> 下列均为**设计规范**，不在本任务中实施；改动点列出现有真实文件/函数/配置键，新增键给出推荐键名与默认值。

| 改动点 | 落点文件/函数/配置键 | 新增参数与默认值 | 协议要点 | 验收方式 |
|---|---|---|---|---|
| 新增 gate 行为开关 | `configs/sdwpf_v3_2.yaml` 新增 `graphs.gate_mode` | `graphs.gate_mode: learned`（枚举：`learned`/`uniform`/`frozen_mean`/`shuffled`/`parameter_matched`，默认 `learned` = 现状） | 在 `_mix_graphs`（backbone.py:170-186）算出 `weights` 后按 mode 分支替换；`learned` 时行为与现状逐位一致 | 切到每个枚举值，前向输出 shape 仍为 `(B,N,N)`；`graph_weights` 输出 shape 仍为 `(B, K)` |
| uniform 模式 | 同 `graphs.gate_mode: uniform` | 无额外键 | `weights = 1/len(enabled_graphs)` 沿 K 维广播；未启用图仍被 `enabled_graph_mask`（backbone.py:124-127）屏蔽 | 输出 `graph_weights` 每行在启用图上严格相等 |
| frozen_mean 模式 | 同开关；新增缓存键 `graphs.gate_frozen_weights` | `graphs.gate_frozen_weights: null`（后验脚本填充为长度 = 启用图数的 float 列表） | `weights = 缓存常数向量广播到 batch`；该向量由后验脚本在**验证集**上对 learned checkpoint 的 gate 输出求均值得到；只在验证集拟合，绝不触碰测试集 | 验证：同一 checkpoint 在验证集上求得的向量可复现；frozen_mean 下 `graph_weights` 跨样本完全相同 |
| shuffled 模式 | 同开关；新增种子键 `graphs.shuffle_seed` | `graphs.shuffle_seed: 2025` | 后验推理干预：`weights = weights[torch.randperm(B, generator=seed)]`，保持边缘分布、打破样本对应；不重训 | 验证：置换后 `graph_weights` 的逐图均值与 learned 模式在数值上一致（仅顺序打乱），跨样本对应关系被破坏 |
| parameter_matched 模式 | 同开关 | 无额外键（复用现有 gate MLP 结构） | 保留 `self.gate`（backbone.py:147-151）相同参数量，但前向时**忽略 logits**、输出固定均匀分布并对 gate 参数 `requires_grad=False`（或等效 detach）；使总参数量与 learned 一致但无学习门控 | 验证：`sum(p.numel() for p in model.parameters())` 与 learned 模型完全相等；gate 参数不更新 |
| 模型构造接受新键 | `DynamicMultiGraphResidualForecaster.__init__`（backbone.py:87-105）新增 `gate_mode` / `gate_frozen_weights` / `shuffle_seed` 形参；`_mix_graphs`（:170-186）消费 | `gate_mode: str = "learned"`；`gate_frozen_weights: Tensor | None = None`；`shuffle_seed: int = 2025` | 在 `:180` 计算 logits、`:182` softmax 之后插入分支；frozen_mean/shuffled/parameter_matched 均不依赖输入 logits | 单元级：给定固定输入与固定 gate 权重，五种 mode 的输出可被独立复算 |
| 新增实验矩阵规格 | `src/prwarn/experiments/matrix.py` 新增 `ExperimentSpec`（建议 logical_id `A12`，group `ablation`，接在 A11 之后） | variants 复用现有 `(variant, overrides)` 机制 | `learned / uniform / parameter_matched` 三个可从头训练变体直接写成覆盖键；`frozen_mean / shuffled` 标注为后验干预（`server_required` 或备注说明依赖 learned checkpoint） | `build_experiment_matrix()` 能展开出 A12 的可执行 run contract；`experiment_id` 稳定、`evidence_status=planned_not_run` |
| 后验干预 CLI（新增文件） | 建议新增 `src/prwarn/cli/evaluate_gate_intervention.py` | 输入：learned checkpoint 路径、`--mode {frozen_mean,shuffled}`、`--split val` | 加载 learned checkpoint → 在验证集前向收集 gate 权重求均值（frozen_mean）或现场置换（shuffled）→ 输出各 mode 的 RMSE/CRPS 与 `graph_weights` 统计；**只读验证集** | CLI 退出码 0；产物含每图权重均值/标准差、与 learned 模式的配对差异 |
| 门控权重统计产物 | 复用 `backbone.py:244` 已输出的 `graph_weights`；建议在评估产物中落盘 | 无新配置键 | 记录四图（geo/corr/directional/adaptive）权重的跨样本均值与标准差，作为「gate 是否非平凡」的证据列 | 评估日志中出现 `graph_weight_mean` / `graph_weight_std` 字段 |

**新增配置键汇总（推荐键名与默认值）：**

- `graphs.gate_mode: learned`
- `graphs.gate_frozen_weights: null`
- `graphs.shuffle_seed: 2025`

### 5. 验收自检清单

- [ ] `configs/sdwpf_v3_2.yaml` 出现 `graphs.gate_mode`，默认值 `learned`，切到 `learned` 时训练/推理行为与改动前逐位一致（回归不破坏 A0–A11）。
- [ ] `uniform` 模式输出的 `graph_weights` 在启用图上严格等权。
- [ ] `frozen_mean` 模式的常数权重向量仅在**验证集**上从 learned checkpoint 求得，产物中可追溯其拟合 split；推理时跨样本权重完全相同。
- [ ] `shuffled` 模式用固定种子置换，逐图边缘均值与 learned 一致；标注为后验干预、不重训。
- [ ] `parameter_matched` 模式总参数量与 learned 模型逐位相等（用 `sum(p.numel())` 核对），gate 参数不更新。
- [ ] `matrix.py` 新增 A12 规格，`build_experiment_matrix()` 能展开出 run contract，`experiment_id` 稳定、`evidence_status=planned_not_run`。
- [ ] 后验干预 CLI 可加载 learned checkpoint 并在验证集输出五模式对比表（RMSE / CRPS / 各图权重均值与标准差）。
- [ ] 论文结果表对四类对照成套报告，且结论措辞限定为「动态门控相对对照的改善」，未声称 gate 编码尾流/因果物理（对齐 C2-4）。
- [ ] 每个干预变体在多 seed（≥10，对齐 C0-2）下报告 mean ± SD，而非单 seed。

### 6. 引用来源列表

1. Sarthak Jain, Byron C. Wallace. Attention is not Explanation. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT), pp. 3543–3556, 2019. DOI: 10.18653/v1/N19-1357. URL: https://aclanthology.org/N19-1357/ （arXiv: https://arxiv.org/abs/1902.10186 ）
2. Sarah Wiegreffe, Yuval Pinter. Attention is not not Explanation. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pp. 11–20, 2019. DOI: 10.18653/v1/D19-1002. URL: https://aclanthology.org/D19-1002/ （arXiv: https://arxiv.org/abs/1908.04626 ）
3. Paul Michel, Omer Levy, Graham Neubig. Are Sixteen Heads Really Better than One? In Advances in Neural Information Processing Systems 32 (NeurIPS 2019). arXiv:1905.10650. URL: https://arxiv.org/abs/1905.10650
