## 缺口 6：配对统计推断增强（效应量 + Holm 多重比较校正）（优先级 P1）

> 对应约束：`docs/约束符合性核查报告.md` **C2-2**（配对差异/CI/效应量/Holm，状态 △ 部分满足）；`docs/投稿约束清单.md` **C2-2**（配对差异、95% CI、效应量、p 值；多重比较用 Holm 校正）。
> 本文档只做调研与可落地实现规范设计，不含任何训练/实验执行。

### 1. 缺口定位

**现状（已 Read 核实，行号对应当前代码）：**

| 能力 | 落点文件 / 函数 / 行号 | 现状 |
|---|---|---|
| HAC 方差 | `src/prwarn/eval/statistics.py:newey_west_variance`(:37) | Bartlett 核长run方差，已实现 |
| 配对 DM 检验 | `statistics.py:diebold_mariano`(:56) → `DMResult`(:13) | 已输出 `statistic / p_value / mean_loss_difference / hac_lags / n`；**无效应量** |
| 聚类 bootstrap CI | `statistics.py:event_bootstrap_ci`(:94) → `BootstrapCI`(:28) | 已按事件簇重抽样给百分位 CI |
| 事件窗口切分 | `statistics.py:contiguous_event_ids`(:139) | 已实现 |
| seed 聚合 | `statistics.py:aggregate_seed_metrics`(:189)；`src/prwarn/cli/aggregate_seed_runs.py` | 仅输出 `mean/std(ddof=1)/min/max/count`；**无跨 seed 配对效应量、无检验** |
| 实验指标元组 | `src/prwarn/experiments/matrix.py` | 各消融/基线 `primary_metrics` 声明 rmse/crps/energy_score 等，但无配对检验输出位 |

**缺口是什么（C2-2 缺的两件事）：**

1. **无效应量（effect size）**：现有 `DMResult` 只给"差异是否非零"的 p 值与均值差，不给"差异有多大（标准化）"。样本量一大时即使微小退化也显著，审稿人（R1.3）要求同时报告效应量以区分统计显著与实质显著。
2. **无多重比较校正**：当同一主模型与多个基线（B0–B9、A0–A11 变体、多个压力档）做 k 次配对检验时，未做家族错误率控制；k 个 p 值直接逐个与 α=0.05 比较会抬高假阳性。C2-2 指定用 **Holm 序贯拒绝法**。

### 2. 文献依据表

| 文献（作者-年份） | 出处/年份 | DOI/arXiv | 核验状态 | 关键做法（可确认部分） | 可借鉴的协议要素 |
|---|---|---|---|---|---|
| Diebold & Mariano (1995) | *Journal of Business & Economic Statistics*, 13(3), 253–263, 1995 | 10.1080/07350015.1995.10524599 | VERIFIED（作者 PDF、CRAN 参考文献页、IMF 工作论文三方一致） | 配对预测精度比较：对两条配对损失序列求损失差 `d_t = loss_A,t − loss_B,t`，用 HAC 方差标准化均值差做等预测能力检验 | 配对单位 = 同一 origin/事件上两条损失序列；损失差符号约定；HAC 滞后处理时序相关（代码 `diebold_mariano` 已实现） |
| Holm (1979) | *Scandinavian Journal of Statistics*, 6(2), 65–70, 1979 | 10.2307/4615733 | VERIFIED（CRAN、PLOS 参考文献、arXiv:1707.06706 三方一致） | 序贯拒绝（step-down）多重检验：k 个 p 值升序排列，第 i 个与 α/(k−i+1) 比较，首个不拒绝即停；在控制 FWER 上比 Bonferroni 更强 | 多重比较校正算法骨架（见 §3-2）；调整后 p 值单调非降 |
| Benjamini & Hochberg (1995) | *JRSS-B*, 57(1), 289–300, 1995 | 10.1111/j.2517-6161.1995.tb02031.x | VERIFIED（Oxford Academic JRSS-B 官方页、IMS Pearson 奖页、Tel Aviv 作者主页三方一致） | FDR 控制的 step-up 法；与 Holm（FWER）构成"严格 vs 宽松"两档选择 | 作为 Holm 的对照/备选写入文档：投稿默认 Holm（C2-2 硬性要求），探索性分析可用 BH |
| Cohen (1988) | *Statistical Power Analysis for the Behavioral Sciences*, 2nd ed., Lawrence Erlbaum, 1988（专著） | 无 DOI（经典著作） | VERIFIED（Cambridge MRC CBU stats wiki、Bristol R 包参考文献、Wiley Encyclopedia 三方一致） | 标准化均值差 `d = (μ₁−μ₂)/σ`；给出小/中/大效应经验阈值 0.2/0.5/0.8 | 效应量定义基准；配对设计用差值序列 SD 标准化（d_z），而非合并 SD |

> 补充方法脉络（非表格行，仅作设计依据）：Giacomini & White (2006, *Econometrica* 74(6):1545–1578, DOI 10.1111/j.1468-0262.2006.00718.x，VERIFIED）把 DM 推广到条件预测能力与估计样本外场景；本代码 `diebold_mariano` 属其无条件 EPA 框架，保留 `hac_lags` 即对应其"用滚动/固定估计窗口评估"的设定，不需另引入 GW 检验。

### 3. 标准做法要点

**3-1 配对效应量（Cohen's d 配对版）**

对配对损失序列 `a_i = loss_A,i`、`b_i = loss_B,i`（同一 origin/事件 i，已按 `diebold_mariano` 同样剔除非有限对），定义差值 `d_i = a_i − b_i`：

- 均值差：`M_d = mean(d_i)`（即 `DMResult.mean_loss_difference`，单位 kW）。
- 差值 SD：`SD_d = sqrt( Σ(d_i − M_d)² / (n−1) )`（ddof=1）。
- **配对 Cohen's d_z = M_d / SD_d**（无量纲）。这是配对设计下与 DM 检验同一差值序列配套的效应量，避免用合并 SD 导致的高估。
- 小样本校正 **Hedges' g = J · d_z**，`J = 1 − 3/(4·(n−1) − 1)`（小样本无偏校正因子）。
- 判读阈值沿用 Cohen 1988 经验值：|d_z|≈0.2 小、0.5 中、0.8 大（仅作参考，须结合问题语境）。

**3-2 Holm 序贯拒绝校正算法（step-down）**

输入：k 个原始 p 值 `p_1..p_k`，家族显著性 α（默认 0.05）。
1. 升序排序 `p_(1) ≤ p_(2) ≤ … ≤ p_(k)`，记录原假设下标映射。
2. 从 i=1 起依次比较 `p_(i) < α/(k − i + 1)`。
3. 一直拒绝到第一个不满足的 i*，停；i* 及之后全部不拒绝，i* 之前全部拒绝。
4. 输出调整后 p 值（单调化）：`p̃_(i) = max_{j≤i} (k − j + 1)·p_(j)`，并截断到 ≤1。
- 性质：在任意相关结构下控制 FWER，且统一功效高于 Bonferroni。

**3-3 与现有函数的关系**

- `diebold_mariano` 不动签名；新增一个组合函数把 DMResult、d_z、g、CI 打包。
- CI 直接复用 `event_bootstrap_ci`（按事件簇重抽样估计 M_d 的百分位区间），或在样本充足时用 `M_d ± z·SE`（SE 来自 `newey_west_variance`）；优先事件 bootstrap 以贴合时序聚类。
- 多重比较的"k"= 同一主模型在同一张表里要比较的基线/变体个数（如 A9 的 x_only/x_mask/x_mask_delta，或 B0–B9）；在报告里显式写明 k 与 α。

### 4. 实现规范（映射到本代码库）

> 原则：在 `statistics.py` 内新增纯函数（无新依赖、可单测），不改 `diebold_mariano`/`event_bootstrap_ci` 既有行为；聚合 CLI 只消费新函数。

| 改动点 | 落点文件/函数/配置键 | 新增参数与默认值 | 协议要点 | 验收方式 |
|---|---|---|---|---|
| ① 配对效应量函数 | `statistics.py` 新增 `EffectSizeResult` dataclass 与 `cohens_d_paired(loss_a, loss_b) -> EffectSizeResult` | 内部自动对齐 `diebold_mariano` 的有限对掩码逻辑 | 计算 `mean_difference / sd_difference(ddof=1)` 得 d_z；同函数算 Hedges g（J 校正）；输出 `n_pairs` | 单测：两序列完全相等 → d_z=0；`a=b+1` 常量偏移 → d_z 发散（SD=0），应返回 inf 并按约定标记 |
| ② Holm 校正函数 | `statistics.py` 新增 `holm_correction(p_values, *, alpha=0.05) -> HolmResult` | `alpha: float=0.05` | 实现 §3-2 step-down；返回每个原假设的 `raw_p / adjusted_p / reject / rejected_order` | 单测：p=[0.001,0.008,0.039,0.041,0.042]（k=5,α=.05）应与教科书手算一致；调整后 p 单调非降 |
| ③ 配对比较组合器 | `statistics.py` 新增 `paired_model_comparison(loss_a, loss_b, event_ids=None, *, hac_lags, alpha=0.05, n_boot=2000, seed=0) -> dict` | `hac_lags: int`（必填，沿用 DM）；`event_ids=None` 时退化为 HAC 正态 CI | 内部串起 `diebold_mariano` + `cohens_d_paired` + （可选）`event_bootstrap_ci`；输出 `statistic,p_raw,p_adjusted_holm,reject,cohens_d_z,hedges_g,mean_diff,ci_lower,ci_upper,n_pairs,hac_lags` | 对两条已知损失序列跑通，字段齐全；event_ids 给定时 CI 来自聚类 bootstrap |
| ④ 多比较批量校正接入 | `statistics.py` 在 `paired_model_comparison` 之上新增 `apply_holm_to_reports(comparisons: list[dict], alpha=0.05)` | 无 | 收集一张表内所有 comparisons 的 p_raw，统一 `holm_correction` 后回填 `p_adjusted_holm`/`reject`；写清 k | 传入 3 个比较，输出每个都带同一家族的 adjusted p |
| ⑤ seed 聚合扩展 | `aggregate_seed_runs.py` / `aggregate_seed_metrics`(:189) | 新增可选输出块 `paired:`（消费 ④，不改既有 `summary` 结构） | 跨 seed 配对时，配对单位仍是 origin/事件级损失差，不是把 seed 均值直接当样本；seed 只用于报告 mean±SD（C0-2） | 聚合产物 JSON 同时含 `summary`（mean/std）与 `paired`（效应量+Holm），二者并存不互相覆盖 |

**新增输出字段约定（写入比较报告 JSON）：**

```json
{
  "model_a_vs_b": {
    "mean_loss_difference_kW": 12.4,
    "dm_statistic": 2.13,
    "p_raw": 0.033,
    "p_adjusted_holm": 0.099,
    "reject_after_holm": false,
    "cohens_d_z": 0.22,
    "hedges_g": 0.21,
    "ci95": [2.1, 22.7],
    "n_pairs": 4120,
    "hac_lags": 6,
    "family_k": 5,
    "alpha": 0.05
  }
}
```

### 5. 验收自检清单

- [ ] `cohens_d_paired`：常量偏移场景下 d_z 符号与 `mean_loss_difference` 一致；n=1 时不崩（返回 NaN 并提示样本不足）。
- [ ] `holm_correction`：手算样例 p 数组的 adjusted p 与文献示例一致；调整后序列单调非降；k 个结论的 reject 集合是"前 r 个拒绝"的前缀形式。
- [ ] `paired_model_comparison`：同时给出 p_raw 与（家族层面）p_adjusted_holm；给 event_ids 时 CI 来自 `event_bootstrap_ci`（n_events 字段可见）。
- [ ] 报告中每个关键对比都同时出现：配对差值、95% CI、p_raw、p_adjusted_holm、reject、cohens_d_z、n_pairs——满足 C2-2 四件套。
- [ ] `diebold_mariano`/`event_bootstrap_ci`/`aggregate_seed_metrics` 既有签名与返回未变（A 系列/G 系列/B 系列指标链路不破坏）。
- [ ] 探索性分析若用 BH，文档中与 Holm 明确区分（Holm 为投稿默认，对应 FWER）。
- [ ] 聚合 CLI 产物里 `summary`（mean±SD）与 `paired`（效应量+校正）并存。

### 6. 引用来源列表

1. Diebold, F. X., & Mariano, R. S. *Comparing Predictive Accuracy.* Journal of Business & Economic Statistics, 13(3), 253–263, 1995. DOI: 10.1080/07350015.1995.10524599. URL: https://faculty.washington.edu/ezivot/econ584/dieboldMariano.pdf
2. Holm, S. *A Simple Sequentially Rejective Multiple Test Procedure.* Scandinavian Journal of Statistics, 6(2), 65–70, 1979. DOI: 10.2307/4615733. URL: https://www.jstor.org/stable/4615733
3. Benjamini, Y., & Hochberg, Y. *Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing.* Journal of the Royal Statistical Society Series B, 57(1), 289–300, 1995. DOI: 10.1111/j.2517-6161.1995.tb02031.x. URL: https://academic.oup.com/jrsssb/article/57/1/289/7035855
4. Cohen, J. *Statistical Power Analysis for the Behavioral Sciences*, 2nd ed. Hillsdale, NJ: Lawrence Erlbaum Associates, 1988.（专著，无 DOI；经验阈值 0.2/0.5/0.8 与 d=(μ₁−μ₂)/σ 出处）
5. （方法脉络补充）Giacomini, R., & White, H. *Tests of Conditional Predictive Ability.* Econometrica, 74(6), 1545–1578, 2006. DOI: 10.1111/j.1468-0262.2006.00718.x. URL: https://onlinelibrary.wiley.com/doi/10.1111/j.1468-0262.2006.00718.x
