# SCI 论文写作 — Skill 与工具调研报告

> 调研时间：2026-10-02
> 调研对象：PR-Warn++ v3.2（IEEE TSTE 投稿方向，`../paper/PR-Warn++_TSTE_v3.2.tex`，IEEEtran 模板 + BibTeX）

---

## 0. 项目现状（调研前提）

| 项目 | 现状 |
|---|---|
| 正文 | `../paper/PR-Warn++_TSTE_v3.2.tex`（约 29 KB，IEEEtran journal） |
| 参考文献 | `../paper/prwarn_tste_v3_2_refs.bib`，正文已用 `\cite{}` 标注 |
| 图 | `../paper/figures/` + Fig1 生成提示词 |
| 修订记录 | `../paper/PR-Warn++_TSTE_v3.2_逐段修订与引用审计.md`、`PR-Warn++_TSTE_v3.2_修订说明.md` |
| 实验 | 230 个五折 job（A0–A11 消融 / G1–G6 联合概率 / B0–B9 基线） |
| 待办 | 正文含 `\TODO{}` 占位，需填入服务器实测结果 |

**结论：** 论文骨架、公式、叙事已经冻结，当前瓶颈不在"写"，而在 **结果回填 + 引用核验 + 图表 + 格式合规** 四件事。

---

## 1. 可安装的市场 Skill（推荐清单）

来自 WorkBuddy 推荐市场（BuiltinMarket），均**未安装**，可按需装：

| 优先级 | Skill | skillId | 用途 | 对应阶段 |
|---|---|---|---|---|
| ★★★ | **citation-manager** | `skill_2053081456646492160` | 学术引用管理，为论文补充**真实**参考文献并规范引用标注 | 引用管理 |
| ★★★ | **deep-research** | `skill_2053082035566706688` | 结构化深度调研：大纲生成 → 并行检索 → 报告输出 | 文献调研 |
| ★★☆ | **arxiv-reader** | `skill_2053081352472563712` | 基于 LLM 的 arXiv 论文分类与深度阅读 | 文献调研 |
| ★★☆ | **web-search-exa** | `skill_2053083260766593024` | Exa 神经语义搜索：网页 / 论文 / 代码 / 深度调研 | 文献调研 |
| ★★☆ | **arxiv-watcher** | `skill_2053081360459321344` | 搜索并总结 arXiv 最新论文（跟踪同赛道新工作） | 文献调研 |
| ★★☆ | **humanizer** | `skill_2053082097175687168` | 去除文本 AI 写作痕迹（应对期刊 AI 检测） | 论文写作 |
| ★☆☆ | **tavily** | `skill_2053082852155592704` | AI 优化搜索引擎，智能摘要 + 领域过滤 | 文献调研 |
| ★☆☆ | **fbs-bookwriter** | `skill_2053083113738002432` | 长文档手稿工具链，含 S/P/C/B 分层审校与排版 | 论文写作 |

> SkillHub（lightmake.site）上另有第三方 `paper-polisherskill`（论文降重 / 去 AI 味）、`academic-paper-writing-style`（地学领域写作规范，**与本课题方向不符**）、`edu-thesis-writing`（毕业论文向，**偏学位论文，不适用 SCI**）。优先级低于上表。

---

## 2. 已内置、开箱即用的工具

无需安装，直接调用：

| 工具 | 在论文中的用途 |
|---|---|
| **tencent-docx** | 生成/美化 Word 版论文、审稿回复信（Response Letter） |
| **tencent-pptx** | 生成会议报告 PPT / 答辩幻灯片 |
| **pdf / pdfkit-py** | 读取参考文献 PDF、抽取表格、合并投稿材料、OCR |
| **tencent-docs-sheetagent** | 分析 230-job 实验矩阵结果、生成对比表、画图 |
| **tencent-docs-sheet-generation** | 从零生成结果汇总 xlsx |
| **ImageGen** | 生成 Fig1 示意图（已有提示词 `../paper/PR-Warn++_Fig1_v3.2_生成提示词.txt`） |
| **library（资料库）** | 归档文献、稿件版本、投稿材料 |
| **skill-creator** | 把本项目的写作/审计流程固化为可复用 Skill |

---

## 3. 能力缺口（需自建或走本地工具链）

调研后确认 **市场上没有** 以下现成 Skill，建议自建：

1. **LaTeX 编译与 IEEE 格式合规检查**
   - 缺口：无 LaTeX 编译 / `IEEEtran` 排版检查 Skill。
   - 方案：本地安装 MiKTeX/TeX Live，写一个 `latex-check` Skill 固化
     「编译 → 检查 overfull box → 检查未定义引用 → 检查图表编号」流程。

2. **BibTeX 引用真实性审计**
   - 缺口：`citation-manager` 负责"加引用"，但不做**已有 .bib 的逐条真伪核验**。
   - 风险：正文已有 ~40 条 `\cite{}`，其中 `hou2026wake`、`liu2026diffusion`、`ma2026stgld` 等 2026 年条目需核实是否真实存在（**幻觉引用是 SCI 投稿致命伤**）。
   - 方案：自建 `bib-audit` Skill —— 逐条比对 DOI / Crossref / arXiv，输出「已核实 / 存疑 / 查无此文」三态表。

3. **实验结果的统计显著性表述**
   - 缺口：项目已有 HAC/DM 检验与聚类自助法 CI，但没有把结果**转成论文级表格与措辞**的 Skill。
   - 方案：结合 `tencent-docs-sheetagent` + 一个写作模板 Skill。

4. **投稿信 / Cover Letter 与审稿回复**
   - 缺口：无专项 Skill。可用 `tencent-docx` + 自建模板解决。

---

## 4. 建议执行顺序

```
第 1 步  装 citation-manager + deep-research        → 补齐/核验文献
第 2 步  自建 bib-audit（或手工核验 2026 年条目）    → 消除幻觉引用（最高风险）
第 3 步  sheetagent 汇总 230-job 结果               → 生成论文级表格
第 4 步  ImageGen + figures                         → 完成 Fig1 与实验图
第 5 步  回填 \TODO{}，本地 LaTeX 编译出 PDF        → 校验 IEEEtran 合规
第 6 步  humanizer 润色 + tencent-docx 出回复信     → 投稿材料
```

---

## 5. 一句话结论

**写论文本身不缺工具，缺的是"引用核验"和"LaTeX 合规"两个环节。**
建议优先安装 `citation-manager` + `deep-research`，并自建 `bib-audit` Skill 处理 2026 年文献的真实性问题。
