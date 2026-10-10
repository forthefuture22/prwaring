# 学术 AI 技能 — GitHub 原版定位与安装报告

> 执行时间：2026-10-02
> 方法：GitHub API 按 **fork 数降序** 检索 → 浅克隆原版仓库 → 静态安全审计 → 安装
> 安装目录：`C:\Users\25785\.workbuddy-ai\skills\`（用户级）
> 原版仓库克隆位置：`tmp\github-skills\`（13 个仓库，约 1.1 GB）

---

## 📊 执行摘要

| 项目 | 数量 |
|---|---|
| 博客列出的技能 | 13 |
| 在 GitHub 定位到原版仓库 | 12 |
| 已安装的技能条目 | **33** |
| 其中替换了此前的 SkillHub 版本 | 3 |
| 确认不是技能（是应用/代码/书） | 3 |
| 克隆仓库总大小 | 13 个仓库，约 1.55 GB |

**关键发现：** 博客里的技能名全部对应真实开源项目，且 **fork 数最高的确实就是上游原版**——你的判断是对的。但要注意：其中 3 个根本不是"技能"，`clouddesign` 是个**空仓库**（只有 LICENSE）。

---

## 🗺️ 博客技能 → GitHub 原版（按 fork 排序）

| 博客分类 | 博客名称 | GitHub 原版 | Stars | **Forks** |
|---|---|---|---|---|
| 课题构思 | Academic Research Skills | `Imbad0202/academic-research-skills` | 50,118 | **3,868** |
| 课题构思 | Claude Scholar | `Galaxy-Dawn/claude-scholar` | 5,641 | **443** |
| 课题构思 | 微软 Research Studio | `microsoft/ResearchStudio` | 2,982 | **172** |
| 实操辅助 | Scientific Agent Skills | `K-Dense-AI/scientific-agent-skills` | 47,334 | **4,285** |
| 实操辅助 | Auto claudecode | `wanshuiyin/Auto-claude-code-research-in-sleep` | 16,890 | **1,427** |
| 实操辅助 | 卡帕西 auto research | **`karpathy/autoresearch`** | **97,135** | **13,538** |
| 论文写作 | Nature Skills | `Yuan1z0825/nature-skills` | 45,547 | **2,378** |
| 论文写作 | Paper Spine | `WUBING2023/PaperSpine` | 5,712 | **224** |
| 论文绘图 | scientific Visualization book | `rougier/scientific-visualization-book` | 11,587 | **1,014** |
| 论文绘图 | 复杂示意图组件 | `LigphiDonk/academic-figure-generator` | 2,513 | **127** |
| 学术汇报 | codex ppt skill | `ningzimu/codex-ppt-skill` | 6,306 | **310** |
| 学术汇报 | latex beamer | `Faust-Donf/beamer-academic` | 305 | **11** |
| 学术汇报 | clouddesign | `cloudskyme/clouddesign` | 2 | **3** ⚠️ **空仓库** |

> ⚠️ **注意**：卡帕西的 `karpathy/autoresearch` 是 **13,538 forks / 97,135 stars** 的原始仓库，但它是**单卡 nanochat 训练的科研代码**（`train.py` + `program.md`），**不是 Agent Skill**，无法作为技能安装。同理 `microsoft/ResearchStudio` 是完整应用（含 Idea / Reel 两个子产品），`rougier/scientific-visualization-book` 是一本开源书。这三者是**参考资料**，不是可安装技能。

---

## ✅ 已安装的 32 个技能（按博客五大分类）

### 一、课题构思类（5）
| 技能 | 来源 | 说明 |
|---|---|---|
| `deep-research` | academic-research-skills | 13-agent 深度调研流水线，8 种模式 |
| `research-ideation` | claude-scholar | 研究选题构思 |
| `idea-spark` | microsoft/ResearchStudio | 微软官方：想法火花 |
| `idea-creator` | ARIS | 自动生成研究想法 |
| `novelty-check` | ARIS | 新颖性查重 |

### 二、实操辅助类（5）
| 技能 | 来源 | 说明 |
|---|---|---|
| `academic-pipeline` | academic-research-skills | 全流程编排 |
| `experiment-plan` | ARIS | 实验规划 |
| `analyze-results` | ARIS | 结果分析 |
| `statistical-analysis` | scientific-agent-skills | 统计分析（**自建 venv 隔离**） |
| `paper-lookup` | scientific-agent-skills | 论文检索 |

### 三、论文写作类（12）
| 技能 | 来源 | 说明 |
|---|---|---|
| `academic-paper` | academic-research-skills | 12-agent 论文写作流水线，11 种模式 |
| `academic-paper-reviewer` | academic-research-skills | 模拟审稿 |
| `ml-paper-writing` | claude-scholar | ML 论文写作（含引用工作流） |
| `writing-anti-ai` | claude-scholar | 去 AI 味 |
| `paper-self-review` | claude-scholar | 投稿前自审 |
| `review-response` | claude-scholar | 审稿回复 |
| `nature-writing` | nature-skills | Nature 风格写作（77 文件） |
| `nature-polishing` | nature-skills | Nature 风格润色 |
| `nature-citation` | nature-skills | 引用处理 |
| `nature-ref-verifier` | nature-skills | **参考文献真伪核验** |
| `scientific-writing` | scientific-agent-skills | 科学写作 |
| `paper-spine` | WUBING2023/PaperSpine | 端到端论文（166 文件，**已替换 SkillHub 版**） |

### 四、论文绘图类（7）
| 技能 | 来源 | 说明 |
|---|---|---|
| `scientific-visualization` | scientific-agent-skills | 顶刊配图（**已替换 SkillHub 版**） |
| `matplotlib` | scientific-agent-skills | matplotlib 模板 |
| `nature-figure` | nature-skills | Nature 风格科研绘图（126 文件） |
| `publication-chart-skill` | claude-scholar | 出版级图表 |
| `academic-figure-prompt` | LigphiDonk | 论文配图 Prompt 生成 |
| `figure-spec` | ARIS | 图表规格 |
| **`scientific-visualization-book`** | **rougier（新增封装）** | **177 个 matplotlib 代码模板**，按 18 个主题分类 |

> `scientific-visualization-book` 由我基于 Rougier 开源书的 `code/` 目录封装而成（该书本身不是技能，是 590 MB 的开源书）。
> 含 `rules/`（十条作图法则 13 个）、`layout/`（7）、`colors/`（11）、`ornaments/`（11）、`showcases/`（11 个多面板成品图）、`reference/`（16）等，共 194 个文件。
> 其中 3 个模板（`earthquakes.py` ×2、`github-activity.py`）会联网抓取 USGS / GitHub 公开数据，已核实为正常教学示例，无外送。

### 五、学术汇报类（4）
| 技能 | 来源 | 说明 |
|---|---|---|
| `codex-ppt` | ningzimu/codex-ppt-skill | 图像化 PPT（**已替换 SkillHub 版**） |
| `beamer-academic` | Faust-Donf | 论文 → 学术答辩 Beamer |
| `nature-paper2ppt` | nature-skills | 论文 → PPT |
| `paper-slides` | ARIS | 论文幻灯片 |

---

## 📋 安全审计结果

按 `skills-security-check` 流程对 32 个候选做静态扫描：

| 检查项 | 结果 |
|---|---|
| 远程下载 + 执行（`curl \| bash`） | ✅ 未发现 |
| 数据外送（POST 到第三方） | ✅ 未发现 |
| 破坏性命令（`rm -rf /` 等） | ✅ 未发现 |
| 敏感文件读取（`.ssh` / `.aws` / `id_rsa`） | ✅ 未发现 |
| 权限提升（`sudo`） | ⚠️ 仅出现在 **TeX Live 安装说明文档**中（`sudo apt install texlive-xetex`），非自动执行 |
| 全局依赖安装 | ⚠️ 均为**文档说明**；`statistical-analysis` 实际使用 `uv pip install --python .venv-statistics/bin/python`（**venv 隔离，良好实践**） |
| `paper-spine/scripts/open_release.py` | ✅ 命中的 "credentials" 是**检测**私钥的正则与断言名，非读取外送 |

**审计结论：32 个全部 P2（可安全使用）。**

---

## ⚠️ 需要说明的几点

1. **网络**：GitHub 直连仅 ~21 KB/s，改走 `gh-proxy.com` 镜像（~325 KB/s）完成克隆。克隆的是**上游原版**，未做任何修改。
2. **不是技能的三项**：`karpathy/autoresearch`（科研代码）、`microsoft/ResearchStudio`（应用）、`rougier/scientific-visualization-book`（开源书）已克隆到 `tmp/github-skills/`。前两项保持原样供参考；第三项我已把其中的 `code/` 目录（177 个模板）封装成了 `scientific-visualization-book` 技能。
3. **`clouddesign` 是空仓库**——只有 LICENSE，无任何内容。博客推荐的这一项实际不可用。
4. **`academic-research-skills` 的 `skills/` 目录是符号链接**，实际技能在仓库根目录，已按真实路径处理。
5. 此前从 SkillHub 安装的 3 个技能（`paper-spine` / `scientific-visualization` / `codex-ppt`）已用 GitHub 原版覆盖。

---

## 💡 对 PR-Warn++ 论文的推荐组合

| 环节 | 推荐技能 |
|---|---|
| 补文献 / 查新颖性 | `deep-research` + `paper-lookup` + `novelty-check` |
| **核验 2026 年参考文献真伪** | `nature-ref-verifier` + `academic-paper` 的引用检查模式 |
| 正文写作 / 重构 | `paper-spine` + `academic-paper` + `nature-writing` |
| 投稿前自审 / 模拟审稿 | `paper-self-review` + `academic-paper-reviewer` |
| 审稿回复 | `review-response` |
| 去 AI 味 | `writing-anti-ai` |
| 实验图（CRPS/ES/VS） | `scientific-visualization` + `nature-figure` + `matplotlib` |
| 具体作图配方（布局/配色/标注） | `scientific-visualization-book`（177 个模板直接抄改） |
| 框架图 | `academic-figure-prompt` + 已装的 `paper-framework-figure-studio-pro` |
| 会议报告 | `codex-ppt` / `nature-paper2ppt` |
| 学术答辩 Beamer | `beamer-academic` |
