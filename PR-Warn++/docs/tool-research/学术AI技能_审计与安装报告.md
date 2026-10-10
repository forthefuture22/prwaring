# 学术 AI 技能 — 审计与安装报告

> 执行时间：2026-10-02
> 审计依据：腾讯云鼎实验室 `skills-security-check` 静态审计流程（P0/P1/P2 三级）
> 安装目录：`C:\Users\25785\.workbuddy-ai\skills\`（用户级，全项目可用）

---

## 📊 执行摘要

| 结果 | 数量 |
|---|---|
| 博客列出技能 | 13 |
| ✅ 已安装 | 6 |
| ⛔ 审计后不建议安装 | 5 |
| ❓ 市场中查无此技能 | 2 |

**关键结论：** 博客里的技能名大多能在 SkillHub 找到，但其中 **5 个是空壳或不可移植的**——包括 3 个只写了一行 `git clone` 的"安装器包装"、1 个内容农场生成的假技能、1 个绑定作者私有局域网的脚本。真正有内容、能在这台机器上跑起来的只有 6 个。

---

## 🗺️ 博客技能 → 实际技能 对照表

| 博客分类 | 博客名称 | 实际 SkillHub slug | 结论 |
|---|---|---|---|
| 课题构思 | Academic Research Skills | `academic-research-skills` | ⛔ 空壳 |
| 课题构思 | Claude Scholar | `run-academic-research-workflows-with-claude-scholar` | ⛔ 空壳 |
| 课题构思 | 微软 research studio-idea | `research-idea` | ⛔ 不匹配/不可移植 |
| 实操辅助 | Scientific Agent Skills | `run-research-and-scientific-analysis-workflows-...` | ⛔ 空壳 |
| 实操辅助 | Auto claudecode | — | ❓ 查无此技能 |
| 实操辅助 | 卡帕西 auto research | `auto-research` | ⛔ 不可移植 |
| 论文写作 | Nature Skills | `run-nature-style-academic-writing-and-figure-workflows-...` | ⛔ 空壳 |
| 论文写作 | Paper Spine | `paper-spine` | ✅ **已安装** |
| 论文绘图 | scientific Visualization | `scientific-visualization` | ✅ **已安装** |
| 论文绘图 | 复杂示意图组件 | `paper-framework-figure-studio-pro` | ✅ **已安装** |
| 学术汇报 | clouddesign | — | ❓ 查无此技能 |
| 学术汇报 | codex ppt skill | `codex-ppt` | ✅ **已安装** |
| 学术汇报 | latex beamer | `beamer-pipeline-public` + `latex` | ✅ **已安装** |

---

## ✅ 已安装（6 个）

| 技能 | 文件数 | 审计等级 | 说明 |
|---|---|---|---|
| **paper-spine** | 56 | ✅ P2 | 全流程论文写作方法论：贡献提炼、深度仿写、LaTeX 排版、投稿、审稿回复、降重去 AI 味。**纯文档，无可执行脚本。** |
| **scientific-visualization** | 9 | ✅ P2 | 顶刊级科研配图：多面板布局、色盲友好配色、Nature/Science/Cell 格式要求、matplotlib/seaborn/plotly 样式模板。 |
| **paper-framework-figure-studio-pro** | 32 | ✅ P2 | 论文框架图设计：方法总览图、架构图、流程图、系统数据流图，含候选图筛选工作流。**纯文档。** |
| **latex** | 2 | ✅ P2 | LaTeX 语法速查：转义字符、数学模式、浮动体、bibtex 编译顺序、常见报错。**纯文档。** |
| **codex-ppt** | 41 | ✅ P2 | 从论文/大纲生成图像化 PPT。⚠️ 需 `OPENAI_API_KEY`（图像生成用）；首次运行会**自动创建 venv** 并装依赖。 |
| **beamer-pipeline-public** | 22 | ⚠️ P1 | 论文 → 中文 Beamer 七阶段流水线。⚠️ 需 Node.js 18+ 与 TeX Live；完整流程依赖 OpenClaw 运行时，本机可用的部分是 `--dry-run` 和 prompt 生成。**安装时补写了缺失的 frontmatter。** |

---

## ⛔ 审计后不建议安装（5 个）

### 1–3. 三个"安装器空壳"（同一作者）
- `run-academic-research-workflows-with-claude-scholar`
- `run-nature-style-academic-writing-and-figure-workflows-with-nature-skills`
- `run-research-and-scientific-analysis-workflows-with-scientific-agent-skills`

**问题：** 三个技能全部来自 `github.com/1991513ccie-png/skills`，**正文只有一句话**——`git clone https://github.com/1991513ccie-png/skills /tmp/...`，没有任何实际内容。装上等于装了个"叫你去 clone 别人仓库"的纸条。
**供应链风险：** 它们指向的是个人镜像仓库，而非上游原仓库（`Galaxy-Dawn/claude-scholar`、`Yuan1z0825/nature-skills`、`K-Dense-AI/scientific-agent-skills`）。frontmatter 里的 `verification: security_reviewed` 是作者自填的元数据，不构成任何保证。
**建议：** 如需这些能力，直接从上游官方仓库获取。

### 4. `academic-research-skills` — 内容农场生成的假技能
**问题：** SKILL.md 自述 "Auto-generated from GitHub trending project"，内容是**三遍重复的 B 站视频链接**；配套的 `academic_research_skills.py` 只是往 JSON 里写死一个模板段落（Abstract 永远是 "A comprehensive study of..."）。里面还残留了另一个用户的路径 `C:\Users\pc\.config\opencode\skills\`。
**结论：** 零功能，不装。

### 5. `auto-research` — 绑定作者私有环境
**问题：**
- 硬编码作者内网地址 `http://10.0.0.120:6333`（Qdrant）、`10.0.0.120:6379`（Redis）
- 读取技能目录**之外**的文件取密码：`../../tools/secrets.py get REDIS_PASSWORD`
- 硬编码作者个人 Obsidian 路径 `~/Documents/Obsidian/YoderVault`
- 依赖作者的 `yoder-kb.sh`、`clawhub-skills/` 目录结构

**定级说明：** `rm -rf` 仅作用于 `/tmp/research-cache`（临时目录），按审计规则降为 P1；未发现外送数据行为。**不是恶意技能，但它在这台机器上根本跑不起来**，而且与博客描述的"卡帕西半自动科研"完全不是一回事（它做的是网络调研简报）。

---

## ❓ 查无此技能（2 个）

| 博客名称 | 检索结果 |
|---|---|
| **Auto claudecode** | 市场中无同名技能；`claude-code-*` 系列均为 Claude Code 集成/用量类，与"自动任务完成"无关 |
| **clouddesign** | 市场与 SkillHub 均无匹配。功能相近的替代：`adaptive-presentation-studio`（智能演示文稿生成器，25.8k 下载） |

---

## 📋 审计详细结果

### 危险模式扫描（12 个候选技能全量）

| 检查项 | 命中情况 |
|---|---|
| 远程下载 + 执行（`curl \| bash`） | ✅ **未发现** |
| 敏感文件读取 + 外送 | ✅ 未发现（`auto-research` 读 `secrets.py` 但仅用于连本地 Redis） |
| 破坏性命令 | 仅 `auto-research`（`rm -rf /tmp/research-cache/*`，已降级 P1）；`codex-ppt` 的 `os.remove` 仅删自身临时文件 |
| 权限提升（`sudo`/`chmod 777`） | ✅ 未发现 |
| 隐蔽执行 | `2>/dev/null` 均用于正常静默探测（`redis-cli ping` 等） |
| Prompt 注入话术 | `scientific-visualization` 的 "Critical requirement" 经核实是 **DPI/格式要求**，非攻击载荷；其余 "mandatory/silently" 均为英文文档正常用词 |
| 硬编码凭证 | ✅ 未发现（`codex-ppt` 的 `OPENAI_API_KEY` 为环境变量引用，非硬编码） |
| Base64 混淆载荷 | ✅ 未发现（正则命中项均为普通英文词组） |

### 依赖安装风险

- `codex-ppt`：`codex_ppt_runtime.py bootstrap` 会**创建独立 venv** 后安装 `requirements.txt` —— 符合隔离要求，✅
- `scientific-visualization`：`pip install` 仅出现在报错提示文本中，✅
- `beamer-pipeline-public`：无依赖安装行为，✅

---

## 🎯 与 PR-Warn++ 论文的匹配建议

| 论文环节 | 用哪个已装技能 |
|---|---|
| 正文写作 / 审稿回复 / 去 AI 味 | `paper-spine` |
| 实验图（CRPS/ES/VS 对比、可靠性图） | `scientific-visualization` |
| Fig1 框架图（已有提示词） | `paper-framework-figure-studio-pro` |
| IEEEtran 排版报错排查 | `latex` |
| 会议报告 PPT | `codex-ppt` |
| 学术答辩 Beamer | `beamer-pipeline-public` |

---

## 💡 一句话总结

**博客推荐的 13 个技能里，6 个真正可用且已装好；5 个是空壳/假技能/私有环境绑定，已明确排除。** 其中 `paper-spine` 和 `scientific-visualization` 对当前 TSTE 投稿价值最高。
