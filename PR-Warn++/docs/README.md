# docs — PR-Warn++ v3.2 文档

本目录存放项目的**当前最终版（v3.2）文档**。历史版本见上级目录 [`legacy/`](../legacy/README.md)。

## 文档导航

| 文档 | 用途 | 读者 |
|---|---|---|
| [`paper/`](paper/PR-Warn++_TSTE_v3.2_修订说明.md) | 最终论文 TSTE 源：`PR-Warn++_TSTE_v3.2.tex`、参考文献 `prwarn_tste_v3_2_refs.bib`、`figures/`、修订说明与逐段引用审计、Fig.1 提示词 | 投稿 / 写作 |
| `PR-Warn++_v3.2_最终研究方案_干净版.docx` | v3.2 最终研究方案与可复现技术实施规范（Clean v3.2） | 理解方法全貌 |
| `PR-Warn++_v3.2_从零理解完整阅读版.docx` | 从零开始的通俗完整阅读版，与 v3.2 主线逐层同步 | 入门 / 精读 |
| `PR-Warn++_v3.2_代码使用说明.md` | 完整中文操作手册：数据下载、环境、预处理、训练、校准、压测、每个源码/CLI 职责 | 动手跑代码 |
| `论文前三章修订基线_v3.2.md` | 论文前三章的修订基线与技术核对要点 | 写作 |
| `SERVER_RUNBOOK.md` | 服务器运行手册（八个 Gate，禁止在 Test 上调后续阶段） | 服务器执行 |
| `IMPLEMENTATION_STATUS.md` | 当前实现状态：已完成、已本地验证、仍需服务器验证的边界 | 了解进度 |
| [`tool-research/`](tool-research/SCI论文写作Skill与工具调研.md) | SCI 写作 Skill/工具调研、学术 AI 技能安装报告 | 工具选型 |

## 论文编译

`paper/` 内为 IEEEtran 双栏稿，图片与文献均为相对路径（`figures/`、
`prwarn_tste_v3_2_refs.bib`）。编译产物统一输出到 `output/pdf/`（该目录已被
`.gitignore` 忽略）；当前已有的最终编译 PDF 即
`output/pdf/PR-Warn++_TSTE_v3.2.pdf`。

```bash
cd docs/paper
pdflatex -output-directory=../../output/pdf PR-Warn++_TSTE_v3.2
bibtex ../../output/pdf/PR-Warn++_TSTE_v3.2
pdflatex -output-directory=../../output/pdf PR-Warn++_TSTE_v3.2
pdflatex -output-directory=../../output/pdf PR-Warn++_TSTE_v3.2
```

> 论文正文含 `\TODO{}` 占位，需在服务器跑完五种子实验后回填结果；
> 当前无实测结果的稿件不是可提交终稿。
