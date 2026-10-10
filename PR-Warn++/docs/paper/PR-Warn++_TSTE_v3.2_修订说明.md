# PR-Warn++ TSTE v3.2 修订说明

> 路径说明（目录重组后）：本文档现位于 `docs/paper/`；文中提到的旧 `3.2/`
> 即当前目录，旧 `3.0/` 已移至 `legacy/3.0/`。

## 版本基线

- 旧 PDF 与旧 TeX 的主体内容一致；PDF 不是一个独立的更晚正文版本。
- 旧 TeX 的 `\bibliography{...}` 名称与实际 `.bib` 文件名不一致，无法按原目录直接稳定重编译。
- 新稿独立放在 `3.2/`，不覆盖 `3.0/` 文件。

## 逐段层面的主要修订

### 标题与摘要

- 缩短标题并突出 availability、weak physics、final-error flow matching 和 ramp-risk assessment。
- 删除“calibrated scenarios”式混合表述，改为 joint scenarios 与 conformalized marginal intervals 分开。
- 不填入任何未经服务器验证的结果数字。

### 第一节 Introduction

- 从“模块清单”改为“预测对象和信息边界”驱动的论证。
- 统一为 3 项方法贡献 + 1 项协议贡献。
- 强化 `R` 与 `E` 的不同科学含义。
- 将动态图、功率曲线、生成模型和 conformal 降为支撑组件，而不是各自单独宣称创新。

### 第二节 Related Work

- 重组为 availability/weak physics、joint generation、calibration/risk 三条文献线。
- 补入缺失感知的 GRU-D 基础引用。
- 保留经 DOI/出版元数据核对的 2025--2026 TSTE 与 Applied Energy 工作。
- 修正风险场景文献的卷期页码。

### 第三节 Problem Formulation and Proposed Framework

- 将旧稿的 Section III 与 Section IV 合并，符合“前三个主体 Section 完成方法定义”的 TSTE 写法。
- 把骨干从旧的 graph-recurrent 叙述改为当前实现的 temporal convolutions + residual graph blocks。
- 明确原因码是审计 sidecar；模型主输入为 `X_fill`, `M`, `log(1+Delta t)`。
- 密度修正只使用预测起点可获得的压力/温度。
- CFM conditioner 按代码修正为确定性隐藏表示和融合邻接图。
- 主方法明确为 independent-pair CFM；Hungarian minibatch OT 为 A5，VAE-CFM 为 A4。
- 主 conformal 公式改为 farm-level per-horizon split conformal，与当前配置和评估代码一致。
- 删除未实现的 density-ratio weighted conformal 方法主张。
- 增加 A7/A8 的保证边界和 A10/A11 的辅助协议定位。
- 修正 ramp 首步定义：使用预测起点的观测农场功率作为 `h=0` 前缀。

## 引用与模板

- 使用 `\documentclass[journal]{IEEEtran}` 与 IEEE Transactions 双栏结构。
- 新参考文献文件使用 ASCII 文件名 `prwarn_tste_v3_2_refs.bib`，避免旧版路径/名称不一致。
- BibTeX 仅保留当前稿实际需要的来源，并补充 DOI、卷期和页码。
- 初次投稿仍需将全文控制在 10 页以内；当前无服务器结果的稿件不应被当成可提交终稿。

## 强化审计（2026-09-01）

- 新增 `PR-Warn++_TSTE_v3.2_逐段修订与引用审计.md`，逐段记录旧稿问题、v3.2 改写和代码/理论依据。
- 对正文实际引用的 37 条来源完成引用位置、论断强度和出版元数据闭合审计；引用键、BibTeX 条目均为 37，未定义和未使用条目均为 0。
- 将风场集群的风速/风向相关图与机组级 wake-directed graph 分开表述，避免跨空间尺度过度概括。
- 将 CQR 引用限定为 asymmetric interval-score construction，不把 scenario-quantile wrapper 错写成训练式 CQR。
- 增加 NREL WIND retrospective reforecast 报告，并明确 A11 仍需实际 issue/valid timestamp 审计。
- Fig. 1 已替换为可编辑 SVG 和嵌入字体的矢量 PDF；论文不再引用旧 PNG。最终图在双栏整宽下完成逐页检查。

## 仍需服务器完成

- 冻结数据审计、五种子训练和模型选择；
- 所有主表、消融、置信区间与统计检验；
- A11 真实 issue-time weather archive；
- GPU 延迟、吞吐、显存和参数统计；
- 根据实验结果回填 Abstract、Results 和 Conclusion。
