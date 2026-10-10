# PR-Warn++ 工作区

PR-Warn++ 是一个面向**极端条件与退化数据下 10–60 分钟联合概率风电功率预测**的
可复现研究实现（当前版本 v3.2）。

## 目录结构

| 路径 | 说明 |
|---|---|
| [`PR-Warn++/`](PR-Warn++/README.md) | 项目本体（git 仓库）：源码、测试、配置、`docs/` 文档 |
| [`PR-Warn++/legacy/`](PR-Warn++/legacy/README.md) | 项目内历史版本（早期方案、v3.1、叙事冻结版 3.0） |
| [`archive/`](archive/README.md) | 历史打包快照（仓库/原型/基准结果的压缩包留档） |

## 从哪里开始

- 了解项目与论文：进入 [`PR-Warn++/`](PR-Warn++/README.md)，阅读其 `README.md`
- 完整中文操作手册（数据下载、环境、预处理、训练、校准、压测、源码职责）：
  `PR-Warn++/docs/PR-Warn++_v3.2_代码使用说明.md`
- 找回旧版本或历史结果：见项目内 [`legacy/`](PR-Warn++/legacy/README.md) 与外层 [`archive/`](archive/README.md)

## 主链路

```text
raw SCADA
  -> 刚性时间网格 + X_fill/M/delta_t/reason_code
  -> 密度修正经验功率曲线 P_pc
  -> 动态多图残差预测器 R_hat
  -> 确定性中心 Y_det
  -> 直接条件流匹配（Direct CFM）于 E = Y - Y_det
  -> 分机/分时长联合场景
  -> 分时长 split conformal 区间
  -> 爬坡/偏差/CVaR/OOD/数据质量代理指标
```

> 大数据集与训练权重刻意不入库；输出为风侧运行/爬坡/尾部风险**代理指标**，
> 不构成关于线路过载、电压频率稳定或真实调度成本的结论。
