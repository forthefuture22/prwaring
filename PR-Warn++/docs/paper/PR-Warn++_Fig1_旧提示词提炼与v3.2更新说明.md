# PR-Warn++ Fig. 1 旧提示词提炼与 v3.2 更新说明

## 两份旧提示词的共同优点

- 都采用 A--D 四阶段横向结构，适合 IEEE TSTE 双栏宽图。
- 都把训练、推理、校准和 oracle/evaluation 路径用不同线型分开。
- 都抓住了最重要的科学对比：`R = Y - P_pc` 与 `E = Y - Y_det`。
- 都要求把联合场景、边际校准和事件/尾部风险拆成三个输出分支。
- 都明确禁止把未来 ERA5 接入主在线路径，也禁止将风侧风险写成电网安全状态。

## 两份旧提示词的差别

“精简形象版”更适合直接生图：视觉隐喻多、文字量受控、强调缩小后的可读性；“高级提示词”更适合作为技术审查清单：公式、张量、训练/推理路径和理论边界更完整，但若原样生图容易变成文字密集的海报。

新版采用“精简版负责画面，高级版负责自检”的组合策略。

## v3.2 必须更新的内容

1. 将旧的 `Graph-Recurrent Residual Model` 改为与当前代码一致的 `Temporal Conv + Multi-Graph Residual Blocks`。
2. 将原因码作为审计/压力测试 sidecar，而不是虚构为当前骨干的直接输入。
3. 将密度修正限定为预测起点可获得的压力/温度，不依赖未来 ERA5。
4. 主 CFM conditioner 收敛为确定性隐藏表示 `H_t` 与融合图 `A_t`；缺失和天气信息通过确定性表示间接进入。
5. 保留标准 independent-pair CFM 为主线；真实 minibatch OT 只标作 A5 消融，VAE-CFM 只标作 A4 消融。
6. 将主 conformal 输出改为当前配置对应的 farm-level per-horizon interval `C_h^farm`。
7. 删除尚未实现的 density-ratio weighted conformal。
8. 增加 A7 ACI、A8 context/OOD heuristic 的“非主线、不同保证”提示。
9. 增加 A10 的 OOD/data-quality 冻结组合支线，但不把它提升为主要创新。
10. 将 A11 表示为带 `issue_time <= forecast_origin` 的可审计 as-of join；未来 ERA5 继续保持 detached oracle branch。

## 新版图的叙事顺序

`available history -> rigid grid -> weak P_pc anchor -> temporal multi-graph R_hat -> Y_det -> direct CFM on E -> joint scenarios -> three separated reliability objects`

其中前三个最醒目的视觉锚点应是：

1. `Y_det = clip(P_pc + R_hat)`；
2. `E = Y - Y_det`；
3. `joint fidelity != marginal coverage != event/tail reliability`。
