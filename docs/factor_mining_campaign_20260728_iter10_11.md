# Factor Mining Campaign 2026-07-28：Iteration 10–11

## 结论

Iteration 10 和 11 均未发现可进入 observation、WF 或 production 的有效因子：

- iter10 从 iter9 的边界 FDR discovery 出发，强制表达式包含
  `curve_oi_concentration`，结果显示单一终端约束降低了泛化表现；
- iter11 回到普通 `curve_` family 并更换 seed，仍未复现 iter9，且没有达到
  `|IC| >= 0.01` 的经济线。

当前这组实验的最佳证据仍是 iter9 的
`mined_gp_a2e3d995180a65d6` raw H24：factor q=`0.09309`、
`IC=-0.00927`。它只是一项边界统计发现，未越过经济线，不能进入观察期或生产。

## 共同研究边界

- 数据源：阿里云 RDS `ths_data_5minute`。
- GP 训练：5min，2018-01-01 至 2022-12-31。
- 训练 universe：
  `RB,HC,I,J,JM,CU,AL,ZN,PB,NI,SN,AU,AG,SC,FU,BU,TA,MA,EG,FG,V,PP,L,RU,CF,SR,Y,M,P,OI,IF,IC`。
- 目标：H=24 个 5min bar。
- GP：population 48，generations 5，最多保留 25 个候选。
- 开发筛选：2023-01-01 至 2024-12-31，同 32 品种 universe。
- 正式 P0：2023-01-01 至 2024-12-31，正式 47 品种 universe。
- 未使用 2025-07-01 之后 locked OOS。
- taxonomy SHA-256：
  `24f7fca1e10328fdacc64afe2388542d21d5b5ebbb7875c3f45243d44cdd426d`。

## 设置差异

| 轮次 | family 约束 | seed | GP 候选 | 机械通过 | 正式假设 | policy SHA-256 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| iter10 | `curve_oi_concentration` | 78024 | 24 | 10 | 20 | `978da49a6f3e3b3676caf07d3edf39cc977ef2adb810797c519a5b629fc3aaa9` |
| iter11 | `curve_` | 79024 | 6 | 3 | 6 | `e749a9e01da5e5ab9938b15ec6361af9280ac540c6a9c5689e19ef098f7c2e1f` |

iter11 第一次执行被中断，随后以
`gp_rds5m_2018_2022_h24_curve32_s79024_v2` 重启；表中只记录完成的 v2 run。

## 统一漏斗

| 阶段 | iter10 | iter11 |
| --- | ---: | ---: |
| GP 训练候选 | 24 | 6 |
| 机械筛选通过 | 10 | 3 |
| 机械筛选淘汰 | 14 | 3 |
| 正式声明假设 | 20 | 6 |
| 可估假设 | 20 | 6 |
| Simes/BH 选中因子 | 0 | 0 |
| 因子内 local FDR discovery | 0 | 0 |
| 经济线通过 | 0 | 0 |
| observation candidates | 0 | 0 |
| WF candidates | 0 | 0 |
| production approved | 0 | 0 |

P0 证据：

- iter10：
  `runs/factor_research/fm_iter10_rds5m_curveoi32_h24_p0_2023_2024/ic_by_window_period.json`
- iter11：
  `runs/factor_research/fm_iter11_rds5m_curve32_h24_p0_2023_2024_v2/ic_by_window_period.json`

## 最接近候选

| 轮次 | 因子 | 版本 | IC | HAC/OLS t | p | factor q | local q | ols_n |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| iter10 | mined_gp_1a0c7eda600e715b | raw H24 | -0.005893 | -2.150 | 0.03159 | 0.52517 | 0.05252 | 30191 |
| iter10 | mined_gp_1a0c7eda600e715b | neutralized H24 | -0.003799 | -1.939 | 0.05252 | 0.52517 | 0.05252 | 30191 |
| iter10 | mined_gp_99a8a43e65538a60 | neutralized H24 | 0.005762 | 1.710 | 0.08727 | 0.75357 | 0.17453 | 30191 |
| iter11 | mined_gp_86e8ecbfc1cd594a | neutralized H24 | 0.008804 | 1.244 | 0.21342 | 0.62546 | 0.42684 | 30191 |
| iter11 | mined_gp_d36901becf2fd691 | neutralized H24 | 0.005488 | 0.706 | 0.48029 | 0.62546 | 0.62546 | 30191 |
| iter11 | mined_gp_86e8ecbfc1cd594a | raw H24 | 0.005242 | 0.606 | 0.54457 | 0.62546 | 0.54457 | 30191 |

iter10 正式 P0 的 IC/OLS-HAC 阶段耗时 `698.3s`，对应 10 个因子、
20 个 raw/neutralized 假设和 47 品种的 2023–2024 5min 数据。

## 后续判断

这两轮共同否定了“继续强制单一 curve 终端即可复制 iter9”的假设。后续应：

1. 合并 iter9/10/11 候选池后统一去重治理和正式复测；
2. 对 iter9 表达式族只做小范围结构扰动；
3. 扩充品种或拉长历史，而不是继续放松 q 或恢复 Bonferroni 硬门槛；
4. 优先降低宽 universe 重复评估成本，保持完整假设分母和冻结研究边界。
