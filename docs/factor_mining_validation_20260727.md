# 1 分钟 GP 因子正式复核（2026-07-27，历史旧口径）

> 本文冻结的是 2026-07-27 当时的批次 Bonferroni + `|IC|>=0.02` 结果，只用于历史
> 对照，不能代表当前 Validation Policy v2 的通过数。v2 使用层级 FDR、`|IC|>=0.01`、
> `|t|>=2`、自然年/成本/观察期治理；重跑结果必须写入新目录，不覆盖本文证据。

## 结论

本次当前口径复核没有产生可进入主框架待选组合的有效因子。六个原始挖掘周期共正式
检验 179 条候选，179 条均有统计观测，2 条通过各自批次的 Bonferroni 显著性门槛，
但没有候选同时达到 `|IC| >= 0.02`、IC 命中率 `>= 52%` 和显著性要求。

这不是零样本或流程中断造成的结论。动态流动性品种池的夜盘交易日映射已修复，研究
流程也会在整批零有效观测时失败关闭。本次每个 horizon 均完成了非零正式检验。

## 冻结边界

- 数据：只读本地 1 分钟 Parquet，未访问 MySQL、阿里云或 DolphinDB。
- 品种：47 个期货品种；每月使用滞后流动性和板块约束选择 32 个品种。
- 因子预热：2023-01-01 至 2024-03-31。
- 评价区间：2024-04-01 至 2024-12-31。
- 目标：每条公式只检验其原始挖掘 horizon；周期单位均为 1 分钟 bar。
- 处理：MAD 缩尾、波动率中性化、板块中性化和横截面标准化。
- 推断：未收缩单因子 OLS/HAC p 值，批次内 Bonferroni，再执行 IC 与命中率门槛。
- 隔离：未读取 2025-07-01 起的 locked OOS，未修改交易配置。

## 六周期结果

| H (分钟 bar) | 候选/有效 | Bonferroni 显著 | 完整通过 | 最大绝对 IC | 该项样本数 | 批次最小 p 值 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 24 / 24 | 1 | 0 | 0.004563 | 62,476 | 0.00108395 |
| 5 | 4 / 4 | 0 | 0 | 0.023125 | 865 | 0.0956439 |
| 15 | 50 / 50 | 1 | 0 | 0.014872 | 19,147 | 0.000584297 |
| 30 | 25 / 25 | 0 | 0 | 0.007047 | 62,116 | 0.0634293 |
| 60 | 51 / 51 | 0 | 0 | 0.010116 | 62,475 | 0.00159496 |
| 240 | 25 / 25 | 0 | 0 | 0.015634 | 30,501 | 0.373042 |

H=5 的最大 IC 看似超过 0.02，但只覆盖 865 个分钟截面，OLS/HAC p=0.3745，不能视为
稳定信号。H=240 的最佳候选同时有 54.85% 命中率，但 p=0.3730，同样不显著。

两条 Bonferroni 显著项为：

- H=1 `mined_gp_fast_numeric_sector_h1_3e9ad02a5d69`：IC=0.003541，HAC t=3.2679，
  p=0.00108395，命中率=50.69%，n=62,471。
- H=15 `mined_gp_conditional_logic_h15_ebfc5c03191b`：IC=-0.006504，HAC t=-3.4390，
  p=0.000584297，命中率=48.33%，n=62,447。

两者都是“大样本下统计显著、经济效应很小”的典型情形，因此没有晋级。

## 候选池状态

SQLite `runs/factor_mining/candidates.sqlite3` 已通过 `PRAGMA integrity_check`。本轮结束时
共 273 条记录，全部为 `rejected`：包括正式门槛失败、预筛机械失败和结构卫生失败。
主框架的 `config/default.yaml` 仍保持 `factors: []`，交易配置未启用。

这并不删除候选。公式、血缘、评估记录、冻结快照和正式输出仍保留，可用于复盘，但
不会被误当成待晋级候选。

## 原始证据

- H=1：`runs/factor_research/gp_1m_dev_primary_h1_sector_current_v2_2024q2q4_20260727/raw/ic_by_window_period.json`
- H=5：`runs/factor_research/gp_1m_dev_primary_h5_sector_current_v2_2024q2q4_20260727/raw/ic_by_window_period.json`
- H=15：`runs/factor_research/gp_1m_dev_primary_h15_sector_current_v2_2024q2q4_20260727/raw/ic_by_window_period.json`
- H=30：`runs/factor_research/gp_1m_dev_primary_h30_sector_current_v2_2024q2q4_20260727/raw/ic_by_window_period.json`
- H=60：`runs/factor_research/gp_1m_dev_primary_h60_sector_current_v2_2024q2q4_20260727/raw/ic_by_window_period.json`
- H=240：`runs/factor_research/gp_1m_dev_primary_h240_sector_current_v2_2024q2q4_20260727/raw/ic_by_window_period.json`

## 下一轮改进

不应通过放宽门槛或在多个 horizon 中事后挑最佳来制造“有效因子”。下一轮更合理的改进
是提高搜索目标与正式门槛的一致性：在 GP 适应度中加大跨时间段稳定性、最小有效覆盖、
换手和成本后收益的权重，并对稀疏条件表达式设置更严格的覆盖惩罚。随后以新的冻结训练
区间、随机种子和 run ID 进行独立搜索，再按原始 horizon 正式检验。
