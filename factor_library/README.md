# 有效因子库文件

`library.json`是唯一机器权威；`current.csv`是其自动导出视图。截至2026-09-13，当前库167条：
2026-09-08批次的151条，加2026-09-10量价批次通过的3条及13个GP改写因子。各条记录保留自身检验批次，
不将不同检验族的q值混作一次全池检验结果。历史119、35、75条版本只作为对应阶段证据。

库只记录通过正式层级FDR及后置门槛的因子。它不保存相关簇、策略子集、权重或生产批准。
相关簇仅存在于具体`runs/factor_selection/<run_id>/`，并随样本合同、方法和阈值解释。

#491仅将`best_period`标注更新为H10；`best_period_review`记录原H20及复核证据哈希。
其IC/t/q仍属原准入快照，不是H10重算值；不得将两者配对作新检验结论。
这不改变方向、调仓或持有期。当前完整说明见`docs/有效因子库.md`。

通过根目录`run_factor_workflow.py`依次运行：

1. `VALIDATE_FACTOR_BATCH`或显式`VALIDATE_ALL_INTRADAY`；
2. 人工审阅结果；
3. `ADMIT_COMPLETED_RUN`；
4. 如需组合候选，再运行`SELECT_EFFECTIVE_SUBSETS`。

任何观察run都会被入库校验拒绝。完整规则见`docs/因子检验与准入流程.md`。
