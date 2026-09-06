# 有效因子库文件

`library.json`是唯一机器权威；`current.csv`是其自动导出视图。当前库包含2026-09-07全池
正式检验通过的119个因子；历史35条OLS-HAC口径结果与75条单切分记录只保留在原运行目录中。

库只记录通过正式层级FDR及后置门槛的因子。它不保存相关簇、策略子集、权重或生产批准。
相关簇仅存在于具体`runs/factor_selection/<run_id>/`，并随样本合同、方法和阈值解释。

通过根目录`run_factor_workflow.py`依次运行：

1. `VALIDATE_FACTOR_BATCH`或显式`VALIDATE_ALL_INTRADAY`；
2. 人工审阅结果；
3. `ADMIT_COMPLETED_RUN`；
4. 如需组合候选，再运行`SELECT_EFFECTIVE_SUBSETS`。

任何观察run都会被入库校验拒绝。完整规则见`docs/因子检验与准入流程.md`。
