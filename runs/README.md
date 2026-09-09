# 运行产物保留规则

runs/只保存证据，不是配置或代码目录。按<流程>/<run_id>/分类，每次正式运行使用新目录，
不覆盖为可变latest目录；大型数据与普通输出留本地，不提交Git。

| 目录 | 职责 |
|---|---|
| factor_validation | 正式检验合同、逐假设统计、通过列表及性能画像 |
| factor_selection | 库内搜索、相关簇、候选成员及分阶段验证 |
| portfolio_backtest | 固定成员同口径账本、净值、指标及中文图 |
| factor_research | 探索/迁移研究、只追加holdout账本 |
| walkforward / historical_portfolio_search | 显式组合验证、隔离历史实验 |

## 保留

- factor_research/holdout_ledger.jsonl：只追加，不删除，不把已消费日期标为未见样本。
- 被当前有效库、选择或比较合同引用的run：保留完整证据和哈希。
- 当前准入：factor_validation/20260908_intraday605_datarepair_ic_hac/（151入库）。
- 当前对照：portfolio_backtest/20260909_candidates5_vs_legacy_latest/，
  及其引用的20260909_candidates5_vs_legacy_same_contract/。
- 八组分析：factor_selection/20260909_eight_strategy_review/。
- 旧119/35/75及迁移、共同H5实验仅作历史审计，不可用于当前入库。

正式准入保存full/passed表、summary、run_contract及artifacts。
组合输出保存账本、横向指标、中文净值图及实际日期/方法合同；
失败明确记录，不填零伪造结果。修补合同内数据必须重新核验。

## 整理与清理

日志归入对应run；临时脚本完成后删除，复用功能回到既有模块，不维护第二套入口。
只清理已确认未引用、可重建的临时文件；有价值失败实验保留说明，不作为有效结果。
不得删除数据源、有效库、holdout账本、正式证据或用户手动快照。
操作前验证绝对路径和范围，权限受阻时保留并报告。

最终九组对照使用canonical_factor_panel_checkpoint。
同目录factor_panel_checkpoint是已否决的短预热实验，不可续跑；
删除曾受执行策略阻止，尚保留本机。重放必须使用合同约定的全局分块边界，
核对重叠期数值及缺失掩码，不能仅按预热天数认定缓存等价。
