# 运行产物保留规则

runs/只保存证据，不是配置或代码目录。按<流程>/<run_id>/分类，每次正式运行使用新目录，
不覆盖为可变latest目录；大型数据与普通输出留本地，不提交Git。

| 目录 | 职责 |
|---|---|
| factor_validation | 正式检验合同、逐假设统计、通过列表及性能画像 |
| factor_selection | 库内搜索、相关簇、候选成员及分阶段验证 |
| portfolio_factor_search | 显式邻域/配方及全池预算研究；不自动登记策略 |
| portfolio_backtest | 固定成员同口径账本、净值、指标及中文图 |
| factor_research | 探索/迁移研究、只追加holdout账本 |
| factor_mining | 显式挖掘产物及不可变候选快照，不由默认入口主动加载 |
| walkforward / historical_portfolio_search | 显式组合验证、隔离历史实验 |

## 当前证据索引（更新至2026-09-18）

以下目录及其合同递归引用的面板、报告、验证结果均保留；不是可清空的临时缓存。

| 阶段 | 本地run | 当前意义 |
|---|---|---|
| 新登记10组成本与缓冲验证 | portfolio_backtest/20260915_retained10_cost_buffers | 30/30成功、10项B0逐日精确复现；6项有历史资产明细者也精确复现；29因子/130席位；中文报告、B0总净值、逐组缓冲净值、热力图、类型分布及成本曲线；默认缓冲仍0，不自动采纳 |
| 新增21因子与交易缓冲 | portfolio_factor_search/20260914_incremental21_trade_buffers | 292项完成、0失败，10项B0精确复现；17项增量诊断合格、5项短名单；中文报告、总净值/逐组缓冲图及2张热力图；188库、10策略和默认不变，不自动采纳 |
| 冻结5项增量的缓冲交互 | portfolio_factor_search/20260914_shortlist5_trade_interactions | 15项完成、0失败，5项B0逐日精确复现；中文报告、净值、缓冲热力图、原10组换手来源/成本曲线；不自动采纳 |
| 188有效库复用两条旧选集方法 | portfolio_factor_search/20260914_effective188_repeat_methods | 已验收；A57/B512成功，各5组候选，20项完整原生对照；原10基准精确一致，不改变库或策略目录 |
| 排名缓冲共享接入验收 | portfolio_factor_search/20260915_rank_buffer_integration | 已完成；默认0，25项日账本/资产及10组目标精确一致，1079项回归；通用Polars/Rust权重计算受控计时；分段摘要修正见acceptance |
| 新增98因子准入 | factor_validation/20260914_intraday98_admission | 98/294统一假设族；20个正式通过并追加入库167→187；library_admission.json记录登记回执 |
| 12项正式准入 | factor_validation/20260914_gap12_admission | 12/36假设；1项通过并按授权入库187→188；library_admission.json为回执，report.md保留检验及编号方案 |
| 编号统一与登记收口 | factor_research/20260914_numbering_closeout | 早期34项原地补#700–733；baseline.json/acceptance.json记录三种内核实值一致性，report.md记录清理与下一阶段建议 |
| 12个原创补位/34项旧因子复核 | factor_research/20260914_gap12_extension | 原始设计及计算验收；后续正式结果见上一行；旧34中6项原已入库、28项未过FDR |
| 原151条全池准入 | factor_validation/20260908_intraday605_datarepair_ic_hac | 原605/1815假设族；不是623因子全量再认证 |
| 追加3条准入 | factor_validation/20260910_daily_volume_corr5 | 151→154的正式批次证据；后续加13个改写因子为167 |
| 13个改写因子检验 | factor_validation/20260910_intraday_structure_migration | 2026-09-13显式提取规范名通过者入库；完整检验族与原q值保留 |
| 原8组目录审阅 | factor_selection/20260909_eight_strategy_review | 冻结成员与登记证据；当前数值见后续固定验证 |
| 邻域/96配方研究 | portfolio_factor_search/20260911_native14_neighborhood96 | 规划和已执行子集，不宣称全部笛卡尔积完成 |
| 后续全池预算搜索 | portfolio_factor_search/20260911_effective_pool_multistart | 512次尝试、5个研究候选；含修复前因子，原路径未重跑 |
| 原13组归因 | portfolio_backtest/20260911_thirteen_strategy_validation | 修复前快照；旧8控制和未变删除签名的对照来源 |
| 无感工程验收 | portfolio_backtest/20260912_engineering_parity | 性能改动前后13组账本精确一致，与因果修复分开 |
| 当前固定13组修复 | portfolio_backtest/20260912_flow_residual_repair | 已审阅report、净值/热力图、原生账本、188项剔除和局部准入复核 |
| 当前10组交付 | portfolio_backtest/20260913_active_strategy_review | 截至2026-09-11的报告、净值、热力图、类型/簇及通用成本压力；历史剔除证据另行标明 |
| 当前10组延长重算 | portfolio_backtest/20260913_active_strategy_latest | 10组同口径回测已完成；原生账本、重叠对照、性能及费用字段审阅记录；原区间净值逐点一致 |

表中各历史批次及数量保留当时含义；原十组交付看相应目录的report.md、report_contract.json和acceptance.json。
新增曲线精萃两项接入证据在`portfolio_backtest/20260918_curve_sleeve_registration/`；
其原始冻结研究在项目本地`comparison_research/20260918_full_library/`，均不入Git。
只做多扩展及后续负面诊断在`strategy_extension/`，详见docs的趋势配置扩展与验收；
被后续合同引用的面板、系数、脚本、账本和失败证据一并保留，不只保留胜出图表。
详细分类/归因不是实盘批准。当前默认中性比较为12组，正式策略不变；
归档证据及旧run内状态描述保留为当时快照，当前状态以策略目录和有效库文档为准。
日常入口仍是根目录两个workflow；上述研究由workflows/experiments中的显式分支执行，
不会挂入默认流程。归因模块保留是为了复现真实证据，不是待删除的一次性调试脚本。
对已审阅报告不要直接使用report-only覆盖人工审阅；图表或报告重生成应另行审阅。

## 保留

- factor_research/holdout_ledger.jsonl：只追加，不删除，不把已消费日期标为未见样本。
- 被当前有效库、选择或比较合同引用的run：保留完整证据和哈希。
- 历史九组对照：portfolio_backtest/20260909_candidates5_vs_legacy_latest/，
  及其引用的20260909_candidates5_vs_legacy_same_contract/。
- 旧119/35/75及迁移、共同H5实验仅作历史审计，不可用于当前入库。

正式准入保存full/passed表、summary、run_contract及artifacts。
组合输出保存账本、横向指标、中文净值图及实际日期/方法合同；
失败明确记录，不填零伪造结果。修补合同内数据必须重新核验。

## 整理与清理

日志归入对应run；临时脚本完成后删除，复用功能回到既有模块，不维护第二套入口。
2026-09-13将两份已完成的频率对照/迁移审计脚本封存至本地只读
factor_research/archived_audit_scripts_20260913.zip（保留原相对路径，逐文件SHA-256核对），
再移出原执行位置；需要重放时可恢复到原目录。检验结果、候选快照及合同不变。
同次测试缓存递归清理被本机策略阻止，部分.test_tmp和.pytest_cache目录另有权限限制，
均仍保留并由Git忽略；没有修改权限或把它们误记为已清除。
最终交付时再次确认没有新的一次性脚本或未完成.tmp产物；根目录字节码缓存删除亦被
本机策略拦截，仍保留且不入Git。当前报告按用户要求覆盖展示版本，原始审计证据保留。
只清理已确认未引用、可重建的临时文件；有价值失败实验保留说明，不作为有效结果。
不得删除数据源、有效库、holdout账本、正式证据或用户手动快照。
操作前验证绝对路径和范围，权限受阻时保留并报告。
2026-09-18发布审阅的临时Git暂存副本位于`release_review/20260918_staged/`，
其清理也被本机策略阻止，仍保留且不入Git；对应测试XML独立保存，不将副本当作新的研究版本。

历史九组对照使用canonical_factor_panel_checkpoint。
该历史目录中的factor_panel_checkpoint是已否决的短预热实验，不可续跑；
删除曾受执行策略阻止，尚保留本机。重放必须使用合同约定的全局分块边界，
核对重叠期数值及缺失掩码，不能仅按预热天数认定缓存等价。
