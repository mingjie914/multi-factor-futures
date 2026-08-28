# 运行产物保留规则

`runs/` 只保存运行证据，不是生产配置。正式研究按不可变 study/run 目录输出，避免
覆盖后失去数据边界、参数和哈希；这类时间戳目录是审计设计，不应改成可变 `latest/`。

## 永久保留

- `factor_research/holdout_ledger.jsonl`：只追加的已消费 holdout 记录，禁止删除、
  修改或把其中日期重新称为未见样本。
- `factor_validation/20260824_intraday588_is126_oos42_cutoff_20260515/`：本次588个
  日内因子的标准准入证据。保存588行全量明细、75行通过明细、IS筛选产物、OOS统计、
  摘要和文件哈希合同；研究窗口固定截至2026-05-15。

以后同类检验统一写入`factor_validation/<run_id>/`，至少包含
`factor_validation_full.csv`、`passed_factors.csv`、`validation_summary.json`、
`oos_factor_ic.json`和`run_contract.json`，逐假设IS证据统一放入`artifacts/`。不得为单次
需求在项目根目录新增一次性脚本或散落结果。

有效库之后的周期匹配、相关性去重和并行子集筛选统一由
`workflows/factor_selection.py`写入`factor_selection/<run_id>/`，至少保留
`factor_diagnostics.csv`、`factor_clusters.csv`、`factor_sets.json`、
`selection_summary.json`、相关矩阵和`run_contract.json`。该目录是可重放的选择证据，
不回写有效因子库，也不替代策略库目录。

确定组合回测统一写入`portfolio_backtest/<run_id>/`。由IDE入口
`run_portfolio_workflow.py`运行时，每个策略保存独立结构化结果，run根目录保存配置路径与
哈希、策略库完整快照与哈希、横向指标、`segment_comparison.csv`、
`portfolio_report.md`、`nav_comparison.csv`及多曲线图；run目录不得覆盖复用。
横向比较入口固定采用`COMPARISON_START=2024-01-01`至数据源最新完整交易日，
并调用原有`BacktestResult.plot`/`MultiPortfolioResult.plot`生成各策略图；根目录的
`nav_comparison.png`只负责多策略叠加和指标表，沿用同一中文绘图口径。旧10因子观察策略
单独走`legacy_production_portfolio`路由；IDE的默认`RUN_AND_COMPARE`会让所有选定平行
策略走同一`config/default.yaml::production_portfolio`生产账本（ICIR + Top10/Bottom10 +
cap3 + ERC、总敞口2），只改变因子集合，并在内存中使用365日面板预热。原独立配置不被
改写，缺失暴露不做比较专用的零值填充。只有显式切换`RUN_AND_COMPARE_CONFIGURED`才读取
候选YAML中的模型、风险与优化器参数；该分支用于方法挑战，不改变默认生产基线，并记录
有效库方向文件的哈希。

归档的 6f/8f/13f 定义登记在同一策略目录但不参与普通比较；
`RUN_AND_COMPARE_SNAPSHOT_AUDIT` 会读取其不可变 `factor_definition_path`，使用当前
DuckDB、当前 T+1 账本和默认生产方法重新评价，并按同一格式写入新的回测 run。
6f 等权与 6f ICIR、8f 等权影子与 8f ICIR 的因子定义分别相同；当前审计固定方法时只按
不同因子集合各跑一次，避免把同一组合重复计入横向结果。
若某个快照触发严格数据/因子可用性门禁，run 仍会完成其余快照，失败策略写入
`<strategy>/failure.json`，根目录写入 `snapshot_failures.json`，横向表和中文图以“未形成”标记，
不以零值或填充结果冒充净值。

需要把当前三套策略与 6f/8f/13f 一次性放在同一张图时，IDE 入口选择
`RUN_AND_COMPARE_ALL`；它是显式六策略比较分支，仍只使用默认生产方法，不改变普通
`RUN_AND_COMPARE` 对归档快照的排除规则。

`RUN_AND_COMPARE_ALL` 不代表 5/10/20 多周期子组合。只有
`RUN_AND_COMPARE_CONFIGURED` 才按策略 YAML 的 `sub_portfolios` 运行实际多周期 sleeve，
但该分支同时切换 YAML 中的 alpha、risk、rebalance 与 meta-optimizer，属于完整配置挑战。
共同 H5 的独立证据保存在 `factor_validation/20260826_intraday588_common_h5.../` 和
`factor_selection/20260826_common_h5_subset_selection/`；
`RUN_AND_COMPARE_COMMON_H5` 使用默认日度 IC，
`RUN_AND_COMPARE_COMMON_H5_MATCHED` 使用统一 H5 IC，二者均不写回有效因子库，且运行到
最新数据日期。两条分支仅用于拆分“因子集合”和“IC 标签周期”影响，不能替代真正的
5/10/20 配置 sleeve 运行。

## 历史审计证据

- `factor_research/20260820_intraday599_rebuild/`：截至2026-08-20的全历史正式单因子
  研究证据。运行提交599个历史类，588个可估计；形成20个统计发现、13个相关簇，
  发布批准数为0。目录保留筛选结果、验证漏斗、相关矩阵JSON和相关图，不保存可再生
  检查点。
- `external_guosen_trend_index/20260817_correctness_rebuild/`：最终频率路由与因子实现
  修复前的固定集合审计资料；其中绩效不得作为当前策略结论。

下列旧运行已因分钟根前缀污染或组合口径不一致而删除，不得从外部副本恢复为当前证据：
`20260810_full_prod_sort`、`20260813_contract_symbol_fix`、
`20260815_13f_8f_union`、`20260815_latest_core_compare`和
`20260816_correctness_rebuild`。

当前固定定义位于 `snapshot/`；快照和运行证据均不会发布订单。
`config/target_publication.yaml` 是独立的目标权重发布门，关闭时固定为 `NO_TARGETS`。

## 清理规则

- 中断且无结论的目录、根目录散落图表、被修正版替代的结果可以直接删除。
- 普通研究目录在结论失效后可清理；仍需长期审计的证据应先保存 manifest、哈希和
  外部只读归档位置。
- 完成的工作流必须删除 `_factor_panel_cache.pkl` 等可再生中间缓存。
- 研究入口不得把图、CSV 或 JSON 直接写到项目根目录；输出必须落入显式 run 目录。
