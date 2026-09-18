# 期货多因子研究框架

本地期货因子研究、有效库管理与组合回测框架，默认不发布交易目标。

截至2026-09-18：intraday注册733个因子，权威有效库188条（含13个GP改写因子及1个SPEC迁移因子），
当前中性比较组12组：正式主策略 **多源韧衡(18)/B1**＋11组观察备选；均由有效库成员构成。
新增曲线精萃子组合等权(9)、曲线精萃子组合日度ERC(9)，复用同一九因子成员；不是新增因子或实盘批准。
另登记12持仓品种研究版本**多源精择12品种(18)**（`multi_source_resilient_p12`），共用原18因子，仅显式调用，不纳入默认12组；见[策略目录](docs/策略候选库.md)。
同一策略目录另登记3组固定九品种只做多观察策略，方法配置独立，默认比较按组隔离；见[趋势配置扩展与验收](docs/只做多趋势配置策略扩展方案.md)。
实际默认方法为 `lw_abs`（IC均值与Ledoit-Wolf协方差求解后取绝对值）＋
Top10/Bottom10＋五大分组cap3＋分侧ERC；不与另一选项`diag_icir`混称。
实际持仓排名缓冲`production_portfolio.rank_exit_buffer`全局默认0；原10组在策略目录中
显式采用五组关闭、四组B1、一组B2，正式主策略多源韧衡采用B1；
新增两组在各单因子子组合内采用B1，外层日度等权/ERC后净额合并及一次计费；不把最终并集再裁成20品种。
规则与验收见[策略目录](docs/策略候选库.md)及[交易成本与换手口径](docs/交易成本与换手口径.md)。
目录`formal`表示用户指定的正式策略身份，`preferred`表示默认运行选择；
均不等于实盘批准，日常仍只回测/观察，目标权重发布门保持关闭（NO_TARGETS）。

本轮98个新增因子统一正式检验已完成，20个已按授权追加有效库；部分经组合研究后进入备选池。
另原创12个补位因子已完成正式检验：1项按授权登记、11项未过FDR。
编号已统一为#1–733，与733个注册因子一一对应；早期34项原地补为#700–733，
注册名、公式、周期契约及位置不变。编号仅供检索，注册名仍是持久身份。
检验报告与性能记录见[产物索引](runs/README.md)。

2026-09-15批准固定新10组备选池（联合29因子），旧9组归档、精简板块继续保留。
基础方法和费用不变；新池30项成本/缓冲验证已完成，10项B0逐日精确复现。
后续按用户授权逐组采纳缓冲；参数、依据及当前表现见策略候选库。旧实验保留原B0基线，
不得将其净值图误认为当前混合缓冲配置的净值图；不自动批准实盘。
有效库保留各批次资格快照；局部复核不是全库重新认证。当前进度、历史边界和本地报告入口见
[策略候选库](docs/策略候选库.md)、[有效因子库](docs/有效因子库.md)及[产物索引](runs/README.md)。

## 日常入口

| 目的 | 入口 | 权威配置/产物 |
|---|---|---|
| 因子检验、入库、库内搜索 | run_factor_workflow.py | config/default.yaml、factor_library/library.json |
| 默认组合、备选比较、配置校验 | run_portfolio_workflow.py | config/strategy_library.yaml |
| 目标权重发布检查 | main.py close | config/target_publication.yaml |
| 独立权重、资金折算与交易计划 | run_trading_workflow.py（显式分步；可选auto每日准备） | config/trading.yaml；发布和执行默认关闭 |

因子入口默认要求明确填写新增批次，不会意外启动全量检验。
组合入口无参数运行默认首选策略；RUN_AND_COMPARE空ID比较默认中性12组，跨组比较须显式选择。
交易模块不随研究或回测自动启动，接口及已验证边界见[交易模块文档](docs/交易模块设计与验收.md)。

## 流程

因子来源并行：`intraday`人工创造、GP挖掘、SPEC枚举。默认流程只加载内置因子，
不主动加载GP/SPEC候选目录或启动挖掘；后二者须显式运行，产出候选后进入下述同一流程。
通过检验不等于自动入库或加入策略；改写并入`intraday.py`还需要单独授权。

```text
认证数据 → 因子计算 → 正式统计检验 → 有效因子库
                                       ↓
                        库内搜索/相关性诊断 → 策略候选目录
                                               ↓
                             同口径组合验证 → 人工批准 → 发布门
```

单因子准入使用近期窗口，组合搜索可使用更长的截止日前历史。
冻结组合回测可延长至最新完整交易日，截止日后观察不得反向用于选择或调参。
输入bar频率强校验；日频输出的best_period是预测期证据，不决定持仓期或自动分袖。
历史观察组合中未通过当前准入的成员，不会自动进入有效库。

## 环境与数据

Python 3.10+，依赖只维护在requirements.txt。本机解释器：
`E:\Python\Pythonvenv\Scripts\python.exe`。

```powershell
$PY = 'E:\Python\Pythonvenv\Scripts\python.exe'
& $PY -m pip check
& $PY -B -m pytest -q -p no:cacheprovider
& $PY -X utf8 -B main.py data-health --config config/default.yaml --strict
```

Parquet是权威发布/恢复层，认证DuckDB是默认运行源。长表热路径用Polars，
保留既有矩阵接口；已验证数值内核使用单线程Rust。工程加速不得改变公式、日期、选约或成本。

本机路径只放在Git忽略的config/local.yaml或MF_PARQUET_ROOT、MF_DUCKDB_PATH环境变量。
日常required_release_id/MF_DATA_RELEASE_ID留空，跟随唯一current＋certified发布；
复现冻结研究才固定release，不匹配失败关闭。

## 文档与目录

- [操作指南](框架工作流程与使用方法.md)：分支与日期边界。
- [文档索引](docs/README.md)：准入、有效库、策略、计算、参考及历史审计。
- [维护说明](MAINTENANCE.md)：测试、Rust构建、缓存与版本管理。
- [产物保留规则](runs/README.md)：本地证据与清理范围。
- [GP插件接入](factor_mining/FRAMEWORK_INTEGRATION.md)：挖掘不替代正式准入。

源码保留data/factors/research/workflows/strategies/backtest等既有职责模块；
配置在config/，当前有效库在factor_library/，运行证据在runs/<流程>/<run_id>/。
不在根目录放一次性实验脚本、重复参数文件或散落结果。
既有`comparison_research/`与`exa-results/`为本地冻结研究存档，保留原路径、不入Git；
方法论文字归入docs，运行数据及历史源码快照不进入默认测试发现范围。
