# 期货多因子研究框架

本地期货因子研究、有效库管理与组合回测框架，默认不发布交易目标。

截至2026-09-10：有效因子库154条，策略目录8组观察备选，默认 **多源稳衡**（17因子）；新增入库因子未自动加入策略。
默认组合方法为 ICIR＋Top10/Bottom10＋五大分组cap3＋分侧ERC。
“默认”指生产方法回测/观察，不等于实盘批准；目标权重发布门保持关闭（NO_TARGETS）。

## 日常入口

| 目的 | 入口 | 权威配置/产物 |
|---|---|---|
| 因子检验、入库、库内搜索 | run_factor_workflow.py | config/default.yaml、factor_library/library.json |
| 默认组合、备选比较、配置校验 | run_portfolio_workflow.py | config/strategy_library.yaml |
| 目标权重发布检查 | main.py close | config/target_publication.yaml |

因子入口默认要求明确填写新增批次，不会意外启动全量检验。
组合入口无参数运行默认首选策略；修改IDE SETTINGS显式分支即可比较8组。

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
