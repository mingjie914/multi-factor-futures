# 期货多因子研究框架

这是一个以本地数据研究为优先、默认不交易的期货因子研究与组合框架。当前状态
（2026-08-22）仍为 `NO_TARGETS`：`config/default.yaml` 的10f只是固定观察基线，
`config/target_publication.yaml` 保持关闭且没有获批部署包。挖掘结果只会进入候选池，
不会自动进入组合或发布目标权重。
当前固定观察基线为10f＋60日ICIR＋Top10/Bottom10＋cap3＋分侧ERC；它只用于
`strategies/combined.py`的研究比较，不代表交易获批或已经具备生产替代资格。

> **2026-08-22状态**：Parquet权威数据与认证DuckDB镜像均通过严格健康检查；本机运行
> 已切换为DuckDB＋Polars 1.43.2结果桥接，仓库默认仍为Parquet＋Pandas以保留可移植回退。
> 统一38品种、164交易日、588因子的新全池画像为588/588无错误，纯因子墙钟由旧画像约
> 1,517秒降至约1,111秒；本轮只改共享数值核和重复日内整理，不改策略公式或日期语义。
> 2016-03-31至2026-08-20的冻结历史运行提交599个历史注册类，其中11个不可估计定义已确认
> 为退化/市场标量并从源码清理，剩余588个均可估计。层级FDR得到20个观察发现，按
> `|corr|>=0.5`去重为13簇；该H20只作数据更新前的长期对照，尚未按冻结窗口完成WF/OOS
> 重检，生产批准数为0，目标权重发布门继续关闭。

## 架构

```text
已发布的本地 Parquet（权威、审计、恢复）
  -> 认证 DuckDB 镜像（本机运行源；仓库默认可回退）
  -> Polars 结果传输（公共边界仍为 Pandas DataFrame）
  -> DataManager / FrequencyDataProvider
  -> 内置 Factor + SPEC + user Factor + mined snapshot bridge
  -> IC/HAC/分层/稳健性研究
  -> 冻结 walk-forward / locked OOS
  -> 成本、风险、优化与组合回测
  -> 人工批准部署包
  -> close 目标权重发布门（缺少批准时固定 NO_TARGETS）

独立 factor_mining 插件
  -> 默认复用 config/default.yaml 与当前 DataManager 运行源
  -> 特征引擎 -> GP 搜索 -> 诊断 -> SQLite 候选目录
  -> SHA-256 不可变 JSON 快照 -> 主框架 Factor registry
```

审查时不加载 mined 快照的动态注册表共有 4,049 个因子（2026-08-21 实测）。这个数字是发现范围，
不是已通过检验或可交易的数量。注册表由 `factors.library` 导入触发；主工作流会负责
导入，单独使用 Python API 时应显式 `import factors.library`。

## 环境

要求 Python 3.10+。项目只保留一份依赖清单：

```powershell
python -m pip install -r requirements.txt
```

`python` 必须指向安装了项目依赖的解释器。Windows Store 的占位 `python.exe` 不能
运行本项目，可先用 `Get-Command python` 和 `python -m pip check` 核对。

## 文档导航

- **[框架工作流程与使用方法.md](框架工作流程与使用方法.md)** — 标准工作流程（因子层可变/组合层固化）+ 快速上手（主入口）
- **[docs/多因子框架研究手册.md](docs/多因子框架研究手册.md)** — 总体研究方法与四层架构
- **docs/因子检验与准入流程.md** — 因子检验与准入流程（v2 策略）
- **docs/有效因子库.md** — H20长期统计发现、13个去重观察候选与固定10f口径
- **[docs/因子创造方法论与完整参考.md](docs/因子创造方法论与完整参考.md)** — 日内因子方法论与首批 170 个历史编号参考
- **docs/策略基准记录.md** — 主分支 A 与多品种变种 B* 的基准登记
- **docs/周期一致性与多频率共存设计指南.md** — 周期/频率设计原则
- **docs/扩展窗口历史方法与因子集搜索.md** — 冻结候选清单下的方法与因子集滚动搜索
- **MAINTENANCE.md** — 维护、性能与版本控制
- **docs/性能评估与优化报告.md**、**docs/因子监控与归因仪表盘_设计文档.md** — 性能基线与观察工具
- **[docs/数据与计算加速迁移Handoff.md](docs/数据与计算加速迁移Handoff.md)** — DuckDB、Polars与选择性Rust分阶段实施边界
- **docs/郑商所合约代码修复_20260813.md** — 历史数据事故及失败关闭边界，仅作审计

## 常用入口

```powershell
$PY = 'E:\Python\Pythonvenv\Scripts\python.exe'

# 本地数据健康与研究
& $PY -X utf8 -B main.py data-health --config config/default.yaml --strict
& $PY -X utf8 -B main.py research --help
& $PY -X utf8 -B main.py adaptivity --help

# intraday.py 的588个日频输出发现因子
& $PY -X utf8 -B main.py research --config config/default.yaml `
  --all --multi-period --frequency daily `
  --module-prefix factors.library.intraday --run-id <study_id>

# 因子挖掘（合成冒烟不会写候选库）
& $PY -X utf8 -B main.py mining dev-smoke `
  --periods 600 --symbols 12 --population 80 --generations 4

# 显式启用 GP Accelerator v2-lite；默认仍为 off
& $PY -X utf8 -B main.py mining dev-smoke `
  --periods 600 --symbols 12 --population 80 --generations 4 `
  --accelerator-mode v2-lite --accelerator-chunk-size 50 --jobs 4 `
  --use-fast-rolling

# 冻结验证、回测与目标权重发布检查
& $PY -X utf8 -B main.py walkforward --help
& $PY -X utf8 -B main.py summarize --help
& $PY -X utf8 -B main.py backtest --help
& $PY -X utf8 -B main.py multi --help
& $PY -X utf8 -B main.py close --as-of YYYY-MM-DD --output runs/close_check.json

# 隔离实验：扩展窗口搜索因子合成、Top/Bottom、品种权重和因子集
& $PY -X utf8 -B -m workflows.experiments.historical_portfolio_search `
  --factor-manifest runs/factor_research/<p0_run>/ic_by_window_period.json `
  --output runs/historical_portfolio_search/<run_id>
```

`data-health --strict`检查所选运行源、四个Parquet权威频率、日线全历史和六张席位表；
郑商所三位代码、同键重复、冲突别名或席位自然键异常会使检查失败，但不会把全量扫描
加入回测热路径。
正式研究和回测不会在行情分区损坏或交易日历为空时静默跳过数据或改用普通工作日，
而是在进入因子与组合计算前失败关闭。动态上市过滤由当前运行源的日线具体合约首个已发布
交易日驱动；Parquet与认证DuckDB返回同一日期契约。

挖掘、冻结候选、挂载快照和正式筛选的最短流程见
[`factor_mining/FRAMEWORK_INTEGRATION.md`](factor_mining/FRAMEWORK_INTEGRATION.md)。

## 配置边界

- `config/default.yaml`：研究、筛选、回测、监控与主策略共用的唯一框架配置，包含固定
  38品种、10f观察基线和统一处理语义；默认只读本地Parquet，`duckdb_futures`是认证镜像。
- `config/local.yaml`：只保存本机数据路径、认证release与数据运行时选择，已被Git忽略；
  加载器禁止它覆盖品种、因子、处理或回测语义。
- `config/target_publication.yaml`：独立目标权重发布门，不能传给 `load_config()`当作研究配置。
- 除成本模型的自定义参数外，未知配置键会直接报错，避免拼写错误被静默忽略。

本地 Parquet 根目录通过 `MF_PARQUET_ROOT` 提供；切换认证库时另设
`MF_DATA_SOURCE=duckdb_futures`、`MF_DUCKDB_PATH`、当前`MF_DATA_RELEASE_ID`和可选的
`MF_DUCKDB_RESULT_BACKEND=pandas|polars|shadow`。仓库默认后端为`pandas`；只有通过
真实A/B的本机部署才设为`polars`。夜间DuckDB发布新release后，必须先验证成功，再更新
绑定的release ID并重启；旧ID会失败关闭，不能自动漂移到未经确认的数据。
框架不包含远程行情查询、核对或回填旁路；数据修复与发布属于独立数据工程。

## 研究治理

正式查看真实历史上的 IC、HAC t 值、收益或最优周期时，必须冻结公式、频率、日期、
目标和假设数，并为每次研究使用新的只写一次输出目录。仓库内的 `main.py research`
会记录检验配置、完整假设数和逐因子结果；编写表达式、合成数据调试和单元测试不要求
执行完整流程。

GP 搜索期的 IC/IR、分层和成本后收益只是优化适应度；换手仅作诊断，二者都不是正式
入围证据。SQLite 只管理候选与血缘；主框架只接收带哈希的 JSON 快照。候选即使通过筛选，也必须继续
经过相关性、容量、成本、风险、冻结OOS和`simulated_live`审核。

成本口径分两阶段：筛选期将年化半换手还原为完整成交名义，再乘每单位 0.02% 的成本率；
因子通过后的研究回测按实际生效日的 `executed_traded_notional` 计提同一成本，并按总暴露
摊销年化 0.105% 的保守移仓预算。`decision_turnover` 仅是决策诊断，不是成本基数。当前观察基线定义见
[`docs/生产方案技术说明.md`](docs/生产方案技术说明.md)，正式门槛、观察期和接入顺序以
[`docs/因子检验与准入流程.md`](docs/因子检验与准入流程.md) 的 v2 策略为准。
组合账本仍逐日记录具体合约的开仓、平仓和非调仓日换月；这些腿进入换手诊断和自定义
逐笔成本模型，但默认年化移仓费不会因此重复计费。

本次政策升级前生成的本地结果和日期化报告已经清除；它们不得再作为当前结论引用。
冻结历史对照位于`runs/factor_research/20260820_intraday599_rebuild/`；目录名保留最初
提交规模。其研究契约、代码/配置/数据哈希用于更新后同区间重放比较，不代表当前数据
更新后的正式结论。
历史研究阶段由`research/validation.py::period_protocol_snapshot()`单一维护：最后一折OOS固定结束于
`2026-05-14`，`2026-05-15`起至本地最新数据固定标为`simulated_live`。运行日期和数据库
最新日期不得反向改写OOS边界；`holdout_ledger.jsonl`只记录历史样本消费事实，不是边界配置。

## 验证与保留

```powershell
$PY = 'E:\Python\Pythonvenv\Scripts\python.exe'
& $PY -B -m pytest -q -p no:cacheprovider
& $PY -B -m compileall -q alpha backtest core data factor_mining factors `
  optimization pipeline processing research risk strategies testing workflows tests main.py
```

- 保留 `cache/` 中的本地行情缓存，除非明确执行缓存重建。
- 永久保留 `runs/factor_research/holdout_ledger.jsonl`，防止已消费 OOS 被重新标成未见样本。
- `runs/` 中的普通结果可在政策失效或结论被替代后清理；重要证据应先外部只读归档。
- `_work/`、`.pytest_cache/`、`__pycache__/` 是可再生内容，不进入版本控制。
- 实验工作流位于 `workflows/experiments/`，不属于正式命令入口。

总体研究方法见 [`docs/多因子框架研究手册.md`](docs/多因子框架研究手册.md)；维护、性能基线和
版本控制建议见 [`MAINTENANCE.md`](MAINTENANCE.md)。
