# 期货多因子研究框架

这是一个以本地数据研究为优先、默认不交易的期货因子研究与组合框架。当前状态
（2026-07-29）仍为 `NO_TRADE`：`config/default.yaml` 没有获批因子，
`config/trading.yaml` 也保持关闭。挖掘结果只会进入候选池，不会自动进入组合或交易。

## 架构

```text
本地 Parquet / 可选 MySQL、DolphinDB、AkShare
  -> DataManager / FrequencyDataProvider
  -> 内置 Factor + SPEC + user Factor + mined snapshot bridge
  -> IC/HAC/分层/稳健性研究
  -> 冻结 walk-forward / locked OOS
  -> 成本、风险、优化与组合回测
  -> 人工批准部署包
  -> close 决策门（缺少批准时固定 NO_TRADE）

独立 factor_mining 插件
  -> 特征引擎 -> GP 搜索 -> 诊断 -> SQLite 候选目录
  -> SHA-256 不可变 JSON 快照 -> 主框架 Factor registry
```

审查时不加载 mined 快照的动态注册表共有 3,490 个因子。这个数字是发现范围，
不是已通过检验或可交易的数量。注册表由 `factors.library` 导入触发；主工作流会负责
导入，单独使用 Python API 时应显式 `import factors.library`。

## 环境

要求 Python 3.10+。手动安装按用途选择：

```powershell
python -m pip install -r requirements-minimal.txt  # 本地 Parquet、挖掘、研究、回测
python -m pip install -r requirements.txt          # 再加入全部外部数据源/研究扩展
python -m pip install -r requirements-dev.txt      # 最小运行时 + 测试工具
```

`python` 必须指向安装了项目依赖的解释器。Windows Store 的占位 `python.exe` 不能
运行本项目，可先用 `Get-Command python` 和 `python -m pip check` 核对。

## 文档导航

- **[框架工作流程与使用方法.md](框架工作流程与使用方法.md)** — 标准工作流程（因子层可变/组合层固化）+ 快速上手（主入口）
- **docs/多因子框架研究手册.md** — 总体研究方法与四层架构
- **docs/因子检验与准入流程.md** — 因子检验与准入流程（v2 策略）
- **docs/因子创造方法论与完整参考.md** — 日内因子创造方法论与 170 因子参考
- **docs/策略基准记录.md** — 所有基准方案（B1/B2/B3）
- **docs/周期一致性与多频率共存设计指南.md** — 周期/频率设计原则
- **docs/扩展窗口历史方法与因子集搜索.md** — 当前有效因子库下的方法与因子集滚动搜索
- **MAINTENANCE.md** — 维护、性能与版本控制

## 常用入口

```powershell
$PY = '.\.venv\Scripts\python.exe'

# 本地数据健康与研究
& $PY -X utf8 -B main.py data-health --config config/parquet_research.yaml --strict
& $PY -X utf8 -B main.py research --help
& $PY -X utf8 -B main.py adaptivity --help

# 因子挖掘（合成冒烟不会写候选库）
& $PY -X utf8 -B main.py mining dev-smoke `
  --periods 600 --symbols 12 --population 80 --generations 4

# 显式启用 GP Accelerator v2-lite；默认仍为 off
& $PY -X utf8 -B main.py mining dev-smoke `
  --periods 600 --symbols 12 --population 80 --generations 4 `
  --accelerator-mode v2-lite --accelerator-chunk-size 50 --jobs 4 `
  --use-fast-rolling

# 冻结验证、回测与关闭决策
& $PY -X utf8 -B main.py walkforward --help
& $PY -X utf8 -B main.py summarize --help
& $PY -X utf8 -B main.py backtest --help
& $PY -X utf8 -B main.py multi --help
& $PY -X utf8 -B main.py close --as-of YYYY-MM-DD

# 隔离实验：扩展窗口搜索因子合成、Top/Bottom、品种权重和因子集
& $PY -X utf8 -B -m workflows.experiments.historical_portfolio_search
```

`data-health --strict`会检查四个Parquet频率的最新发布分区；郑商所三位代码、同键重复
或冲突别名会使检查失败，但不会把全量扫描加入回测热路径。

挖掘、冻结候选、挂载快照和正式筛选的最短流程见
[`factor_mining/FRAMEWORK_INTEGRATION.md`](factor_mining/FRAMEWORK_INTEGRATION.md)。

## 配置边界

- `config/default.yaml`：完整研究/回测基线，`factors: []`，当前数据源配置为 MySQL。
- `config/parquet_research.yaml`：继承默认值并切换到本地 Parquet；本地研究优先使用它。
- `config/local.yaml`：仅保存本机路径或凭据，已被 Git 忽略。
- `config/trading.yaml`：独立交易批准门，不能传给 `load_config()` 当作研究配置。
- 除成本模型的自定义参数外，未知配置键会直接报错，避免拼写错误被静默忽略。

本地 Parquet 根目录通过 `MF_PARQUET_ROOT` 或 `config/local.yaml` 提供。因子挖掘适配器
明确传入 `mysql_config=None`，不会因本地缺字段自动查询阿里云。

配置为 MySQL/RDS 的 5 分钟读取路径会先检查本地
`ths_data_5minute.db` 镜像；可用 `MF_THS_5MINUTE_DB` 或
`local_ths_5minute_db` 覆盖位置。本地文件不存在或只读查询失败时才沿用原有 RDS
endpoint failover，其他表和其他数据源不受影响。

## 研究治理

正式查看真实历史上的 IC、HAC t 值、收益或最优周期时，必须冻结公式、频率、日期、
目标和假设数，并为每次研究使用新的只写一次输出目录。仓库内的 `main.py research`
会记录检验配置、完整假设数和逐因子结果；编写表达式、合成数据调试和单元测试不要求
执行完整流程。

GP 搜索期的 IC/IR、分层和成本后收益只是优化适应度；换手仅作诊断，二者都不是正式
入围证据。SQLite 只管理候选与血缘；主框架只接收带哈希的 JSON 快照。候选即使通过筛选，也必须继续
经过相关性、容量、成本、风险和新 OOS 审核。

成本口径分两阶段：筛选期按总暴露摊销年化 0.02%；因子通过后的研究回测再加入年化
0.105% 移仓成本。换手率和换手次数不产生费用，也不作为因子准入门槛。当前状态见
[`最新因子与组合表现说明.md`](最新因子与组合表现说明.md)，正式门槛、观察期和接入顺序以
[`docs/因子检验与准入流程.md`](docs/因子检验与准入流程.md) 的 v2 策略为准。

本次政策升级前生成的本地结果和日期化报告已经清除；它们不得再作为当前结论引用。
新的正式研究必须使用新目录重新生成完整 bundle。

## 验证与保留

```powershell
$PY = '.\.venv\Scripts\python.exe'
& $PY -B -m pytest -q -p no:cacheprovider
& $PY -B -m compileall -q alpha backtest core data factor_mining factors `
  optimization pipeline processing research risk signals strategies testing workflows tests main.py
```

- 保留 `cache/` 中的本地行情缓存，除非明确执行缓存重建。
- 永久保留 `runs/factor_research/holdout_ledger.jsonl`，防止已消费 OOS 被重新标成未见样本。
- `runs/` 中的普通结果可在政策失效或结论被替代后清理；重要证据应先外部只读归档。
- `_work/`、`.pytest_cache/`、`__pycache__/` 是可再生内容，不进入版本控制。
- 实验工作流位于 `workflows/experiments/`，不属于正式命令入口。

总体研究方法见 [`多因子框架研究手册.md`](多因子框架研究手册.md)；维护、性能基线和
版本控制建议见 [`MAINTENANCE.md`](MAINTENANCE.md)。
