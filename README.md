# 期货多因子研究框架

这是一个以本地数据研究为优先、默认不交易的期货因子研究与组合框架。当前状态
（2026-07-27）仍为 `NO_TRADE`：`config/default.yaml` 没有获批因子，
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

审查时不加载 mined 快照的动态注册表共有 3,487 个因子。这个数字是发现范围，
不是已通过检验或可交易的数量。注册表由 `factors.library` 导入触发；主工作流会负责
导入，单独使用 Python API 时应显式 `import factors.library`。

## 环境

要求 Python 3.10+。Windows 可直接运行：

```powershell
.\setup.bat
$PY = '.\.venv\Scripts\python.exe'
& $PY main.py --help
```

手动安装时按用途选择：

```powershell
python -m pip install -r requirements-minimal.txt  # 本地 Parquet、挖掘、研究、回测
python -m pip install -r requirements.txt          # 再加入全部外部数据源/研究扩展
python -m pip install -r requirements-dev.txt      # 最小运行时 + 测试工具
```

`python` 必须指向安装了项目依赖的解释器。Windows Store 的占位 `python.exe` 不能
运行本项目，可先用 `Get-Command python` 和 `python -m pip check` 核对。

## 常用入口

```powershell
$PY = '.\.venv\Scripts\python.exe'

# 本地数据健康与研究
& $PY -X utf8 -B main.py data-health --config config/parquet_research.yaml
& $PY -X utf8 -B main.py research --help
& $PY -X utf8 -B main.py adaptivity --help

# 因子挖掘（合成冒烟不会写候选库）
& $PY -X utf8 -B main.py mining dev-smoke `
  --periods 600 --symbols 12 --population 80 --generations 4

# 冻结验证、回测与关闭决策
& $PY -X utf8 -B main.py walkforward --help
& $PY -X utf8 -B main.py summarize --help
& $PY -X utf8 -B main.py backtest --help
& $PY -X utf8 -B main.py multi --help
& $PY -X utf8 -B main.py close --as-of 2026-07-27
```

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

## 研究治理

正式查看真实历史上的 IC、HAC t 值、收益或最优周期时，必须冻结公式、频率、日期、
目标和假设数，并按 `research-futures-factors` 协议留下可审计产物。编写表达式、合成数据
调试和单元测试不要求执行完整流程。

GP 搜索期的 IC/IR、分层、换手和成本只是优化适应度，不是正式入围证据。SQLite 只
管理候选与血缘；主框架只接收带哈希的 JSON 快照。候选即使通过历史筛选，也必须继续
经过相关性、容量、成本、风险和新 OOS 审核。

历史 3,111 因子全量研究及其零 Bonferroni 通过结果仍是有效的历史证据，但不是当前
注册表清单。详见 [`最新因子与组合表现说明.md`](最新因子与组合表现说明.md) 顶部的
历史快照说明和 [`多因子框架研究手册.md`](多因子框架研究手册.md)。

## 验证与保留

```powershell
$PY = '.\.venv\Scripts\python.exe'
& $PY -B -m pytest -q -p no:cacheprovider
& $PY -B -m compileall -q alpha backtest core data factor_mining factors `
  optimization pipeline processing research risk signals strategies testing workflows tests main.py
```

- 保留 `cache/` 中的本地行情缓存，除非明确执行缓存重建。
- 保留 `runs/locked_oos/` 和 `runs/factor_research/` 中的证据及 holdout ledger。
- `_work/`、`.pytest_cache/`、`__pycache__/` 是可再生内容，不进入版本控制。
- 实验工作流位于 `workflows/experiments/`，不属于正式命令入口。

维护、性能基线和版本控制建议见 [`MAINTENANCE.md`](MAINTENANCE.md)。
