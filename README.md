

# 期货多因子研究框架

这是一个以本地数据研究为优先、默认不交易的期货因子研究与组合框架。当前状态（2026-07-28）仍为 `NO_TRADE`：`config/default.yaml` 没有获批因子，`config/trading.yaml` 也保持关闭。挖掘结果只会进入候选池，不会自动进入组合或交易。

## 核心特性

- **多数据源支持**：本地 Parquet 优先，可选 MySQL、DolphinDB、AkShare
- **因子研究**：内置因子、SPEC 因子、用户自定义因子与挖掘快照桥接
- **检验体系**：IC/HAC 分层回测、稳健性分析、Walk-forward 验证
- **成本模型**：筛选期年化 0.02% 摊销成本，回测期加入 0.105% 移仓成本
- **三层资产配置**：资产层、行业层、品种层风险平价优化
- **因子挖掘**：独立 GP 搜索插件，SQLite 候选管理，SHA-256 不可变快照

## 快速开始

### 环境要求

Python 3.10+，Windows 可直接运行：

```powershell
.\setup.bat
$PY = '.\.venv\Scripts\python.exe'
& $PY main.py --help
```

手动安装时按需选择：

```powershell
# 本地 Parquet、挖掘、研究、回测
python -m pip install -r requirements-minimal.txt
# 再加入全部外部数据源和研究扩展
python -m pip install -r requirements.txt
# 最小运行时 + 测试工具
python -m pip install -r requirements-dev.txt
```

### 常用命令

```powershell
$PY = '.\.venv\Scripts\python.exe'

# 数据健康与本地研究
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
& $PY -X utf8 -B main.py close --as-of YYYY-MM-DD
```

## 配置说明

| 配置文件 | 用途 |
|---------|------|
| `config/default.yaml` | 完整研究/回测基线，当前数据源为 MySQL |
| `config/parquet_research.yaml` | 继承默认值并切换到本地 Parquet |
| `config/local.yaml` | 本机路径或凭据（已被 Git 忽略） |
| `config/trading.yaml` | 独立交易批准门，不可作为研究配置 |

本地 Parquet 根目录通过 `MF_PARQUET_ROOT` 或 `config/local.yaml` 提供。

## 研究治理

正式查看真实历史上的 IC、HAC t 值、收益或最优周期时，必须冻结公式、频率、日期、目标和假设数，并为每次研究使用新的只写一次输出目录。

GP 搜索期的 IC/IR、分层和成本后收益仅作优化适应度诊断，换手率不作因子准入门槛。SQLite 只管理候选与血缘，主框架只接收带哈希的 JSON 快照。候选即使通过筛选，也必须继续经过相关性、容量、成本、风险和新 OOS 审核。

正式门槛、观察期和接入顺序以 `docs/factor_validation_pipeline.md` 的 v2 策略为准。

## 验证与清理

```powershell
$PY = '.\.venv\Scripts\python.exe'
& $PY -B -m pytest -q -p no:cacheprovider
& $PY -B -m compileall -q alpha backtest core data factor_mining factors `
  optimization pipeline processing research risk signals strategies testing workflows tests main.py
```

保留策略：
- `cache/` 中的本地行情缓存，除非明确执行缓存重建
- `runs/factor_research/holdout_ledger.jsonl` 永久保留，防止 OOS 被重新标记
- `runs/` 中的普通结果可在政策失效后清理
- `_work/`、`.pytest_cache/`、 `__pycache__/` 可再生，不进入版本控制

## 目录结构

```
├── alpha/           # 收益模型（OLS、IC 加权、分组 OLS）
├── backtest/        # 回测引擎、指标计算、研究账本
├── core/            # 配置、接口、注册表、类型定义
├── data/            # 数据源（MySQL、Parquet、AkShare、DolphinDB）
├── factors/         # 因子库、因子引擎、处理器、合成器
├── factor_mining/   # GP 搜索、验证、SQLite 存储库
├── optimization/    # 约束、成本、均值方差/风险平价优化器
├── pipeline/        # 研究与回测流程编排
├── processing/      # 缺失值填充、中性化、标准化 Winsorize
├── research/        # 研究产物、治理、统计、验证策略
├── risk/            # Barra 期货风险模型
├── signals/         # 信号生成、仓位管理、止盈止损
├── strategies/      # 防御趋势风险平价、Supertrend ATR
├── testing/         # IC、分层、回归、稳健性、换手率测试
├── workflows/       # 研究、Walk-forward、回测、多组合、诊断
├── tests/           # 单元测试与集成测试
└── docs/            # 因子检验流程、三层资产配置说明
```