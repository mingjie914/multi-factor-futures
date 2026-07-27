# Factor Mining Plugin

`factor_mining` 是与主框架解耦的本地因子挖掘插件。它负责生成和管理候选；主框架仅在
`main.py mining` 或显式 `--mined-snapshot` 时调用它，不修改 `ALL_SPECS`、默认因子
配置或现有因子计算逻辑。

主框架接入只需阅读 [FRAMEWORK_INTEGRATION.md](FRAMEWORK_INTEGRATION.md)。

## 边界

- 数据：默认只读本地 1 分钟 Parquet；适配器显式传入 `mysql_config=None`。
- 搜索：首版使用内置轻量 GP，无 `gplearn`/`DEAP` 新依赖，不使用 Python `eval`。
- 验证：挖掘期提供 rank IC/IR、分层、换手、成本和时间分段诊断；它们不是正式
  HAC 检验或 OOS 证据。
- 仓库：SQLite 只保存候选、运行参数和血缘，不参与主框架运行时计算。
- 接入：主框架只加载经过 SHA-256 校验的不可变 JSON 快照，并动态注册普通
  `Factor` 子类。
- 黑箱：模型候选的存储接口已经预留，首版桥接器只接受符号表达式。黑箱模型和
  Agent 应在特征引擎、GP、验证和仓库稳定后再接入。

## 1 分钟语义

默认特征包含当前 bar 的 OHLCV、成交额、持仓量，以及 `return_1p`、
`intrabar_return_1p`、`close_lag_1p`。`rolling_windows` 从 3 开始，因为单个样本的
滚动标准差、偏度、排名没有统计意义；这不代表遗漏 1 分钟信息。

默认预测目标是未来 15 个 bar，而不是未来 1/5/10 个交易日：

- `5,15,30,60`：日内短周期；
- `120,240`：半日到约一个交易日；
- 更长目标必须显式指定。

目标使用 `close[t+entry_delay+horizon] / close[t+entry_delay] - 1`，候选信号还会
额外执行至少 1 bar 的决策滞后，避免当前收盘数据被当前决策消费。

## 特征和算子

默认快速路径读取 `open/high/low/close/volume/amount/oi/oi_change`，并构建：

- 多周期收益、量/额/持仓变化、lag；
- MA、EMA、Bollinger、RSI、ATR、MACD；
- 实现波动率、上下半方差、偏度、峰度；
- 量价相关、成交集中度、Amihud 流动性代理。

`--include-curve` 可加入本地期限结构聚合字段，但会增加 I/O 和计算量。库存、仓单、
结算价、现货基差、会员持仓、库存/仓单、宏观和另类数据目前没有被本地 Parquet
适配器可靠暴露为 point-in-time 面板，因此首版不伪造代理值。未来将真实本地面板传给
`FeatureEngine` 并把字段加入 `FeatureConfig.raw_fields` 后，字段会自动成为 GP
terminal；若走 CLI，还需在 `LocalParquetData` 中显式映射对应数据集。

算子包括保护四则运算、平方/立方/有符号开方和对数、max/min/avg、delay/delta、
滚动统计、相关/协方差、时序和横截面 rank/zscore/demean，以及可选条件逻辑。
`ts_rank` 和 `decay_linear` 等 Python 滚动回调算子默认不进入 GP 搜索，仅按需启用。

## 性能原则

- 特征和表达式数组使用只读 `float32`。
- 特征一次构建、多候选复用；表达式按内容哈希去重。
- 子树缓存是按字节预算的 LRU，不会无限增长。
- `max_feature_memory_mb` 默认 4096 MB，超出时直接失败并提示缩短区间或减少窗口。
- 默认 `--jobs 1`。GP 的瓶颈通常是内存带宽和滚动计算；多线程可能增加内存压力，
  应先基准再调整。
- 首版不要求 GPU。符号树控制流、Pandas 滚动和中等宽度横截面通常无法覆盖 CPU/GPU
  搬运开销。未来训练 XGBoost/神经网络黑箱候选时，再通过独立搜索后端启用 GPU。

大型历史建议按训练区间分批开展独立 mining run，不要一次加载多年全市场分钟数据。
候选的跨区间稳定性应由正式研究协议验证，而不是把所有历史塞进一次搜索。

本机合成基准（5000 bar × 20 品种）中，262 个默认特征约占 100 MB；32 个体 × 2 代、
50 个唯一表达式的端到端冒烟在候选诊断向量化后约 2.19 秒（优化前约 3.65 秒）。该数字
只用于发现明显性能退化，不代表真实数据吞吐或收益表现。

波动率中性化默认开启；缺少 15/30/60-bar 实现波动率控制时搜索会拒绝运行。板块
中性化可通过 `GPSearch(..., group_labels=...)` 开启。CLI 的纯本地分钟面板目前不携带
可靠板块标签，因此不猜测板块；正式筛选阶段仍应使用主框架的板块中性化处理。

## 开发冒烟

```powershell
$PY = '.\.venv\Scripts\python.exe'
& $PY -B main.py mining dev-smoke `
  --periods 600 --symbols 12 --population 80 --generations 4
```

此命令只使用合成数据，不写 SQLite，也不触发正式研究流程。

也可继续使用独立入口 `python -m factor_mining`；`main.py mining` 是并入主框架后的
首选入口，两者调用同一实现。

## 程序接口

稳定的模块边界如下：

```python
from factor_mining.data import LocalParquetData, LocalParquetSpec
from factor_mining.features import FeatureEngine
from factor_mining.gp import GPConfig, GPSearch
from factor_mining.repository import CandidateRepository
from factor_mining.validation import PreparedTarget, ValidationConfig
```

`CandidateSpec` 是挖掘、SQLite 和桥接器之间的唯一候选契约；表达式是结构化 AST，
不执行任意代码。
