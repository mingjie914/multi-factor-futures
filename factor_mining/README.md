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

默认预测目标是未来 15 个 bar，而不是未来 1/5/10 个交易日。可显式使用 H=1，
此时预测的是未来 1 根 1 分钟 bar，而不是“日频因子”：

- `1`：下一根分钟 bar；
- `5,15,30,60`：日内短周期；
- `120,240`：半日到约一个交易日；
- 更长目标必须显式指定。

目标使用 `close[t+entry_delay+horizon] / close[t+entry_delay] - 1`，候选信号还会
额外执行至少 1 bar 的决策滞后，避免当前收盘数据被当前决策消费。

挖掘端和正式框架都按每个合约自身的有效 bar 移位，不会因为某个合约在公共分钟索引上
缺一根报价，就把该合约的 H=15 错算成不同经济长度。正式研究的 IC/OLS 也是逐分钟
时间截面计算；只有自然年稳健性、日换手聚合和治理记分卡使用日/年汇总。因此不能把
当前零产出归因为“1 分钟信号被先聚合成日频 IC”。

预筛默认额外输出两类不参与正式准入的诊断：

- `predictive_ic_decay`：同一已滞后信号分别预测未来 1/3/5/10/20/40 根 bar 收益；
- `signal_rank_persistence`：横截面信号排名在上述 bar lag 的平均相关性，并报告首次跌破
  0.5 的半衰期。未在最大 lag 内跌破时明确标记为右删失，而不是伪造半衰期。

它们回答的是“预测作用在哪个分钟 horizon”与“排名能保持多久”。旧 `ICTest.ic_decay`
计算的是 IC 时间序列本身的自相关，不等同于这两项诊断。

预筛目前使用候选快照声明的静态品种全集，结果会明确写入
`diagnostic_universe_policy=static_declared_universe`。正式 P0 仍使用主框架的滞后
流动性动态品种池；静态预筛曲线只能定位 horizon，不能替代动态池下的正式 IC。

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
`ts_rank` 使用 pandas 原生滚动排名，`decay_linear` 使用按列卷积；两者不再逐窗口
回调 Python。慢统计族仍默认不进入快速 profile，应通过 `--operators` 显式分配搜索预算。

“全面覆盖”指某次 campaign 明确启用上述特征族和算子族，并给每个 profile 分配搜索
预算；它不可能也不应解释为穷举无限深度的全部表达式组合。每次运行的候选数、随机种子、
profile、特征配置和目标 horizon 都必须冻结，未被该次表达式实际引用的 terminal 不应
被宣称为已经得到充分搜索。

`--gp-windows` 冻结 GP 时序算子的可用窗口；`--coverage-penalty` 与
`--segment-floor-weight` 可在搜索适应度中惩罚稀疏覆盖和最差时间段，但这些仍是搜索
排序项，不是正式 HAC/FDR 证据。run 元数据会保存人口、代数、窗口、算子和全部适应度
参数；预筛可用 `--candidate-run-ids` 一次选择该 run 的全部候选。

GP fitness 不再只奖励 IC/IR。默认 `economic_fitness_weight=0.50`，使用横截面 rank
权重组合在声明调仓节奏下的成本后收益，经同期目标收益横截面离散度归一化后参与搜索；
原换手与复杂度惩罚继续保留。`--rebalance-every-bars 0` 默认采用目标 horizon，显式非零
值则冻结为独立执行节奏。成本使用 `TargetSpec.cost_bps`，因此生产 campaign 必须声明
现实的单边成本，并按 `half_turnover × 2 × one_way_cost` 扣减；不能用 0 成本搜索后再
宣称具有经济意义。该 fitness 仍是训练期搜索
目标，不替代正式成本覆盖、FDR 或 OOS。

## 期货特异数据覆盖

当前 GP 默认并未把 WorldQuant 101 或 GTJA 191 公式当作 terminal；它们位于独立的兼容
公式库，不能解释本轮 GP 候选零通过。现有 terminal 已覆盖 OHLCVA、持仓量及其变化，
并可选加入合约曲线的持仓广度/集中度，但以下真实期货数据仍未形成可靠的本地 point-in-time
面板：近远月价格与到期日、现货基差、会员多空持仓、库存/仓单、逐笔方向和订单簿。

因此当前 `curve_*` 不得被称为展期收益率或期限斜率，分钟 OHLCV 代理也不得被称为真实
订单簿不平衡/VPIN。下一步应先补齐这些本地数据契约和发布时间滞后，再作为 terminal
加入；库存/仓单需要按预声明板块适用性路由。IPCA、Conditional Autoencoder、XGBoost
和 LLM 生成表达式属于后续独立搜索后端，必须复用同一候选快照、成本和 OOS 协议。

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
中性化可通过 `GPSearch(..., group_labels=...)` 或 CLI 的 `--sector-neutralization`
开启。CLI 使用主框架唯一的静态期货板块表，并把标签冻结进候选；正式研究仍会执行
主框架配置声明的板块中性化。

## 预筛与正式检验

`screen` 会一次处理显式给定的全部候选，不设入围数量。硬门槛只处理重复公式、依赖、
覆盖率、横截面变化和计算错误；同样本 IC、相关性、成本后收益和分段稳定性只作诊断。
输出包括完整 CSV、相关性矩阵和不可变候选快照。

正式研究必须按候选原始 `target.horizon_bars` 分组；H=1 挖掘因子只检验 1 bar，不能
事后在 5/15/30 bar 中挑最优。若启用默认动态流动性品种池，`factor-start` 必须早于
评价起点至少 `min_listing_days` 个交易日，并覆盖流动性 lookback。研究批次若得到
0 个有效因子×周期观测会直接失败，不再产生“零样本但完成”的结论。

2026-07-27 的 H=1/5/15/30/60/240 当前口径正式复核结果见
[`../docs/factor_mining_validation_20260727.md`](../docs/factor_mining_validation_20260727.md)。
该文件是旧 Bonferroni 口径的历史基线；当前正式准入使用
[`../docs/factor_validation_pipeline.md`](../docs/factor_validation_pipeline.md) 所述 v2
层级 FDR、自然年、成本和观察期流程，不能直接比较“通过数”而忽略方法版本。

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
