# Factor Mining Plugin

`factor_mining` 是与主框架解耦的本地因子挖掘插件。它负责生成和管理候选；主框架仅在
`main.py mining` 或显式 `--mined-snapshot` 时调用它，不修改 `ALL_SPECS`、默认因子
配置或现有因子计算逻辑。

主框架接入只需阅读 [FRAMEWORK_INTEGRATION.md](FRAMEWORK_INTEGRATION.md)。

## 边界

- 数据：只读已发布的本地1分钟Parquet，不包含远程数据旁路。
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

默认 `screen` 预筛使用候选快照声明的静态品种全集，结果会明确写入
`diagnostic_universe_policy=static_declared_universe`。正式 P0 仍使用主框架的滞后
流动性动态品种池；静态预筛曲线只能定位 horizon，不能替代动态池下的正式 IC。
动态池下的正式检验统一通过 `main.py research` 执行，不再保留一次性对齐脚本。

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
复杂度惩罚继续保留，换手率仅作诊断且不影响 fitness。`--rebalance-every-bars 0` 默认采用目标 horizon，显式非零
值则冻结为独立执行节奏。成本使用 `TargetSpec.cost_bps` 声明固定年化费率，当前默认
为年化 2 bp，并按目标持有 bar 数摊销；换手率仅作诊断，不按换手率或换手次数扣费。
因子筛选成功后的完整研究回测会另外加入年化 10.5 bp 移仓成本。该 fitness 仍是训练期搜索
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

政策升级前的日期化复核报告和本地结果已经清除。当前正式准入只使用
[`../docs/因子检验与准入流程.md`](../docs/因子检验与准入流程.md) 所述流程；
新结果必须写入新目录，不能与旧方法的通过数直接比较。

## 开发冒烟

```powershell
$PY = 'E:\Python\Pythonvenv\Scripts\python.exe'
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

## GP Accelerator（可选）

GP 加速层默认关闭，只作用于 `mining mine` / `mining dev-smoke` 的
population evaluation，不替换 P0、单因子研究、自有因子库或
BacktestEngine。

```powershell
& $PY -B main.py mining dev-smoke `
  --periods 600 --symbols 12 --population 80 --generations 4 `
  --use-accelerator --use-fast-rolling
```

- `--accelerator-mode off`：旧路径，默认值。
- `--accelerator-mode context`：只使用只读 terminal snapshot。
- `--accelerator-mode batch`：增加分块批量 MAD、中性化、rank IC 和
  fitness。
- `--accelerator-mode dag` / `--use-accelerator`：增加 population DAG。
- `--accelerator-mode chunk`：保留 v1 batch fitness，只把 expression
  evaluation 按 factor chunk 交给现有 ThreadPool，用于隔离调度收益。
- `--accelerator-mode v2-lite`：在 chunk 路径上增加在线 fitness 统计并降低
  block 峰值内存；默认仍不启用。
- `--accelerator-chunk-size`：每个 expression task 的因子数，默认 50；
  worker 数继续使用 `--jobs`，应以本机 benchmark 选择。
- `--use-fast-rolling`：仅在兼容探针通过时使用可选 Bottleneck；
  不可用或不等价时自动回退 Pandas。Bottleneck 不是硬依赖。

snapshot 将 FeatureEngine 已生成的完整 terminal vocabulary 以只读
`.npy` 保存。物理布局为 `(F,T,N)`，以便每个 `(T,N)` terminal 是连续
零拷贝 view；运行时同时提供逻辑 `(T,N,F)` view。metadata 绑定
FeatureConfig、TargetSpec、taxonomy、source fingerprint 和所有 snapshot
artifact 的 SHA-256。Windows 下继续使用现有 ThreadPool，不传递大型
DataFrame/ndarray，也不假设 `fork`。v2-lite worker 只计算共享只读 view
上的表达式；主线程按完成的 factor chunk 执行 MAD、中性化、rank-IC 和
portfolio fitness，不创建整代 `(M_all,T,N)` tensor。

`scripts/benchmark_gp_accelerator.py` 固定同一批至少 100 棵 AST，比较
baseline、Accelerator v1、v1+FactorChunk 和 v2-lite，同时输出 JSON/CSV。
benchmark 会保存固定 AST、raw factor memmap、worker/chunk 调优、各 kernel
计时和 RSS，并硬断言 NaN mask、factor 容差、IC、direction、candidate
集合和排序。运行时必须显式传入 `--output-dir`，脚本不会自动创建时间戳目录。

NumPy stable-argsort rank 曾通过 ties/NaN 等价探针，但在完整 12,000×47
benchmark 上慢于现有 SciPy `rankdata`，因此未进入生产路径；rank 仍使用 v1
已验证的 SciPy/Pandas fallback。

v2-lite 使用算子能力白名单。`ts_ema`、`ts_corr`、`ts_cov`、横截面有状态/
边界敏感算子及未知算子在计算前直接进入 legacy evaluator；不会在完整运行后因误差
再重复计算。

### 2026-07-29 固定 100 AST 基准

环境为 Windows、12,000×47、`block_T=2500`，每条正式路径重复三次并取中位数：

| 路径 | 中位耗时 | 相对 baseline | 峰值 RSS |
| --- | ---: | ---: | ---: |
| Baseline | 10.415s | 1.00× | 0.74GB |
| Accelerator v1 | 7.382s | 1.41× | 2.01GB |
| v1 + FactorChunk（75×2） | 6.964s | 1.50× | 1.84GB |
| v2-lite（75×2） | 6.974s | 1.49× | 1.84GB |

四条路径的 NaN mask、factor value、IC、direction、candidate ID/集合和 fitness
排序全部通过硬断言，fallback 为 0。Factor Chunk 提供了确定性耗时收益；在线
accumulator 的主要收益是避免整代状态常驻并将 RSS 控制在 2GB 内，不能把它描述成
额外数量级加速。历史 benchmark 产物已经清理；如需复核，使用
`python -m scripts.benchmark_gp_accelerator --output-dir <目录>` 显式重建。

同一形状扩展到 220 population 后，Baseline、v1、FactorChunk、v2-lite 分别为
23.313s、16.331s、15.540s、15.493s；最佳调度是 `chunk_size=50`、
`workers=4`，v2-lite 峰值 RSS 为 1.61GB。目标 7–9 秒未达成，原因是表达式阶段
已经降至约 0.98s，剩余时间主要是主线程 MAD、neutralization 和 rank-IC。继续增加
expression worker 不能解决这部分瓶颈。历史 220 规模报告也已清理；该扩展测试复用
同一生成规则并检查 candidate 结果，但为控制耗时没有重复 raw factor value 断言，
raw value 的正式硬验收仍以固定 100 AST 报告为准。
