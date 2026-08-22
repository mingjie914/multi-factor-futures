# 数据与计算加速迁移实施 Handoff

> 基线日期：2026-08-22
>
> 解释器：`E:\Python\Pythonvenv\Scripts\python.exe`（Python 3.12.10）
>
> 总目标：不改变因子创建、research、回测、组合和风险工作流，以最小改动依次引入
> DuckDB、Polars 和选择性 Rust 数值核，最终降低真实因子研究的端到端耗时与峰值内存。

## 1. 硬约束

以下契约在整个迁移过程中保持不变：

- 因子仍通过 `intraday.py`、SPEC、user factor 和 GP snapshot 创建与注册。
- `Factor.compute(data, dates, universe)`、因子名称、依赖、频率和输出含义不变。
- research、walk-forward、回测、组合、风险和监控入口不增加必需参数。
- 主力选择 T-1、复权、夜盘交易日、换月、经济时期、期限结构和席位日期匹配不变。
- Polars 和 Rust 只能替换执行方式，不能重新定义公式、窗口、样本、缺失值或排序规则。
- Parquet 保持权威发布与恢复层；DuckDB 是认证的运行时镜像，不反向修改 Parquet。
- 每个阶段只切换一个变量，保留可配置回退，不把多个大型改造合并发布。
- 不依赖、安装、fork 或复制 `py-alpha-lib`；Rust 模块在本项目内原创实现。

迁移验收采用“语义等价”，不是无意义的字节等价：

- 日度时间按交易日等价，分钟时间按实际 bar 时间等价；`datetime64[us]` 与
  `datetime64[ms]` 若表达相同时间，不构成策略差异。
- 数据发布的 release、分区摘要和 Schema 摘要仍严格保留，用于完整性与恢复审计；
  但 Pandas/Polars/Rust 的中间对象不要求二进制哈希相同。
- 品种集合、日期集合、顺序、缺失位置、公式、窗口、rank、信号方向、选品、订单和持仓
  必须一致。浮点归约按算子冻结足够严格的容差，不能用一个宽松全局阈值掩盖差异。

## 2. 当前基线

### 2.1 数据层

- 权威数据：本地分月 Parquet。
- 认证运行库：
  `E:\程明杰公司内容\期货行情数据\本地表\futures_data.duckdb`。
- 当前认证 release：
  `e5b668b69d03658b716422ad953bddcbb46b154c312663078919f445068c9bfd`。
- Schema：`futures_data_v1_seat_contract_v2`。
- 表：4 张行情、6 张席位、2 张元数据。
- 当前主框架已支持 `parquet_futures` 与 `duckdb_futures`；仓库默认仍为 Parquet，
  通过环境配置切换 DuckDB。
- 正式`factor_mining mine/screen`默认复用`config/default.yaml`与DataManager；显式
  `--data-root`仅保留为Parquet审计/回退入口，不维护第二份框架配置。

### 2.2 计算层

- `config/default.yaml` 是研究、筛选、回测、监控和主策略的唯一框架配置；
  `target_publication.yaml` 只是发布审批门，不是第二套策略配置。
- `core.sectors.FRAMEWORK_UNIVERSE` 是唯一有序品种契约；当前为38品种。配置文件保留显式
  列表便于审计，但加载时必须与代码契约完全一致。
- 公共契约仍是 Pandas `DataFrame(dates × roots)`。
- 当前注册表约 4,049 个因子，其中大量由共享 SPEC/公式路径动态生成；迁移共享算子，
  不逐个重写注册类。
- 当前已存在 `factors/numerics.py` 和 GP rolling backend，应复用这些接入点，
  不另建通用 Operator 框架、DAG 或缓存体系。
- Polars 1.43.2 是冻结实施版本；Rust 工具链尚未建立。

## 3. 目标架构与职责

```text
RDS / CSV 发布流程
        ↓
Parquet（权威、审计、恢复）
        ↓ 只在成功 manifest 后发布
DuckDB（认证物理镜像）
        ↓ SQL 投影、过滤
Polars（长表读取与经准入的数据热点）
        ↓ 唯一受控边界
Pandas DataProvider 矩阵（过渡期公共契约）
        ↓
现有因子 / SPEC / GP / research / backtest
        ↓ 仅对 profile 热点
本地 Rust ndarray 数值核（逐算子准入）
```

执行方式选择：

| 工作类型 | 首选执行器 | 原因 |
|---|---|---|
| 分区完整性、列投影、日期/品种过滤 | DuckDB SQL | 已有物理表和发布元数据 |
| 长表排序、连接、group aggregation、批量派生 | Polars | Arrow 数据布局、表达式和并行执行 |
| 日期×品种标签对齐、期限结构矩阵、回测矩阵 | Pandas，初期保留 | 现有索引语义成熟，迁移收益不明确 |
| 已向量化 NumPy/Bottleneck 算子 | 保留现状 | FFI 或转换可能比计算更贵 |
| `rolling.apply`、逐窗口 percentile/slope 等热点 | Rust 候选 | Python 回调和窗口循环有明确加速空间 |
| GP 内部纯 ndarray rolling | Rust 候选 | 边界清晰，无需理解 DataFrame 或期货语义 |

不得让 Polars 与 Rust 同时重写同一算子。先 profile，再依据数据布局和端到端成本选择一个
执行器。

## 4. 分阶段实施

### D0：冻结与版本收口

目标：建立所有后续 A/B 的唯一参考版本。

实施：

1. 合并依赖为唯一 `requirements.txt`，记录 Python、Pandas、Polars、DuckDB、PyArrow、
   NumPy 版本。
2. 提交当前 DuckDB 适配、健康检查、文档和已有研究治理修改。
3. 运行 `compileall`、全量 pytest、`pip check`、严格 DuckDB 健康检查。
4. 记录 Git commit、两个远端 commit、DuckDB release ID 和 Parquet/DuckDB最新日期。

通过条件：工作区干净、两个远端指向同一提交、认证库可读、所有门禁通过。

### D1：正式切换 DuckDB 运行源

目标：只替换物理读取源，不改变 Pandas 算法层。

实施：

1. 等待当晚 Parquet 与席位发布成功，再执行 `update_duckdb.py --mode sync`。
2. 在维护窗口关闭持有旧 DuckDB 连接的长进程。
3. 设置 `MF_DATA_SOURCE=duckdb_futures`、`MF_DUCKDB_PATH`、`MF_PARQUET_ROOT` 和本次
   `MF_DATA_RELEASE_ID`，重启进程。
4. 运行严格健康、代表日频/分钟/期限结构/席位任务。
5. 连续观察至少两个夜间增量 release。该运行观察门独立记为D1b；失败时只回退
   `MF_DATA_SOURCE=parquet_futures`。

不修改仓库默认源为 DuckDB：空路径的公开 checkout 必须仍可按显式配置启动；正式运行源
由部署环境冻结并记录。

### P1：Polars 结果桥接

目标：验证 DuckDB→Polars 的传输收益，公共输出仍为 Pandas。

最小代码计划：

- 修改 `requirements.txt`：固定已验证的 Polars 版本（D0 已完成）。
- 修改 `core/config.py`：在 `DuckDBConfig` 增加
  `result_backend: pandas | polars | shadow = pandas`，并增加对应环境覆盖。
- 修改 `config/default.yaml`：显式默认 `pandas`。
- 修改 `data/duckdb_source.py`：复用唯一 `_execute_df()`，通过 DuckDB `.pl()` 获取结果，
  显式规范日期/时间/dtype/列顺序后只转换一次为 Pandas。
- 扩展现有 `tests/test_duckdb_source.py`，首期不新增生产文件或新数据源类。

模式：

- `pandas`：权威现有路径。
- `shadow`：两种方式均读取，按语义比较，返回 Pandas reference。
- `polars`：仅在门禁通过后启用，仍返回现有 Pandas 公共对象。

基准必须覆盖日线长区间、1分钟多品种、30分钟重采样、期限曲线和六张席位表。只测
`.pl()` 微基准不构成准入证据。

### P2：Polars 数据层热点

目标：减少真实研究中的读取、整理和聚合成本。

实施顺序：

1. 对 DuckDB SQL、结果转换、主力计划、pivot、重采样、曲线与席位聚合分别 profile。
2. 只迁移占端到端时间显著的长表操作；小对象和标签矩阵继续 Pandas。
3. 第一处与第二处共同需要 Polars helper 时，才允许新增一个 `data/frame_ops.py`；
   在此之前保持逻辑位于现有数据源文件。
4. DataProvider 边界仍返回 `DataFrame(dates × roots)`，因子代码不感知 Polars。
5. 主框架P2稳定后，`factor_mining`正式入口已复用DataManager；独立`LocalParquetData`
   仅保留为显式Parquet审计/回退API，不再是正式CLI默认路径。

### C0：重新剖析因子计算

目标：在 DuckDB/Polars 数据成本下降后重新识别真正计算热点，避免依据旧画像造 Rust。

固定同一 commit、release、日期、universe、因子集合、线程数和缓存状态，分段记录：

- SQL/读取/转换；
- 分钟面板、选约、pivot；
- SPEC base/transform；
- intraday共享 helper和复杂 `rolling.apply`；
- GP expression、MAD、中性化、rank-IC；
- 总墙钟、CPU、复制字节和 peak RSS。

没有占端到端时间的热点，不进入 Rust。

### R1：本地 Rust 工程探针

目标：验证 Windows 构建、wheel、NumPy边界和FFI成本，不改变正式结果。

约束：

- 安装位置固定为 `E:\rust\rustup`、`E:\rust\cargo`、
  `E:\rust\target\multi_factor`。
- 不修改全局 PATH；命令使用显式 `E:\rust\cargo\bin\cargo.exe`。
- 模块名 `_mf_factor_kernels`，不使用 `alpha`，不依赖 `py-alpha-lib`。
- Rust 只接收 ndarray、shape、window等标量，不接触日期、合约、因子或DataFrame。

最小新增：

```text
native/mf_factor_kernels/
  Cargo.toml
  Cargo.lock
  pyproject.toml
  src/lib.rs
```

Python侧优先复用 `factors/numerics.py`；只有可选扩展加载和异常转换无法保持清晰时，
才新增一个薄文件 `factors/native_kernels.py`。不创建多文件通用 backend 包。

探针只实现 `ts_mean`，用于测量转换和FFI，不因为探针成功就替换现有 Bottleneck。

### R2：选择性 Rust 热点

候选优先级：

1. 新profile确认且现有NumPy/Bottleneck仍不能满足门槛的逐窗口Python循环；
2. GP rolling backend中实际存在、语义已冻结并占端到端时间的纯数组算子；
3. 只有多个正式因子共同复用时才考虑通用percentile/slope native核，不能为不存在的
   算子名称预建工程。

暂不迁移 shift、diff、pct_change、EMA/Wilder、截面/group rank、复杂期限结构、选约、
复权、夜盘、回测、风险和组合逻辑。mean/std/sum 已有向量化或Bottleneck时，必须先证明
包含复制和FFI后的端到端收益。

运行模式：

- `reference`：默认权威路径；Rust缺失不影响项目。
- `shadow`：reference/native同时计算，返回reference并记录差异和耗时。
- `native`：只允许已逐算子准入的白名单；扩展缺失或不支持时失败，不静默 fallback。

日常因子编写、SPEC、GP、research和回测命令不改变。少数热点因子内部可以等价改为调用
已有共享 helper，但不能要求因子作者直接调用 Rust。

## 5. 语义验收矩阵

| 层级 | 必须一致 | 可按契约容差 |
|---|---|---|
| 数据发布 | 表、分区、键、行数、业务值、release完整性 | 无 |
| 数据读取 | 日期/品种集合与顺序、交易日归属、缺失位置、合约选择 | 时间存储单位、字符串物理类型 |
| 因子算子 | 公式、window、min_periods、ddof、ties、NaN/Inf规则 | 浮点归约末位误差 |
| 因子矩阵 | shape、标签、mask、方向、排名与阈值侧别 | 不影响排名/信号的微小数值误差 |
| 策略与账本 | 选品、权重约束、订单、持仓、换月、成本事件 | 纯报告浮点末位误差 |

若数值处于选品、rank、零值、风险上限或交易阈值附近，必须增加边界样本并要求下游离散
决策一致，不能只看 `allclose`。

覆盖样本至少包括：

- 1d、1m、5m、15m、30m、hourly；
- 日夜盘、跨午夜、节假日前后和缺 bar；
- 主力切换、退市、FU经济时期；
- 近远月期限结构及结算价；
- 行情与席位交易日对齐、delivery_date；
- 全NaN、短窗口、常数窗口、Inf、并列rank；
- SPEC共享base、intraday helper、GP AST路径。

## 6. 性能准入与停止条件

基准统一要求：一次warm-up、至少5次正式重复，报告中位数、离散度、peak RSS、线程数、
输入规模、缓存冷热和复制字节。

建议准入门：

- P1/P2：主导真实工作负载墙钟或peak RSS至少改善约15%～20%，常用路径不出现显著回退。
- R2：kernel含转换成本至少约2倍，相关真实workflow端到端至少改善20%。
- peak RSS原则上不高于reference 10%；多线程不得与现有线程池过订阅。
- 未达到门槛就停止该候选，不为了使用新技术扩大改造。

## 7. 发布、回退与审计

- 每个阶段独立提交，提交信息注明 source/backend 和已通过门禁。
- shadow输出必须记录代码commit、DuckDB release、配置、backend版本和测试样本，但不新增
  每次运行一个永久报告文件；优先写入现有research manifest或benchmark目录。
- P1/P2 回退只需切换 result backend；D1 回退只需切回 `parquet_futures`。
- Rust native以逐算子白名单发布；移除一个白名单项即可回退该算子。
- 不在同一次提交中同时切换 DuckDB默认源、Polars生产路径和Rust native。
- 不删除 Parquet、认证 DuckDB或现有 reference实现。

## 8. 实施者逐阶段清单

每个阶段开始前：

1. 读取本文硬约束与上一阶段验收记录。
2. 确认工作区干净、数据更新未进行、release固定。
3. 先 profile 或复现问题，再修改最少文件。
4. 列出本阶段明确不修改的模块。

每个阶段完成后：

1. 运行聚焦测试、全量测试、compileall和`pip check`。
2. 运行数据健康和真实A/B；记录事实与推断，不用微基准冒充端到端收益。
3. 逐行审阅新增边界、异常路径、fallback和配置默认值。
4. `git diff --check`，确认无临时文件、缓存、数据库或结果被提交。
5. 提交并推送两个远端；未通过门禁则保持reference，不进入下一阶段。

## 9. 当前阶段结论

截至2026-08-22，本轮实施结论如下：

- D0完成：唯一依赖清单、Polars 1.43.2、全量回归和双远端基线已冻结。
- D1a完成：本机运行源已绑定认证DuckDB release；仓库默认仍为Parquet。当前manifest与
  component ID一致、`changed_partitions={}`，因此没有执行无意义的重复sync。
- D1b待完成：仍需连续观察至少两个新的夜间增量release；这不影响当前认证库随时可读或
  通过环境变量回退Parquet，但完成观察前不得把增量运行稳定性写成已验收。
- P1完成：`pandas|polars|shadow`桥接已进入唯一`_execute_df()`边界；本机启用Polars，
  公共DataFrame、行情频率、期限结构和六张席位表语义保持不变。代码提交为`ffcd1b1`。
- P2停止扩张：席位长表和实际席位因子达到准入门；行情与期限结构仅有个位数改善，继续
  把标签矩阵或日期匹配改写为Polars不符合收益/风险边界。
- C0完成：四个重复Python窗口因子已在`6ee134b`改为共享日度聚合和小型NumPy核，真实
  输出哈希完全一致。随后完成38品种、164交易日、588因子全池画像：纯因子计算约
  1,517秒；期限结构共享预热约214秒，主流intraday仍是主要计算热点。
- C1完成：全池排名最前且公式重复的两个OHLC分段波动率因子复用一个NumPy向量化helper，
  真实输出SHA-256及6,042个有限值完全一致；代表性热缓存墙钟分别由约29.02秒降至
  14.14秒、25.36秒降至13.26秒。该结果只证明两个热点约减半，不外推为全框架倍数。
- C2完成：成交额持续性、季节性、滚动斜率、方差比、K线路径和VPIN均通过真实完整因子
  A/B；第二次38品种、164交易日、588因子画像为588/588无错误，纯因子墙钟由约1517秒
  降至约1111秒（`-26.8%`）。成交量稳定性两因子随后复用现有分钟面板内的同一日度矩阵，
  未新增缓存文件或全局缓存层。
- 适配性研究现在按唯一处理链记录`raw|neutralized`，冻结发现文件缺少合法variant会失败
  关闭；正式mining/screen也已收口到统一配置和DataManager。
- Parquet与DuckDB现在都从日线具体合约的首个已发布交易日提供品种上市/可用日期；真实
  U38为38/38无缺失，正式研究与周度回测入口均已完成真实烟测。最终全量回归为689项通过。
- R1/R2本轮No-Go而非永久排除：低风险热点用已有NumPy能力已获得明确端到端收益，因此不安装
  Rust工具链。后续每次真实全池profile都重新开放Rust候选；只有边界清晰的纯ndarray
  算子通过reference、shadow、端到端和RSS门禁后，才逐算子进入native白名单。

`config/default.yaml`现已收口为唯一38品种、10f观察基线和统一处理契约；旧的四个重复或
继承配置已删除。被Git忽略的`config/local.yaml`只能覆盖本机数据运行时字段，当前固定
`duckdb_futures + polars + required_release_id`，不能覆盖品种、因子、处理或回测语义。
当前主框架处理链是MAD＋Z-score且未声明中性化；research现在按实际步骤记录`raw`，不再
把未中性化结果误标为`neutralized`。正式mining/screen和主框架入口均校验同一有序品种
契约；只有经完整38品种screen写出的绑定快照可挂载到主框架，普通池快照或旧未绑定快照
继续可作审计读取，但不能注册为正式因子。
夜间发布新DuckDB release后，必须先通过发布验证，再更新该ID并重启；回退只需切回
`parquet_futures`。D1b不阻塞当前研究；现有认证release、统一配置、DuckDB/Polars读取和
计算热点均已通过门禁，可直接开始正式research、screen、selection与backtest。最终目标
仍是真实research端到端收益，不以使用某项技术作为完成标准。
