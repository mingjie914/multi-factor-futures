# 维护与代码管理

## 当前边界

主框架、`factor_mining` 插件和研究证据必须分开管理：

- 源码与配置模板进入 Git。
- SQLite 候选库、市场数据缓存和普通运行输出留在本机，不进入 Git。
- Git 只长期保留 `runs/README.md` 和 append-only holdout ledger。普通本地结果不进入
  Git；需要长期保留的大型证据应先写入只读归档，再从工作区清理。
- `config/target_publication.yaml`只能由人工批准流程修改，研究代码不得自动写入。

## 例行检查

```powershell
$PY = 'E:\Python\Pythonvenv\Scripts\python.exe'

& $PY -m pip check
& $PY -B -m compileall -q alpha backtest core data external_strategies factor_mining factors `
  monitoring optimization pipeline processing research risk scripts strategies testing `
  workflows tests main.py
& $PY -B -m pytest -q -p no:cacheprovider
& $PY -X utf8 -B main.py mining dev-smoke `
  --periods 5000 --symbols 20 --population 32 --generations 2 --jobs 1
```

测试耗时随机器、BLAS、测试数量和依赖版本变化；实测记录见性能报告，
不在维护步骤中维护另一套易过期的固定秒数或测试项数量。

## Rust数值核心

本地扩展保持单线程，不使用Rayon。修改`native/mf_factor_kernels/`后在VS开发环境中构建：

```powershell
. 'E:\rust\vs-buildtools\Common7\Tools\Launch-VsDevShell.ps1' -Arch amd64 -SkipAutomaticLocation
$env:RUSTUP_HOME = 'E:\rust\rustup'
$env:CARGO_HOME = 'E:\rust\cargo'
$env:CARGO_TARGET_DIR = 'E:\rust\target\multi_factor'
$env:CARGO_BUILD_JOBS = '1'
$env:VIRTUAL_ENV = 'E:\Python\Pythonvenv'
$env:Path = "E:\rust\cargo\bin;E:\Python\Pythonvenv\Scripts;$env:Path"
& $PY -m maturin develop --release --manifest-path native/mf_factor_kernels/Cargo.toml
```

扩展存在时框架自动使用native；`MF_FACTOR_KERNEL_MODE=shadow`用于双算验收，
`reference`用于回退，显式`native`在扩展缺失时失败关闭。新增核心必须先通过真实数组
reference/shadow对照，再运行全量测试与单线程全池画像。
正式research任务显式设置`MF_FACTOR_KERNEL_MODE=native`；运行时模式和扩展版本写入
既有`research_contract`，同时代码哈希覆盖Rust源码与Cargo锁文件。

## 性能原则

- 延长回测或复用因子面板时，保持合同约定的全局分块边界，并验证重叠期数值与缺失掩码。
  足够滚动预热不等于分块语义一致；已否决的短预热实验缓存不能冒充正式断点。
- 当前默认成员只由策略目录决定。旧组合冻结成员定义保存在config/factor_sets/，
  不依赖Git忽略的完整snapshot目录；本地原快照仅作历史审计。

- 先用剖析结果优化。当前 GP 主要是 CPU、内存带宽和 Pandas/NumPy 滚动计算，GPU
  不是首版瓶颈；训练 XGBoost、神经网络等黑箱模型时再增加独立 GPU 后端。
- `--jobs 1` 是稳定默认值。多线程可在本机基准后使用；表达式缓存总预算会按 worker
  数量拆分，不会为每个 worker 重复分配完整预算。
- GP Accelerator v2-lite 仍为显式 opt-in。线程只计算只读 terminal view 上的
  expression block；MAD、中性化、rank-IC 和 fitness 继续由主线程按 factor chunk
  处理。`ts_ema`、`ts_corr`、`ts_cov` 和未进入能力白名单的算子直接走 legacy。
- 修改 accelerator 后必须复用同一份固定 AST，比较 baseline、v1、factor chunk 和
  v2-lite，并核对 NaN mask、factor value、IC、direction、candidate 集合和排序。
- 分钟数据按明确训练区间加载，超过特征内存预算时失败关闭，不用交换分区硬撑。
- 并行因子共享同一请求的分钟面板；冷缓存只允许一次读取，避免多个 worker 重复扫描
  同一批已发布行情。可复用的已测内层循环放在 `factors/numerics.py`并按共享计算族接入
  Rust；没有逐值差分、shadow和端到端基准时不得新增native核心。
- 全量`factors.library.intraday`日频研究按400个目标交易日＋128日预热分块，避免把
  全历史分钟面板和因子中间量同时驻留内存；当前最长跨日依赖为120日，增加更长窗口时
  必须先提高并验证重叠长度。后置检验和相关分析复用同一分块助手；64仅是因子批大小。
- Parquet/DuckDB 数据源拥有的 selected-contract 和 curve cache 可以复用；发布指纹变化时会失效。
- 通用行情缓存命中更宽日期/品种覆盖时直接返回内存切片，不再为每个请求范围落一个
  派生 Parquet；不要恢复这种会持续制造重复缓存文件的写回行为。
- 因子批量计算依赖预取和 SPEC 按 base 分组，不要在单因子中重复读取相同字段。
- 生产组合的风险历史按信号日只读取一次，再在多头池和空头池内分别执行完整性筛选与
  ERC；不要重新引入同一日期的重复行情读取。
- 行情分区读取失败或正式流程取得空交易日历时必须失败关闭，不能把工作日历伪装成
  交易所日历。正式数据源只能是已发布Parquet或认证DuckDB，不再保留失效的`cache-only`分支。
- 生产因子直接读取本地分钟/日线分片时，任一已发现分片损坏必须终止；不得跳过坏分片
  或在同一次计算中静默切换到另一个行情源。

## 行情缺口与停牌语义

- 日线缺行不自动等同于停牌。未知上市后缺口必须失败关闭并先补权威数据源；不得为了
  让净值连续而填0收益。
- 日度前向收益按完整交易日历移位，不得在单品种上先删除缺行再跨日配对；空交易日历、
  空close或全空前向收益均须失败关闭，不能输出平坦净值。
- 只有交易所/权威源证实的停牌日才加入`data.audited_nontrading_closes`。该日沿用上一
  可观察收盘估值、收益为0且禁止调仓；下一次真实报价一次性计入跨期涨跌。
- NI 2022-03-10是当前唯一配置的已审计停牌。SC 2026-05-08属于日线漏数，已在
  数据发布侧补齐到本地Parquet，不进入停牌白名单。
- 郑商所YMM/YYMM别名在发布层统一处理；规范键数值冲突时停止发布，不按文件顺序
  静默选择。全库重复检查留在数据发布和`data-health --strict`，不放入因子热路径。
- 本框架只消费已发布的本地Parquet或其认证DuckDB镜像，不包含远程核对、回填或发布逻辑。
- 严格健康门同时覆盖日线、1/5/15分钟行情和六张席位表；`delivery_seat`即使当前没有
  因子直接消费，也必须与其他五张正式发布席位表一起通过自然键、规范根和分区检查。
- 连续价格与合约日程必须来自同一份点时主力选择。主力换月只向更远交割月份推进，
  不因近月残余持仓反超而倒退；仍要求决策当日双腿可成交，并按既定T-1日程执行。
  这不是完整到期日历：从未形成可成交接续的序列仍失败关闭，不用事后最后报价日补救。
  该语义使用selected缓存v6，旧v5缓存不复用；历史结果需在同版本下重新对照。
  组合账本逐日检查下一交易日具体合约，
  即使根权重不变、当天也不是常规调仓日，换月仍按旧约平仓＋新约开仓记录；停牌日延迟
  至首次可交易收盘执行。数据源不能提供合约日程时，账本元数据必须标记`unavailable`。
- 交易所重启上市或合约规格发生经济断代时，在`data.parquet.root_active_from`按品种配置
  生效日期；这不是普通上市日期，也不写进数据源特例。FU当前配置为2018-07-16，正式
  入口必须经`DataManager.from_config`构造数据源；直接实例化仅限显式传入同等配置的测试。

## 动态注册检查

`factors.library` 默认只注册内置因子，不导入SPEC或扫描user目录。
显式研究调用 `factors.library.load_research_factor_catalog()` 才加载SPEC/user候选目录；
`factors.user` 的显式加载仍按文件名排序扫描，但排除自动快照桥接模块。
新增候选文件需要测试和研究边界；候选生成不能自动写入intraday源码、有效库或策略配置。

Mined 因子只通过以下路径进入同一个 registry：

```text
SQLite candidate catalog -> immutable JSON snapshot
  -> main.py --mined-snapshot
  -> factor_mining.bridge.register_snapshot_from_environment (explicit call)
  -> ordinary Factor subclass
```

不得从 SQLite 直接运行因子，也不得让候选覆盖既有注册名。
遗留 `MF_MINED_CANDIDATE_SNAPSHOT` 环境变量本身不启动默认流程的快照加载。
默认因子计算与方向检验不反向导入生成器；只消费已显式注册因子的接口和元数据。

## 组合优化器生命周期

- `hierarchical_asset_risk_parity` 标记为 `formal_default`，是正式研究、回测与目标权重观察
  信号的唯一默认品种配置器。
- `mean_variance` 标记为 `research_only`，仅允许在预期收益已经按相同周期、相同单位
  完成样本外幅度校准的对照实验中使用。
- 两者不得串联；多周期 `meta_optimizer` 只在完整子组合之后配置资本。
- 已删除被三层结构取代的 `hierarchical_sector` 注册入口。ERC 数值核心仍由
  `risk_budgeting` 复用，不应作为遗留代码删除。
- 更改默认优化器、目标波动率、换手诊断或杠杆限制时，必须同步更新
  `docs/三层资产配置流程.md` 及对应测试。

因子统计准入、板块适配、后置交易属性检验与 Ridge 的完整先后关系见
`docs/因子检验与准入流程.md`。修改任何正式门槛或多重检验方法时必须同步更新
该文档及研究结果中的方法元数据。研究 bundle 同时绑定验证策略 SHA-256 与 taxonomy
SHA-256；任一变化必须新建输出目录并按影响范围重跑迁移对照或正式滚动WF。失效 bundle
在完成必要外部归档后应从工作区删除。

旧599/588/606/605迁移目录名记录当时范围，不代表当前注册池。正式单因子证据来自
`run_factor_workflow.py`的近期准入窗口；长历史回测和滚动WF只验证组合，不能混作全历史准入。
当前库与策略状态分别以其权威文件为准，已完成修复和本地证据链统一列在`runs/README.md`。

## 清理策略

可以直接再生并清理：`.pytest_cache/`、所有 `__pycache__/`、已结束测试的`.test_tmp/`、`_work/`、空的
`signals_output/`、`monitoring_data/` 和 `weeklyreport/`。清理前必须确认路径位于仓库内。

不要清理：`cache/` 和 `runs/factor_research/holdout_ledger.jsonl`。前者是本地行情缓存，
后者记录已消费 OOS。协议变更不自动授权删除旧run：被当前库、修复对照、
配置或任何仍保留合同引用的证据必须保留；确认未引用且仍需审计时，先保存manifest、哈希和归档位置。
正式研究使用不可变 study/run 目录，这是审计要求；只有明确的研究入口可以创建这类
目录。运维脚本默认覆盖固定目标或要求显式输出路径，不得在项目根目录自动堆积日期文件。

期货成本模型不再保留含义重叠的手续费、滑点兼容参数。筛选期将年化半换手还原为
完整成交名义后乘 0.02%；筛选成功后的研究回测按 `executed_traded_notional` 计提 0.02%，
并按总暴露摊销年化 0.105% 的保守移仓预算。`decision_turnover` 只作诊断；显式换月腿
进入实际成交名义和 0.02% 成本，不再额外触发另一笔固定移仓费。
旧配置若仍含 `commission_rate`、`slippage`、`turnover_penalty` 或
`max_monthly_turnover`，必须迁移后再运行，不能静默兼容。

## Git 与远程仓库

Git 是本地版本历史和可回滚边界；当前`origin`是Gitee、`github`是GitHub镜像。
推送前必须确认没有把行情、SQLite、普通runs、
本机配置或凭据加入暂存区：

1. 日常建议短分支与pull request；用户要求直接提交现有分支时，先审阅并完成本地测试。
2. 分支保护与CI是建议配置，不把未核准的远端设置写成已启用；不使用force push。
3. 按唯一依赖清单`requirements.txt`验证本地环境，运行语法检查和完整pytest；CI可配置Python 3.10/3.12矩阵。
4. 密钥、`config/local.yaml`、Parquet、SQLite 和普通 runs 继续由 `.gitignore` 排除。
5. 大型研究证据放对象存储或只读 NAS，并在 Git 中保存 manifest、哈希和证据 URI。
6. Git LFS 只适合少量必须版本化的大文件，不适合作为 1 分钟行情仓库。
7. 推送`origin`与`github`后，分别读取远端`refs/heads/main`并与本地提交核对；不只根据push返回值声称两端一致。

个人单机可选私有 GitHub；需要数据内网、权限审计或自托管时优先公司 GitLab/Gitea。
无论选择哪一种，Git 仍是本地版本历史和可回滚边界。
