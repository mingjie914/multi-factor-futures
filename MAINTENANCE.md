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

完整测试在当前开发机通常约 10～20 秒。机器、BLAS、测试数量和依赖版本会改变绝对值，
持续集成更适合检查明显回退，而不是维护容易过期的固定秒数或测试项数量。

## 性能原则

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
  同一批已发布行情。可复用的已测内层循环放在 `factors/numerics.py`，不得全局替换 Pandas
  算子，也不得在没有逐值差分和端到端基准时引入原生构建依赖。
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
- 连续价格与合约日程必须来自同一份点时主力选择。组合账本逐日检查下一交易日具体合约，
  即使根权重不变、当天也不是常规调仓日，换月仍按旧约平仓＋新约开仓记录；停牌日延迟
  至首次可交易收盘执行。数据源不能提供合约日程时，账本元数据必须标记`unavailable`。
- 交易所重启上市或合约规格发生经济断代时，在`data.parquet.root_active_from`按品种配置
  生效日期；这不是普通上市日期，也不写进数据源特例。FU当前配置为2018-07-16，正式
  入口必须经`DataManager.from_config`构造数据源；直接实例化仅限显式传入同等配置的测试。

## 动态注册检查

`factors.library` 的导入会注册内置、SPEC 和 user 因子；`factors.user` 会按文件名排序
自动发现公开模块。新增 user 文件即改变发现池，必须同时增加测试并记录研究边界。

Mined 因子只通过以下路径进入同一个 registry：

```text
SQLite candidate catalog -> immutable JSON snapshot
  -> main.py --mined-snapshot
  -> factors/user/auto_mined_bridge.py
  -> ordinary Factor subclass
```

不得从 SQLite 直接运行因子，也不得让候选覆盖既有注册名。

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
SHA-256；任一变化必须新建输出目录并全量重跑 P0。失效 bundle 在完成必要外部归档后
应从工作区删除。

当前正式P0证据为`runs/factor_research/20260820_intraday599_rebuild/`。目录名记录最初
599个历史注册类；11个不可估计死定义已清理，现行日内发现池为588。结果中的
`research_contract`绑定运行时代码、配置、Parquet元数据和候选名；20个统计发现经
`|corr|>=0.5`去重为13个观察候选，生产批准仍为0。

## 清理策略

可以直接再生并清理：`.pytest_cache/`、所有 `__pycache__/`、`_work/`、空的
`signals_output/`、`monitoring_data/` 和 `weeklyreport/`。清理前必须确认路径位于仓库内。

不要清理：`cache/` 和 `runs/factor_research/holdout_ledger.jsonl`。前者是本地行情缓存，
后者记录已消费 OOS。`runs/factor_mining/`、普通 `runs/factor_research/<study_id>/` 和
回测输出在协议变更后应清理；若结果仍需审计，先保存 manifest、哈希和外部归档 URI。
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

1. 日常开发使用短分支和 pull request，不直接在 `main` 上累积大批改动。
2. `main` 启用保护：完整测试通过、至少一次审阅、禁止 force push。
3. CI 使用 Python 3.10 和 3.12，安装唯一依赖清单 `requirements.txt`，执行 compileall 和 pytest。
4. 密钥、`config/local.yaml`、Parquet、SQLite 和普通 runs 继续由 `.gitignore` 排除。
5. 大型研究证据放对象存储或只读 NAS，并在 Git 中保存 manifest、哈希和证据 URI。
6. Git LFS 只适合少量必须版本化的大文件，不适合作为 1 分钟行情仓库。

个人单机可选私有 GitHub；需要数据内网、权限审计或自托管时优先公司 GitLab/Gitea。
无论选择哪一种，Git 仍是本地版本历史和可回滚边界。
