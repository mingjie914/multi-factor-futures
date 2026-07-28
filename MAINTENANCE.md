# 维护与代码管理

## 当前边界

主框架、`factor_mining` 插件和研究证据必须分开管理：

- 源码与配置模板进入 Git。
- SQLite 候选库、市场数据缓存和普通运行输出留在本机，不进入 Git。
- Git 只长期保留 `runs/README.md` 和 append-only holdout ledger。普通本地结果不进入
  Git；需要长期保留的大型证据应先写入只读归档，再从工作区清理。
- `config/trading.yaml` 只能由人工批准流程修改，研究代码不得自动写入。

## 例行检查

```powershell
$PY = '.\.venv\Scripts\python.exe'

& $PY -m pip check
& $PY -B -m compileall -q alpha backtest core data factor_mining factors `
  optimization pipeline processing research risk signals strategies testing workflows tests main.py
& $PY -B -m pytest -q -p no:cacheprovider
& $PY -X utf8 -B main.py mining dev-smoke `
  --periods 5000 --symbols 20 --population 32 --generations 2 --jobs 1
```

完整测试在当前开发机通常约 10 秒。机器、BLAS、测试数量和依赖版本会改变绝对值，
持续集成更适合检查明显回退，而不是维护容易过期的固定秒数或测试项数量。

## 性能原则

- 先用剖析结果优化。当前 GP 主要是 CPU、内存带宽和 Pandas/NumPy 滚动计算，GPU
  不是首版瓶颈；训练 XGBoost、神经网络等黑箱模型时再增加独立 GPU 后端。
- `--jobs 1` 是稳定默认值。多线程可在本机基准后使用；表达式缓存总预算会按 worker
  数量拆分，不会为每个 worker 重复分配完整预算。
- 分钟数据按明确训练区间加载，超过特征内存预算时失败关闭，不用交换分区硬撑。
- 本地 Parquet 的 selected-contract 和 curve cache 可以复用；源文件指纹变化时会失效。
- 因子批量计算依赖预取和 SPEC 按 base 分组，不要在单因子中重复读取相同字段。

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

- `hierarchical_asset_risk_parity` 标记为 `formal_default`，是正式研究、回测与实盘参考
  信号的唯一默认品种配置器。
- `mean_variance` 标记为 `research_only`，仅允许在预期收益已经按相同周期、相同单位
  完成样本外幅度校准的对照实验中使用。
- 两者不得串联；多周期 `meta_optimizer` 只在完整子组合之后配置资本。
- 已删除被三层结构取代的 `hierarchical_sector` 注册入口。ERC 数值核心仍由
  `risk_budgeting` 复用，不应作为遗留代码删除。
- 更改默认优化器、目标波动率、换手诊断或杠杆限制时，必须同步更新
  `docs/three_layer_portfolio.md` 及对应测试。

因子统计准入、板块适配、后置交易属性检验与 Ridge 的完整先后关系见
`docs/factor_validation_pipeline.md`。修改任何正式门槛或多重检验方法时必须同步更新
该文档及研究结果中的方法元数据。研究 bundle 同时绑定验证策略 SHA-256 与 taxonomy
SHA-256；任一变化必须新建输出目录并全量重跑 P0。失效 bundle 在完成必要外部归档后
应从工作区删除。

## 清理策略

可以直接再生并清理：`.pytest_cache/`、所有 `__pycache__/`、`_work/`、空的
`signals_output/`。清理前必须确认路径位于仓库内。

不要清理：`cache/` 和 `runs/factor_research/holdout_ledger.jsonl`。前者是本地行情缓存，
后者记录已消费 OOS。`runs/factor_mining/`、普通 `runs/factor_research/<study_id>/` 和
回测输出在协议变更后应清理；若结果仍需审计，先保存 manifest、哈希和外部归档 URI。

期货成本模型不再保留单次手续费、滑点或按换手扣费的兼容参数。筛选期只使用年化
0.02%，筛选成功后的研究回测使用年化 0.02% 加 0.105% 移仓成本。换手只输出诊断；
旧配置若仍含 `commission_rate`、`slippage`、`turnover_penalty` 或
`max_monthly_turnover`，必须迁移后再运行，不能静默兼容。

## Git 与远程仓库

Git 已经是必需工具，不建议改用另一套源码管理系统。当前仓库没有 remote，建议增加
私有 GitHub、GitLab 或公司 Gitea 远程仓库：

1. 日常开发使用短分支和 pull request，不直接在 `main` 上累积大批改动。
2. `main` 启用保护：完整测试通过、至少一次审阅、禁止 force push。
3. CI 使用 Python 3.10 和 3.12，安装 `requirements-dev.txt`，执行 compileall 和 pytest。
4. 密钥、`config/local.yaml`、Parquet、SQLite 和普通 runs 继续由 `.gitignore` 排除。
5. 大型研究证据放对象存储或只读 NAS，并在 Git 中保存 manifest、哈希和证据 URI。
6. Git LFS 只适合少量必须版本化的大文件，不适合作为 1 分钟行情仓库。

个人单机可选私有 GitHub；需要数据内网、权限审计或自托管时优先公司 GitLab/Gitea。
无论选择哪一种，Git 仍是本地版本历史和可回滚边界。
