# 主框架最短接入说明

本文件是 `multi_factor` 主框架加载自动挖掘因子的接口契约。

## 快速流程

以下示例使用本机已安装的项目解释器：

```powershell
$PY = 'E:\Python\Pythonvenv\Scripts\python.exe'
```

### 1. 挖掘到候选池

以下周期全部是 1 分钟 bar 数，不是交易日。先用一个目标周期运行；不同目标周期应
建立不同 run，避免搜索后再挑选周期而低估多重检验。

```powershell
& $PY -B main.py mining `
  --repository 'runs\factor_mining\candidates.sqlite3' mine `
  --start '2023-01-01' --end '2024-12-31' `
  --frequency 1min --horizon-bars 15 `
  --sector-neutralization `
  --population 160 --generations 8
```

此步骤只写独立 SQLite，候选状态为 `mined_candidate`。控制台 IC/IR/收益仅用于搜索
诊断，不是正式入围证据。

### 2. 全量预筛并冻结机械合格集合

```powershell
& $PY -B main.py mining `
  --repository 'runs\factor_mining\candidates.sqlite3' screen `
  --start '2023-01-01' --end '2024-12-31' `
  --screen-id 'screen_id' `
  --output-dir 'runs\factor_mining\screens\screen_id' `
  --candidate-ids 'gp_xxxxx,gp_yyyyy' `
  --min-coverage 0.50
```

`mine`和`screen`默认读取统一的`config/default.yaml`，本机DuckDB/Polars选择只来自受限的
`config/local.yaml`或`MF_DATA_*`运行时覆盖。`--data-root`只用于明确的Parquet审计或回退，
不会另建品种、因子、处理或回测配置。

若要一次预筛某次挖掘 run 的全部候选，可把 `--candidate-ids` 换成
`--candidate-run-ids 'mine_run_id'`；候选 ID、候选文件和 run ID 三种选择方式互斥，
选择结果为空会失败关闭，不会生成“完成但零候选”的快照。

使用输出目录中的 `prescreen_candidates.snapshot.json`。`screen` 不按任意 quota 截断，
相关性和成本结果是注释；只有结构或数据机械无效的候选会被排除。

SQLite 与 `Factor`/spec 不冲突：SQLite 是可变研究目录；JSON 是不可变交付快照；
`Factor` 是主框架运行时接口。首版一般 GP 公式不能无损表达为现有
`base + transform`，因此使用 `Factor` 桥接，而不强塞进 `SpecFactor`。

### 3. 让主框架识别候选

主框架命令直接使用 `--mined-snapshot`，无需先修改配置或环境变量：

```powershell
& $PY -X utf8 -B main.py research `
  --mined-snapshot 'runs\factor_mining\screens\screen_id\prescreen_candidates.snapshot.json' `
  --config config/default.yaml `
  --factors 'mined_gp_xxxxx,mined_gp_yyyyy' `
  --multi-period --periods '15' --frequency 1min
```

周期必须匹配该批候选的挖掘目标。H=1、H=5 和 H=15 候选应拆成三个冻结研究，不能
把同一批公式事后放进多个周期挑最优。

`main.py` 会先校验快照，再导入具体工作流。随后
`factors/user/auto_mined_bridge.py` 会：

1. 校验快照 SHA-256、候选内容哈希、数量和名称唯一性；
2. 把每个符号候选转换为无参数 `Factor` 子类；
3. 通过现有 `register_user_factor` 注册；
4. 保留候选声明的依赖、1 分钟频率、决策 lag、MAD 和波动率中性化。

未传入 `--mined-snapshot` 时，该文件不做任何注册，现有框架行为完全不变。网关内部使用
`MF_MINED_CANDIDATE_SNAPSHOT` 在导入前传递已校验路径，不应由普通研究命令手工设置。
主框架不会读取 SQLite，也不会访问远程行情源。

### 4. 用现有研究流程正式筛选

注册名来自 `pool-list` 的 `framework_name`，例如：

```powershell
& $PY -X utf8 -B main.py research `
  --mined-snapshot 'runs\factor_mining\screens\screen_id\prescreen_candidates.snapshot.json' `
  --config config/default.yaml `
  --factors 'mined_gp_xxxxx,mined_gp_yyyyy' `
  --multi-period --periods '15' --frequency 1min `
  --factor-start '2022-10-01' --start '2023-01-01' --end '2024-12-31' `
  --output-dir 'runs\factor_research\study_id\raw' `
  --refuse-existing-output
```

正式 `mine`、`screen` 与主框架研究统一使用 `FRAMEWORK_UNIVERSE`；`--universe`
仅用于显式复述同一有序全集，传入不同集合或顺序会失败关闭。`factor-start` 仍需覆盖
因子自身的技术指标预热。

正式执行时应先冻结候选快照、因子名、bar 周期、日期和检验边界，并为每次研究指定
新的 `--output-dir` 与 `--refuse-existing-output`。`main.py research` 输出的
`ic_by_window_period.json` 会记录精确配置、验证策略与 taxonomy 哈希、完整假设数、
因子级 Simes/BH、selection-adjusted 因子内 BH、报告用 FWER 标签和逐因子结果；
`validation_funnel.json` 另存完整漏斗及阈值 ±20% 敏感性。正式查看真实历史的 IC、
HAC t 值、收益或最优周期必须走该流程；合成数据调试、表达式编码和单元测试不需要
机械执行完整协议。

当前正式发现门槛读取 `validation_policy`：层级 FDR `q=0.10`、`|IC|>=0.01`、
`|t|>=2.0`。Bonferroni/FWER 只作证据标签，不再是硬闸门。三分组、分钟换手按交易日
聚合、自然年稳定性和成本检查均在同一输出中记录。筛选期仅扣年化 0.02% 固定成本，
换手仅作诊断且不是准入门槛；筛选成功后的研究回测再加入年化 0.105% 移仓成本。

### 5. 将冻结发现集交给部署适配与 WF

手工运行部署参数适配时，必须把 P0 发现文件作为合同传入，不能把任意注册表清单直接
送入 `deployment` 模式：

```powershell
& $PY -X utf8 -B main.py adaptivity `
  --mined-snapshot 'runs\factor_mining\screens\screen_id\prescreen_candidates.snapshot.json' `
  --config config/default.yaml `
  --fdr-method deployment `
  --discovery-file 'runs\factor_research\study_id\ic_by_window_period.json' `
  --frequency 1min --periods '1,5,15' `
  --factor-start '2022-10-01' --ic-start '2023-01-01' --ic-end '2024-12-31' `
  --output-dir 'runs\factor_research\study_id\deployment'
```

该命令会自动读取发现阶段的 `final_factors`、通过的 raw/neutralized 版本、观察期原因和
权重上限，并校验验证策略与 taxonomy 哈希。更推荐使用 `main.py walkforward`：每一折
会自动执行“训练期发现 → 冻结发现集部署适配 → 相关性/家族治理 → Ridge → 测试折”，
主框架无需额外读取 SQLite，也不会自动修改交易配置。

不要把 `mined_candidate` 直接加入交易配置。通过审计后，才用审计 JSON 记录状态：

```powershell
& $PY -B main.py mining `
  --repository 'runs\factor_mining\candidates.sqlite3' promote gp_xxxxx `
  --status development_candidate `
  --evidence-json 'runs\factor_research\study_id\audit.json' `
  --run-id 'study_id'
```

`development_candidate`、`historical_candidate`、`oos_validated` 只表示证据阶段；均不
自动进入组合。组合前仍需相关性、成本、容量、板块暴露和风险审核。

## 主框架代码调用

需要在 Python 进程内显式加载时，可直接调用：

```python
from factor_mining.bridge import register_snapshot

names = register_snapshot("runs/factor_mining/screens/screen_id/prescreen_candidates.snapshot.json")
```

之后 `core.registry.get("factor", names[0])()` 与任何现有 `Factor` 的调用方式相同。

## 失败策略

- 快照被修改、表达式哈希不一致、名称重复：拒绝加载。
- 缺少声明依赖或数据全空：因子输出全 NaN，不使用代理字段。
- 黑箱 `model` 候选：首版桥接器明确拒绝。
- 特征内存超过预算：挖掘进程失败并提示缩小数据/特征范围。
- 环境变量为空：零副作用。
