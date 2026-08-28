# 有效因子库

这里保存“已经通过统一因子有效性门槛”的当前结构化清单，不保存组合选择结果。

## 文件

- `library.json`：唯一权威当前库，适合程序读取和Git审计。
- `current.csv`：由`library.json`生成的人工作业视图，不手工修改。
- `legacy/`：旧Markdown因子库迁移时冻结的历史快照，只用于对照。

每条当前记录包含因子名、注册候选周期、OOS批准周期`approved_periods`、方向、入库日期、来源run、关键IS/OOS
统计和证据文件哈希。完整的逐假设检验值仍保存在不可变
`runs/factor_validation/<run_id>/`，避免把大型结果重复塞进当前库。

## 日常流程（IDE优先）

直接在IDE打开项目根目录`run_factor_workflow.py`，只修改顶部`IDE SETTINGS`后点击Run：

- `FactorWorkflow.VALIDATE_ALL_INTRADAY`：检验全部注册日内因子；
- `FactorWorkflow.ADMIT_COMPLETED_RUN`：审阅完成后显式入库。
- `FactorWorkflow.SELECT_EFFECTIVE_SUBSETS`：只从当前已入库的有效因子派生
  平行子集；它不是全量有效性检验入口，也不会把未通过检验的候选因子带入子集。

三个分支的数量口径严格分离：假设全量注册集合为 688、其中新增 25 个通过而
原有效库为 75，则全量检验明细为 688 行，显式入库后有效库为 100 条，子集筛选
输入为这 100 条。代码不写死 588、75 或 100，均从注册表或 `library.json` 动态读取。

该入口没有全历史分支，默认检验固定使用统一截止日前的90日历日预热 + 126交易日IS +
42交易日OOS。命令行只作为自动化兼容入口：

```powershell
$PY = 'E:\Python\Pythonvenv\Scripts\python.exe'

# 兼容入口：日期窗口仍由代码中的统一政策解析，不能路由到全历史
& $PY -X utf8 -B main.py factor-validation `
  --config config/default.yaml --run-id <run_id>

# 人工确认检验结果后显式入库
& $PY -X utf8 -B main.py factor-library --config config/default.yaml admit `
  --run-dir runs/factor_validation/<run_id> --admitted-at YYYY-MM-DD

# 只读校验
& $PY -X utf8 -B main.py factor-library --config config/default.yaml check
```

检验入口没有入库选项；检验与入库只能分别执行，避免一次Run同时完成研究判断和状态变更。
有效因子库不会修改`strategies/combined.py`、`config/default.yaml::factors`或任何目标权重
发布设置；后续相关性去重、信号合成和组合选择必须显式读取本库并产生各自独立证据。
确认策略应在自己的完整YAML中设置`factor_library.enforce_portfolio_periods: true`，使单组合
持有期或各子组合持有期在回测前与`approved_periods`逐因子核对。

有效库之后的平行因子子集不写回本目录，统一登记在
`config/strategy_library.yaml::factor_sets`。子集可以基于不同品种、杠杆、容量或组合目的
形成，彼此没有默认排名；`selection_context`记录形成原因，真正的杠杆、品种和风险约束仍由
引用它的完整策略YAML执行，避免同一参数维护两份。
