# 运行产物保留规则

`runs/` 只保存运行证据，不是生产配置。正式研究按不可变 study/run 目录输出，避免
覆盖后失去数据边界、参数和哈希；这类时间戳目录是审计设计，不应改成可变 `latest/`。

## 永久保留

- `factor_research/holdout_ledger.jsonl`：只追加的已消费 holdout 记录，禁止删除、
  修改或把其中日期重新称为未见样本。

## 历史审计证据

- `factor_research/20260820_intraday599_rebuild/`：截至2026-08-20的全历史正式单因子
  研究证据。运行提交599个历史类，588个可估计；形成20个统计发现、13个相关簇，
  发布批准数为0。目录保留筛选结果、验证漏斗、相关矩阵JSON和相关图，不保存可再生
  检查点。
- `external_guosen_trend_index/20260817_correctness_rebuild/`：最终频率路由与因子实现
  修复前的固定集合审计资料；其中绩效不得作为当前策略结论。

下列旧运行已因分钟根前缀污染或组合口径不一致而删除，不得从外部副本恢复为当前证据：
`20260810_full_prod_sort`、`20260813_contract_symbol_fix`、
`20260815_13f_8f_union`、`20260815_latest_core_compare`和
`20260816_correctness_rebuild`。

当前固定定义位于 `snapshot/`；快照和运行证据均不会发布订单。
`config/target_publication.yaml` 是独立的目标权重发布门，关闭时固定为 `NO_TARGETS`。

## 清理规则

- 中断且无结论的目录、根目录散落图表、被修正版替代的结果可以直接删除。
- 普通研究目录在结论失效后可清理；仍需长期审计的证据应先保存 manifest、哈希和
  外部只读归档位置。
- 完成的工作流必须删除 `_factor_panel_cache.pkl` 等可再生中间缓存。
- 研究入口不得把图、CSV 或 JSON 直接写到项目根目录；输出必须落入显式 run 目录。
