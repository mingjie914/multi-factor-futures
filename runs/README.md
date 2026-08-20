# 运行产物保留规则

`runs/` 只保存运行证据，不是生产配置。正式研究按不可变 study/run 目录输出，避免
覆盖后失去数据边界、参数和哈希；这类时间戳目录是审计设计，不应改成可变 `latest/`。

## 永久保留

- `factor_research/holdout_ledger.jsonl`：只追加的已消费 holdout 记录，禁止删除、
  修改或把其中日期重新称为未见样本。

## 当前结论相关证据

- `external_guosen_trend_index/20260817_correctness_rebuild/`：本轮底层正确性门禁通过后
  生成的当前固定集合证据，包含8f/10f/13f两种方法、国信参考、方法邻域、集中度、
  10f账本与图表。它是固定集合重评估，不是修复后重新完成的因子搜索。

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
