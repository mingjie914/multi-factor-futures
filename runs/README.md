# Run retention policy

`runs/` contains generated artifacts, not source configuration.

- `factor_research/holdout_ledger.jsonl` is the permanent append-only record of
  consumed holdouts. Never delete, edit, or reuse those dates as unseen data.
- `factor_research/<study_id>/`, `factor_mining/`, and backtest directories are
  disposable workspace copies. A policy/hash/data-boundary change invalidates
  comparison and requires a fresh output directory.
- Evidence that must survive cleanup belongs in read-only external storage;
  keep its manifest, hashes, and archive URI with the study report.
- Ad hoc backtests and experiments must use a new run directory and may be
  removed after their conclusions are superseded.
- `historical_portfolio_search/<timestamp>/` contains the current-library,
  expanding-window method/factor search. Keep the latest conclusion-linked run;
  interrupted runs may retain `_factor_panel_cache.pkl` only until resumed or
  explicitly discarded, and completed runs remove that temporary cache.
- The retained conclusion-linked run is
  `historical_portfolio_search/20260810_full_prod_sort/`; its strict expanding-OOS
  ledger, method/factor evidence, metrics, and two NAV figures are the current
  comparison bundle. Superseded or invalid attempts should not be retained.
- `historical_portfolio_search/20260813_contract_symbol_fix/` is the focused,
  two-recipe R8 walk-forward freeze audit. Retain its resolved boundary,
  fold decisions, weights, ledger, metrics, figure, and review; it is research
  evidence and does not alter the production gate.
- `external_guosen_trend_index/20260813_contract_symbol_fix/` is the current
  frozen-set comparison after CZCE contract-symbol cleanup. It recomputes
  6f/10f/13f/14f/R8 under gross exposure 1 and 2 without rerunning factor selection.
- No run artifact is production configuration. `config/trading.yaml` is the
  separate fail-closed approval gate.

Artifacts created before the 2026-07-28 cost-policy reset were purged locally.
The current production decision is `NO_TRADE`: no factor or portfolio has been
approved for paper or live trading, and no current-policy result bundle exists yet.

On 2026-08-09, superseded local run workspaces were moved to the recoverable
archive `E:/程明杰公司内容/multi_factor_artifact_archive_20260809`.  The project
directory retains the append-only holdout ledger, current production benchmark,
current 6/13/14 comparison, report-linked figures, and external-strategy snapshots.
