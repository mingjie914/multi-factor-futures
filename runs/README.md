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
- No run artifact is production configuration. `config/trading.yaml` is the
  separate fail-closed approval gate.

Artifacts created before the 2026-07-28 cost-policy reset were purged locally.
The current production decision is `NO_TRADE`: no factor or portfolio has been
approved for paper or live trading, and no current-policy result bundle exists yet.
