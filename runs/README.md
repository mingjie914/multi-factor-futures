# Run retention policy

`runs/` contains evidence, not source configuration.

- `locked_oos/` is append-only consumed holdout evidence. Do not delete, edit,
  or reuse its dates as unseen validation data.
- `factor_research/<study_id>/` contains immutable protocol-based studies.
  Only the current registry census and explicitly retained bounded studies
  belong here.
- Ad hoc backtests and experiments must use a new run directory and may be
  removed after their conclusions are superseded.
- No run artifact is production configuration. `config/trading.yaml` is the
  separate fail-closed approval gate.

The current production decision is `NO_TRADE`: no factor or portfolio has been
approved for paper or live trading.
