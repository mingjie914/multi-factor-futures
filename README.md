# Multi-factor futures framework

This project researches daily futures factors and turns only audited evidence
into portfolio candidates. It is not configured to trade by default.

Current status (2026-07-26): **NO_TRADE**. The expanded 3,111-factor historical
census found no factor that passed the predeclared global Bonferroni gate, so
Ridge weighting, portfolio optimization and a new portfolio backtest were not
run.

## Three operational layers

| Layer | Command | Responsibility |
|---|---|---|
| Research | `python main.py research` / `adaptivity` | Point-in-time factors, raw OLS/HAC, multiplicity and robustness tests |
| Validation | `python main.py walkforward` / `backtest` / `multi` | Frozen candidates, costs, risk limits and portfolio evidence |
| Close decision | `python main.py close` | Fail-closed approval gate; emits `NO_TRADE` without an approved deployment package |

Running `python main.py` with no command only prints help. Experimental
Supertrend and comparison workflows remain under `workflows/experiments/` and
are deliberately absent from the production command list.

## Required order

```text
predeclared factor exposures
  -> raw unpenalized univariate OLS + HAC p-values
  -> global Bonferroni
  -> IC / hit-rate / turnover / stability checks
  -> correlation deduplication
  -> economic-family caps
  -> training-only Ridge weights
  -> costed risk-constrained portfolio
  -> frozen walk-forward / locked OOS
  -> explicit trading approval
```

Ridge never supplies screening p-values. If the strict candidate pool is empty,
the downstream portfolio stages stop.

## Factor registry

- Registered factors: **3,111**.
- New practical candidates: **1,512** = 42 economic bases x 6 windows x 6
  transforms.
- New inputs use daily OHLCV and, for six participation bases, open interest.
- Missing OI invalidates only OI-dependent groups; other factors continue.
- The 192 minute-level factors fail closed while the DolphinDB client/data are
  unavailable. Daily data is never substituted for minute data.

The practical SPEC definitions are in `factors/specs/practical.py`; vectorized
implementations are in `factors/practical_bases.py`.

## Data and assumptions

- Alibaba Cloud MySQL connection: healthy on 2026-07-26.
- Historical calendar used by the latest study: 2,321 trading days.
- Warm-up: 2017-01-01 onward; evaluation: 2018-01-01 through 2026-07-24.
- Universe: 47 futures under eight sectors. Stock index/bond and
  nonferrous/precious metals are separate sectors.
- Risk-free rate: 0 for Sharpe and related metrics.
- Trading cost estimate: 0.02% of traded notional.
- Annual roll/management cost: 0.105%.

## Current evidence

The immutable study is
`runs/factor_research/expanded_census_3111_2018_20260726/`.

- Global tests: 18,666; Bonferroni alpha: 2.6787e-6; passed: 0.
- Sector-period tests: 103,434; Bonferroni alpha: 4.8340e-7; passed: 0.
- Study audit: valid.
- Status: `historical_full_sample`; it overlaps consumed 2025-07-01 to
  2026-07-24 OOS evidence and is not independent validation.

See `最新因子与组合表现说明.md` for the closest research-only signals and
`多因子框架研究手册.md` for method details.

## Common commands

```powershell
python main.py data-health --config config/default.yaml
python main.py research --help
python main.py adaptivity --help
python main.py walkforward --help
python main.py close --as-of 2026-07-24
```

`config/default.yaml` intentionally contains no selected factors. Research
artifacts never modify it. `config/trading.yaml` is the separate disabled
trading approval gate.

## Run retention

- `runs/locked_oos/`: retained consumed holdout evidence; never rewrite.
- `runs/factor_research/`: immutable protocol-based studies.
- Old reproducible runs were moved out of the workspace to
  `E:\程明杰公司内容\multi_factor_legacy_20260726`.
- `cache/` is market-data cache and is retained.

Run tests without repository cache churn:

```powershell
python -B -m pytest -q -p no:cacheprovider
python -B -m compileall -q alpha backtest core data factors optimization pipeline processing research risk signals strategies testing workflows tests main.py
```
