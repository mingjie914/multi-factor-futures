from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from backtest.research_ledger import build_close_marked_ledger
from core.config import ProductionPortfolioConfig, load_config, validate_portfolio_config
from research.historical_portfolio_search import PortfolioEvaluator, PortfolioRecipe


def policy(method="equal"):
    return {"method": method, "lookback": 126, "minimum_observations": 60,
            "covariance_shrinkage": .3, "min_weight": .01, "max_weight": .15,
            "final_asset_max_fraction": .2, "final_sector_max_fraction": .6}


def fixture(monkeypatch, method="equal", shock=False):
    dates = pd.bdate_range("2024-01-02", periods=145)
    symbols = [f"a{i:02}" for i in range(38)]
    names = [f"f{i}" for i in range(9)]
    sectors = {s: f"sector{i % 5}" for i, s in enumerate(symbols)}
    rng = np.random.default_rng(195)
    returns = pd.DataFrame(rng.normal(0, .012, (len(dates), 38)), index=dates, columns=symbols)
    runner = SimpleNamespace(cal=dates, u=symbols, daily_ret=returns,
                             env=SimpleNamespace(sector_of=sectors))
    sleeves = {}
    for j, name in enumerate(names):
        weights = pd.DataFrame(0., index=dates, columns=symbols)
        weights.iloc[:, [(j*3+k) % 38 for k in range(10)]] = .1
        weights.iloc[:, [(j*3+19+k) % 38 for k in range(10)]] = -.1
        sleeves[name] = build_close_marked_ledger(weights, returns, trade_cost_rate=.0002)
        if shock:
            sleeves[name].daily.loc[dates[100]:, "gross_return"] *= j+2
    original = PortfolioEvaluator.run
    calls = []

    def run(self, factors, recipe, *, cost_multiplier=1.):
        if not recipe.factor_sleeves:
            assert recipe.factor_weight == "equal" and recipe.rank_exit_buffer == 1
            calls.append(tuple(factors))
            return sleeves[factors[0]]
        return original(self, factors, recipe, cost_multiplier=cost_multiplier)

    monkeypatch.setattr(PortfolioEvaluator, "run", run)
    recipe = PortfolioRecipe.from_config(ProductionPortfolioConfig(rank_exit_buffer=1, factor_sleeves=policy(method)))
    ev = PortfolioEvaluator(runner, start=dates[0], end=dates[-1])
    return ev, recipe, names, dates, calls


@pytest.mark.parametrize("method", ["equal", "erc"])
def test_sleeves_keep_union_delay_history_and_charge_final_account_once(monkeypatch, method):
    ev, recipe, names, dates, calls = fixture(monkeypatch, method)
    result = ev.run(names, recipe)
    assert len(calls) == 9
    active = result.target_weights.abs().sum(axis=1) > 1e-12
    assert result.target_weights.index[active][0] == dates[61]
    weights = result.target_weights.loc[active]
    np.testing.assert_allclose(weights.clip(lower=0).sum(axis=1), 1, atol=1e-9)
    np.testing.assert_allclose(-weights.clip(upper=0).sum(axis=1), 1, atol=1e-9)
    assert weights.abs().max().max() <= .2 + 1e-9
    assert (weights.abs()>1e-12).sum(axis=1).max() > 20
    for sign in (1, -1):
        mass = (sign*weights).clip(lower=0).T.groupby(ev.runner.env.sector_of).sum().T
        assert mass.max().max() <= .6+1e-9
    expected = ev._run_weights(result.target_weights)
    pd.testing.assert_frame_equal(result.daily, expected.daily)
    stressed = ev.run(names, recipe, cost_multiplier=2.)
    assert len(calls) == 9  # The same native sleeves are reusable; only final fees change.
    pd.testing.assert_frame_equal(stressed.target_weights, result.target_weights)
    # Costs change NAV and subsequent normalized turnover; compare the new
    # ledger's own traded amount, not twice the old ledger's fee series.
    np.testing.assert_allclose(stressed.daily.trade_cost, stressed.daily.executed_traded_notional*.0004, atol=1e-12)
    pd.testing.assert_frame_equal(stressed.daily, ev._run_weights(result.target_weights, cost_multiplier=2.).daily)
    diagnostics = result.metadata["factor_sleeves"]
    assert diagnostics["history_return"] == "gross_return"
    assert diagnostics["risk_timing"] == "strictly_prior_close"


def test_sleeve_risk_estimation_never_uses_current_or_future_return(monkeypatch):
    with monkeypatch.context() as context:
        ev, recipe, names, dates, _ = fixture(context, "erc")
        before = ev.run(names, recipe).target_weights
    with monkeypatch.context() as context:
        ev, recipe, names, _, _ = fixture(context, "erc", shock=True)
        after = ev.run(names, recipe).target_weights
    pd.testing.assert_frame_equal(before.loc[:dates[100]], after.loc[:dates[100]])
    assert not np.allclose(before.loc[dates[101]:], after.loc[dates[101]:])


def test_sleeve_config_is_opt_in_and_rejects_ambiguous_routes():
    config = load_config("config/default.yaml")
    assert config.production_portfolio.factor_sleeves is None
    original = PortfolioRecipe.from_config(config.production_portfolio).to_dict()
    assert "factor_sleeves" not in original
    config.production_portfolio = ProductionPortfolioConfig(factor_sleeves=policy())
    validate_portfolio_config(config, factor_names=[f"f{i}" for i in range(9)])
    with pytest.raises(ValueError, match="bounds"):
        validate_portfolio_config(config, factor_names=["one"])
    config.production_portfolio.investment_universe = ["IF", "IC"]
    with pytest.raises(ValueError, match="factor.sleeve"):
        validate_portfolio_config(config)


def test_sleeves_reject_incomplete_risk_history(monkeypatch):
    ev, recipe, names, dates, _ = fixture(monkeypatch)
    ev.run(names, recipe)
    sleeve = next(iter(ev._native_sleeve_cache.values()))
    sleeve.daily.loc[dates[80], "gross_return"] = np.nan
    with pytest.raises(ValueError, match="history is incomplete"):
        ev.run(names, recipe)


def test_sleeves_reject_short_history_instead_of_reporting_flat_nav(monkeypatch):
    ev, recipe, names, dates, _ = fixture(monkeypatch)
    ev.dates = dates[:50]
    with pytest.raises(ValueError, match="insufficient common actual returns"):
        ev.run(names, recipe)


@pytest.mark.parametrize("weights", [[0., 0.], [.2, .3], [np.nan, -.1]])
def test_final_projection_rejects_missing_or_invalid_exposure(weights):
    from optimization.portfolio_construction import project_netted_weights
    with pytest.raises(ValueError):
        project_netted_weights(pd.Series(weights, index=["a", "b"]),
            sector_of={"a": "x", "b": "y"}, gross_exposure=2.,
            asset_max_fraction=1., sector_max_fraction=1.)


def test_registered_sleeve_candidates_join_twelve_but_not_formal(monkeypatch):
    import run_portfolio_workflow as ide
    from trading.weights import strategy_contracts
    monkeypatch.setattr(ide, "WORKFLOW", ide.PortfolioWorkflow.RUN_AND_COMPARE)
    monkeypatch.setattr(ide, "STRATEGY_IDS", ())
    _, catalog, specs = ide._validated_specs()
    assert len(specs) == 12 and len({s.id for s, _, _ in specs}) == 12
    methods = {"curve_essence_sleeve_equal": "equal", "curve_essence_sleeve_erc": "erc"}
    sets = {s.id: s for s in catalog.factor_sets}
    for s, _, c in specs:
        if s.id not in methods:
            continue
        assert s.status == "observing" and not s.formal
        assert s.factor_set_id == "curve_essence"
        assert c.factors == sets["curve_essence"].factors
        assert c.production_portfolio.factor_sleeves.model_dump() == policy(methods[s.id])
        assert c.production_portfolio.rank_exit_buffer == 1
        with pytest.raises(ValueError, match="research.only"):
            strategy_contracts([s.id])
    assert set(methods) <= {s.id for s, _, _ in specs}
    monkeypatch.setattr(ide, "WORKFLOW", ide.PortfolioWorkflow.RUN_PREFERRED)
    assert [s.id for s, _, _ in ide._validated_specs()[2]] == ["multi_source_resilient"]
    assert strategy_contracts(["@formal"])[0]["catalog_strategy_id"] == "multi_source_resilient"
