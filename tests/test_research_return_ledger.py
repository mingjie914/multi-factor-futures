import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from backtest.engine import BacktestResult, MultiPortfolioResult
from backtest.metrics import compute_annual_return, compute_split_metrics
from backtest.research_ledger import (
    MissingActiveReturnError,
    ResearchLedgerError,
    ResearchReturnLedger,
    align_transition_weights,
    build_close_marked_ledger,
    close_marked_step,
    default_research_ledger_metadata,
)
from data.market_quality import CloseDataQualityError, prepare_close_data


def test_build_close_marked_ledger_applies_target_on_next_bar():
    dates = pd.date_range("2025-01-02", periods=3, freq="B")
    target_weights = pd.DataFrame(
        {"A": [0.6, 0.2, 0.9], "B": [0.4, 0.8, 0.1]}, index=dates
    )
    asset_returns = pd.DataFrame(
        {"A": [np.nan, 0.10, -0.20], "B": [np.nan, -0.05, 0.30]},
        index=dates,
    )

    ledger = build_close_marked_ledger(target_weights, asset_returns)

    assert ledger.effective_weights.loc[dates[0]].abs().sum() == pytest.approx(0.0)
    assert ledger.effective_weights.loc[dates[1]].to_dict() == pytest.approx(
        {"A": 0.6, "B": 0.4}
    )
    assert ledger.daily.loc[dates[1], "gross_return"] == pytest.approx(
        0.6 * 0.10 + 0.4 * -0.05
    )


def test_build_close_marked_ledger_turnover_uses_drifted_exposure():
    dates = pd.date_range("2025-01-02", periods=3, freq="B")
    target_weights = pd.DataFrame(
        {"A": [0.5, 0.5, 0.5], "B": [0.5, 0.5, 0.5]}, index=dates
    )
    asset_returns = pd.DataFrame(
        {"A": [np.nan, 0.10, 0.0], "B": [np.nan, -0.10, 0.0]},
        index=dates,
    )

    ledger = build_close_marked_ledger(target_weights, asset_returns)

    # After T+1, the fixed target has drifted to (0.55, 0.45).  The decision
    # made at that close rebalances back to (0.50, 0.50), trading 0.10 in
    # full L1 notional for the next holding bar.
    assert ledger.daily.loc[dates[1], "decision_turnover"] == pytest.approx(0.10)
    assert ledger.daily.loc[dates[2], "turnover"] == pytest.approx(0.10)
    assert ledger.daily.loc[dates[2], "half_turnover"] == pytest.approx(0.05)


def test_ledger_validation_rejects_mislabeled_standard_turnover():
    dates = pd.date_range("2025-01-02", periods=2, freq="B")
    ledger = build_close_marked_ledger(
        pd.DataFrame({"A": [1.0, 1.0]}, index=dates),
        pd.DataFrame({"A": [np.nan, 0.0]}, index=dates),
    )

    ledger.daily.loc[dates[1], "turnover"] = 0.0
    with pytest.raises(ResearchLedgerError, match="executed traded notional"):
        ledger.validate()


def test_build_close_marked_ledger_fails_closed_for_missing_active_return():
    dates = pd.date_range("2025-01-02", periods=2, freq="B")
    target_weights = pd.DataFrame(
        {"A": [0.5, 0.5], "B": [0.5, 0.5]}, index=dates
    )
    asset_returns = pd.DataFrame(
        {"A": [np.nan, np.nan], "B": [np.nan, 0.10]}, index=dates
    )

    with pytest.raises(MissingActiveReturnError, match="A"):
        build_close_marked_ledger(target_weights, asset_returns)


def test_build_close_marked_ledger_rejects_missing_daily_date():
    dates = pd.date_range("2025-01-02", periods=3, freq="B")
    targets = pd.DataFrame({"A": 1.0}, index=dates)
    returns = pd.DataFrame({"A": [np.nan, 0.0]}, index=dates.delete(1))

    with pytest.raises(ResearchLedgerError, match="identical daily dates"):
        build_close_marked_ledger(targets, returns)


def test_build_close_marked_ledger_rejects_nonfinite_targets_and_infinite_returns():
    dates = pd.date_range("2025-01-02", periods=2, freq="B")
    returns = pd.DataFrame({"A": [np.nan, 0.0]}, index=dates)

    with pytest.raises(ResearchLedgerError, match="target weights contain"):
        build_close_marked_ledger(
            pd.DataFrame({"A": [np.nan, 1.0]}, index=dates), returns
        )
    with pytest.raises(ResearchLedgerError, match="asset returns contain infinity"):
        build_close_marked_ledger(
            pd.DataFrame({"A": [0.0, 0.0]}, index=dates),
            pd.DataFrame({"A": [np.inf, 0.0]}, index=dates),
        )


def test_audited_nontrading_close_preserves_move_without_lookahead():
    dates = pd.date_range("2025-01-02", periods=4, freq="B")
    close = pd.DataFrame({"A": [100.0, np.nan, 80.0, 88.0]}, index=dates)

    returns, tradable = prepare_close_data(
        close, {"A": [dates[1].strftime("%Y-%m-%d")]}
    )

    assert np.isnan(returns.loc[dates[0], "A"])
    assert returns.loc[dates[1], "A"] == pytest.approx(0.0)
    assert returns.loc[dates[2], "A"] == pytest.approx(-0.20)
    assert returns.loc[dates[3], "A"] == pytest.approx(0.10)
    assert not bool(tradable.loc[dates[1], "A"])


def test_unapproved_post_listing_close_gap_fails_closed():
    dates = pd.date_range("2025-01-02", periods=3, freq="B")
    close = pd.DataFrame({"SC": [100.0, np.nan, 101.0]}, index=dates)

    with pytest.raises(CloseDataQualityError, match="SC@2025-01-03"):
        prepare_close_data(close)


def test_untradable_close_freezes_drifted_weight_and_blocks_turnover():
    dates = pd.date_range("2025-01-02", periods=5, freq="B")
    targets = pd.DataFrame({"A": [1.0, 0.0, 0.0, 0.0, 0.0]}, index=dates)
    close = pd.DataFrame(
        {"A": [100.0, 100.0, np.nan, 80.0, 80.0]}, index=dates
    )

    returns, tradable = prepare_close_data(
        close, {"A": [dates[2].strftime("%Y-%m-%d")]}
    )
    ledger = build_close_marked_ledger(
        targets,
        returns,
        decision_tradable=tradable,
    )

    # The position is already held before the unavailable row.  Neither the
    # pre-halt nor halted close can close it, so it bears the reopening jump.
    assert ledger.effective_weights.loc[dates[3], "A"] == pytest.approx(1.0)
    assert ledger.daily.loc[dates[1], "decision_turnover"] == pytest.approx(0.0)
    assert ledger.daily.loc[dates[2], "decision_turnover"] == pytest.approx(0.0)
    assert ledger.daily.loc[dates[1], "blocked_target_notional"] == pytest.approx(1.0)
    assert ledger.daily.loc[dates[2], "blocked_target_notional"] == pytest.approx(1.0)
    assert ledger.daily.loc[dates[3], "gross_return"] == pytest.approx(-0.20)


def test_target_is_not_opened_on_following_untradable_close():
    dates = pd.date_range("2025-01-02", periods=4, freq="B")
    targets = pd.DataFrame({"A": 1.0}, index=dates)
    returns = pd.DataFrame({"A": [np.nan, 0.0, -0.20, 0.0]}, index=dates)
    tradable = pd.DataFrame({"A": [True, False, True, True]}, index=dates)

    ledger = build_close_marked_ledger(
        targets, returns, decision_tradable=tradable
    )

    assert ledger.effective_weights.loc[dates[1], "A"] == pytest.approx(0.0)
    assert ledger.effective_weights.loc[dates[2], "A"] == pytest.approx(0.0)
    assert ledger.effective_weights.loc[dates[3], "A"] == pytest.approx(1.0)
    assert ledger.daily.loc[dates[2], "gross_return"] == pytest.approx(0.0)


def test_build_close_marked_ledger_records_roll_legs_and_full_notional():
    dates = pd.date_range("2025-01-02", periods=3, freq="B")
    target_weights = pd.DataFrame({"A": [1.0, 1.0, 1.0]}, index=dates)
    asset_returns = pd.DataFrame({"A": [np.nan, 0.0, 0.02]}, index=dates)
    contract_schedule = pd.DataFrame(
        {"A": ["A2401", "A2401", "A2402"]}, index=dates
    )

    ledger = build_close_marked_ledger(
        target_weights,
        asset_returns,
        contract_schedule=contract_schedule,
        trade_cost_rate=0.0,
    )

    # The root target remains 1.0, but the old contract must be closed and
    # the new contract opened: 1.0 + 1.0 of full traded notional.  The roll
    # is identified on the decision close and executed on the next bar.
    assert ledger.daily.loc[dates[1], "roll_turnover"] == pytest.approx(2.0)
    assert ledger.daily.loc[dates[2], "executed_traded_notional"] == pytest.approx(
        2.0
    )
    assert ledger.daily.loc[dates[2], "executed_roll_turnover"] == pytest.approx(
        2.0
    )


def test_untradable_close_delays_roll_until_contract_can_trade():
    dates = pd.date_range("2025-01-02", periods=5, freq="B")
    targets = pd.DataFrame({"A": 1.0}, index=dates)
    returns = pd.DataFrame({"A": [np.nan, 0.0, 0.0, 0.0, 0.0]}, index=dates)
    schedule = pd.DataFrame(
        {"A": ["A2401", "A2401", "A2402", "A2402", "A2402"]},
        index=dates,
    )
    tradable = pd.DataFrame(
        {"A": [True, True, False, True, True]}, index=dates
    )

    ledger = build_close_marked_ledger(
        targets,
        returns,
        contract_schedule=schedule,
        decision_tradable=tradable,
    )

    assert ledger.daily.loc[dates[1], "roll_turnover"] == pytest.approx(0.0)
    assert ledger.daily.loc[dates[2], "roll_turnover"] == pytest.approx(0.0)
    assert ledger.daily.loc[dates[3], "roll_turnover"] == pytest.approx(2.0)
    assert ledger.daily.loc[dates[4], "executed_traded_notional"] == pytest.approx(2.0)
    assert ledger.metadata["untradable_rollover_policy"] == "delay_until_tradable"


def test_build_close_marked_ledger_does_not_execute_final_target():
    dates = pd.date_range("2025-01-02", periods=3, freq="B")
    target_weights = pd.DataFrame(
        {"A": [1.0, 1.0, 0.0], "B": [0.0, 0.0, 1.0]}, index=dates
    )
    asset_returns = pd.DataFrame(
        {"A": [np.nan, 0.0, 0.0], "B": [np.nan, 0.0, 0.0]}, index=dates
    )

    ledger = build_close_marked_ledger(
        target_weights,
        asset_returns,
        trade_cost_rate=0.0,
    )

    # The B target is dated on the last available close and has no following
    # holding bar, so it must not create a trade or a cost on the final row.
    assert ledger.effective_weights.loc[dates[2], "A"] == pytest.approx(1.0)
    assert ledger.effective_weights.loc[dates[2], "B"] == pytest.approx(0.0)
    assert ledger.daily.loc[dates[2], "decision_turnover"] == pytest.approx(0.0)
    assert ledger.daily.loc[dates[2], "trade_cost"] == pytest.approx(0.0)


def test_close_marked_step_drifts_exposure_without_hidden_daily_rebalance():
    step = close_marked_step(
        pd.Series({"A": 0.5, "B": 0.5}),
        pd.Series({"A": 0.10, "B": -0.10}),
    )

    assert step.gross_return == pytest.approx(0.0)
    assert step.net_return == pytest.approx(0.0)
    assert step.contributions.to_dict() == pytest.approx(
        {"A": 0.05, "B": -0.05}
    )
    assert step.end_weights.to_dict() == pytest.approx(
        {"A": 0.55, "B": 0.45}
    )


def test_close_marked_step_costs_reduce_nav_and_raise_end_exposure():
    step = close_marked_step(
        pd.Series({"A": 0.5}),
        pd.Series({"A": 0.10}),
        trade_cost=0.002,
        holding_cost=0.001,
    )

    assert step.gross_return == pytest.approx(0.05)
    assert step.net_return == pytest.approx(0.047)
    assert step.end_weights["A"] == pytest.approx(0.55 / 1.047)


def test_close_marked_step_fails_closed_for_missing_active_return():
    with pytest.raises(MissingActiveReturnError, match="A"):
        close_marked_step(
            pd.Series({"A": 0.5, "B": 0.0}),
            pd.Series({"A": np.nan, "B": np.nan}),
        )

    inactive = close_marked_step(
        pd.Series({"A": 0.0}), pd.Series({"A": np.nan})
    )
    assert inactive.net_return == 0.0
    assert inactive.asset_returns["A"] == 0.0


def test_transition_alignment_includes_open_and_exit_legs():
    target, current = align_transition_weights(
        pd.Series({"NEW": 0.4}),
        pd.Series({"OLD": 0.3}),
    )

    assert list(target.index) == ["OLD", "NEW"]
    assert target.to_dict() == {"OLD": 0.0, "NEW": 0.4}
    assert current.to_dict() == {"OLD": 0.3, "NEW": 0.0}
    assert (target - current).abs().sum() == pytest.approx(0.7)


def _ledger() -> ResearchReturnLedger:
    dates = pd.date_range("2025-01-02", periods=2, freq="B")
    returns = pd.DataFrame(
        {"A": [0.0, 0.10], "B": [0.0, -0.10]}, index=dates
    )
    weights = pd.DataFrame(
        {"A": [0.0, 0.5], "B": [0.0, 0.5]}, index=dates
    )
    contributions = returns * weights
    gross = contributions.sum(axis=1)
    trade_cost = pd.Series([0.0, 0.001], index=dates)
    holding_cost = pd.Series([0.0, 0.0001], index=dates)
    net = gross - trade_cost - holding_cost
    nav_before = pd.Series([1.0, 1.0], index=dates)
    nav_after = nav_before * (1.0 + net)
    daily = pd.DataFrame(
        {
            "nav_before": nav_before,
            "nav_after": nav_after,
            "gross_return": gross,
            "trade_cost": trade_cost,
            "holding_cost": holding_cost,
            "net_return": net,
            "decision_turnover": [1.0, 0.0],
            "executed_traded_notional": [0.0, 1.0],
            "roll_turnover": [0.0, 0.0],
            "executed_roll_turnover": [0.0, 0.0],
            "gross_exposure": weights.abs().sum(axis=1),
            "net_exposure": weights.sum(axis=1),
            "active_instruments": weights.abs().gt(1e-12).sum(axis=1),
        },
        index=dates,
    )
    return ResearchReturnLedger(
        daily=daily,
        asset_returns=returns,
        effective_weights=weights,
        contributions=contributions,
        metadata=default_research_ledger_metadata(),
    )


def test_research_ledger_validates_accounting_identities_and_saves(tmp_path):
    ledger = _ledger()
    ledger.validate()

    result = BacktestResult(
        nav=ledger.daily["nav_after"],
        weights_history=pd.DataFrame(),
        metrics={},
        research_ledger=ledger,
    )
    result.save(tmp_path)

    assert (tmp_path / "research_return_ledger.csv").is_file()
    assert (tmp_path / "research_asset_returns.csv").is_file()
    assert (tmp_path / "research_effective_weights.csv").is_file()
    assert (tmp_path / "research_return_contributions.csv").is_file()
    metadata = json.loads(
        (tmp_path / "research_return_ledger_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    assert metadata["mark_price_field"] == "close"
    assert metadata["accounting_scope"] == (
        "research_only_not_broker_settlement_or_delivery"
    )
    assert metadata["turnover_cost_policy"] == (
        "diagnostic_only_not_charged"
    )


def test_backtest_result_exports_one_deterministic_target_weight_snapshot(tmp_path):
    dates = pd.to_datetime(["2026-08-13", "2026-08-14"])
    weights = pd.DataFrame(
        {"RB": [0.2, 0.3], "CU": [-0.2, np.nan]}, index=dates
    )
    result = BacktestResult(
        nav=pd.Series([1.0, 1.01], index=dates),
        weights_history=weights,
        metrics={},
    )
    path = tmp_path / "final_target_weights.csv"

    exported = result.export_target_weights(path)
    frame = pd.read_csv(exported)

    assert exported == str(path)
    assert frame["decision_date"].unique().tolist() == ["2026-08-14"]
    assert frame.set_index("ticker")["target_weight"].to_dict() == {
        "CU": 0.0,
        "RB": 0.3,
    }
    assert frame["execution_timing"].unique().tolist() == [
        "following_trading_session"
    ]


def test_research_ledger_rejects_broken_nav_identity():
    ledger = _ledger()
    ledger.daily.loc[ledger.daily.index[-1], "nav_after"] += 0.01
    with pytest.raises(ResearchLedgerError, match="NAV identity"):
        ledger.validate()


def test_research_ledger_rejects_discontinuous_nav_chain():
    ledger = _ledger()
    ledger.daily.loc[ledger.daily.index[-1], "nav_before"] += 0.01
    net = ledger.daily.loc[ledger.daily.index[-1], "net_return"]
    ledger.daily.loc[ledger.daily.index[-1], "nav_after"] = (
        ledger.daily.loc[ledger.daily.index[-1], "nav_before"] * (1.0 + net)
    )

    with pytest.raises(ResearchLedgerError, match="NAV chain"):
        ledger.validate()


def test_research_ledger_rejects_inconsistent_reported_exposure():
    ledger = _ledger()
    ledger.daily.loc[ledger.daily.index[-1], "gross_exposure"] += 0.1

    with pytest.raises(ResearchLedgerError, match="gross exposure"):
        ledger.validate()


def test_fixed_subportfolio_combine_keeps_not_started_capital_as_cash():
    dates = pd.date_range("2025-01-02", periods=3, freq="B")

    def item(name, nav, weight):
        config = SimpleNamespace(
            name=name,
            factors=[],
            rebalance_freq="daily",
            holding_period=1,
            capital_weight=weight,
        )
        result = BacktestResult(
            nav=nav,
            weights_history=pd.DataFrame(),
            metrics={},
        )
        return {"config": config, "result": result}

    combined = MultiPortfolioResult.combine(
        [
            item("early", pd.Series(100.0, index=dates), 0.5),
            item("late", pd.Series(100.0, index=dates[1:]), 0.5),
        ],
        total_capital=100.0,
    )

    np.testing.assert_allclose(combined.combined_result.nav.to_numpy(), 100.0)


def test_fixed_subportfolio_combine_rejects_trailing_nav_gap():
    dates = pd.date_range("2025-01-02", periods=3, freq="B")

    def item(name, nav):
        return {
            "config": SimpleNamespace(
                name=name,
                factors=[],
                rebalance_freq="daily",
                holding_period=1,
                capital_weight=0.5,
            ),
            "result": BacktestResult(
                nav=nav,
                weights_history=pd.DataFrame(),
                metrics={},
            ),
        }

    with pytest.raises(ValueError, match="internal or trailing gap"):
        MultiPortfolioResult.combine(
            [
                item("complete", pd.Series(100.0, index=dates)),
                item("truncated", pd.Series(100.0, index=dates[:-1])),
            ],
            total_capital=100.0,
        )


def test_annual_return_uses_nav_intervals_and_split_excludes_boundary_return():
    assert compute_annual_return(pd.Series([1.0, 2.0]), periods_per_year=1) == pytest.approx(1.0)

    dates = pd.date_range("2025-01-02", periods=8, freq="B")
    nav = pd.Series([1.0] * 6 + [2.0, 2.0], index=dates)
    metrics = compute_split_metrics(
        nav,
        train_ratio=0.75,
        periods_per_year=252,
        minimum_train_bars=1,
        minimum_test_bars=1,
    )

    assert metrics["test"]["total_return"] == pytest.approx(0.0)


def test_backtester_emits_close_marked_ledger_and_uses_drifted_weights(monkeypatch):
    from backtest.engine import Backtester

    class CostModel:
        dates = []

        def estimate_cost(self, target, current, date):
            self.dates.append(pd.Timestamp(date))
            return float((target - current).abs().sum()) * 0.01

    dates = pd.date_range("2025-01-02", periods=4, freq="B")
    universe = pd.Index(["A", "B"])
    daily_returns = pd.DataFrame(
        {"A": [np.nan, 0.10, 0.0, 0.0], "B": [np.nan, -0.10, 0.0, 0.0]},
        index=dates,
    )
    backtester = Backtester(cost_model=CostModel())
    monkeypatch.setattr(
        backtester,
        "_prepare_backtest_data",
        lambda *args, **kwargs: (
            pd.DatetimeIndex([dates[0], dates[2], dates[-1]]),
            universe,
            dates,
            {},
            pd.DataFrame(0.0, index=dates, columns=universe),
            daily_returns,
        ),
    )
    def mark_model_ready(*args, **kwargs):
        del args, kwargs
        backtester._last_fit_factors = {"f"}
        return dates[0], 0

    monkeypatch.setattr(backtester, "_refit_alpha", mark_model_ready)
    monkeypatch.setattr(
        backtester,
        "_predict_returns",
        lambda *args, **kwargs: pd.Series(0.0, index=universe),
    )
    monkeypatch.setattr(
        backtester,
        "_optimize_period",
        lambda *args, **kwargs: pd.Series({"A": 0.5, "B": 0.5}),
    )

    result = backtester.run(
        data_manager=object(),
        factor_engine=object(),
        processor=object(),
        factor_names=["f"],
        universe_schedule={dates[0]: universe},
        rebalance_dates=pd.DatetimeIndex([dates[0], dates[2], dates[-1]]),
        alpha_model=object(),
        risk_model=object(),
        optimizer=object(),
        constraints=[],
    )

    result.research_ledger.validate()
    weights = result.research_ledger.effective_weights
    assert weights.loc[dates[1]].to_dict() == pytest.approx({"A": 0.5, "B": 0.5})
    assert weights.loc[dates[2]].to_dict() == pytest.approx(
        {"A": 0.55 / 0.99, "B": 0.45 / 0.99}
    )
    assert result.research_ledger.daily.loc[dates[1], "gross_return"] == pytest.approx(0.0)
    assert result.research_ledger.daily.loc[dates[1], "trade_cost"] == pytest.approx(0.01)
    assert result.decision_turnover.loc[dates[2]] == pytest.approx(0.10 / 0.99)
    assert result.turnover.loc[dates[-1]] == pytest.approx(0.10 / 0.99)
    assert result.costs.sum() == pytest.approx(0.01 + (0.10 / 0.99) * 0.01)
    assert result.weights_history.index[-1] == dates[-1]
    assert result.research_ledger.daily.loc[
        dates[1], "executed_traded_notional"
    ] == pytest.approx(1.0)
    assert result.research_ledger.daily.loc[
        dates[3], "executed_traded_notional"
    ] == pytest.approx(0.10 / 0.99)
    assert CostModel.dates == [dates[1], dates[3]]


def test_backtester_close_decision_sees_same_close_drifted_weights(monkeypatch):
    from backtest.engine import Backtester

    dates = pd.date_range("2025-01-02", periods=3, freq="B")
    universe = pd.Index(["A", "B"])
    daily_returns = pd.DataFrame(
        {"A": [np.nan, 0.10, 0.0], "B": [np.nan, -0.10, 0.0]},
        index=dates,
    )
    backtester = Backtester(cost_model=None)
    monkeypatch.setattr(
        backtester,
        "_prepare_backtest_data",
        lambda *args, **kwargs: (
            dates,
            universe,
            dates,
            {},
            pd.DataFrame(0.0, index=dates, columns=universe),
            daily_returns,
        ),
    )

    def mark_model_ready(*args, **kwargs):
        del args, kwargs
        backtester._last_fit_factors = {"f"}
        return dates[0], 0

    observed_current = {}

    def capture_optimize(
        predicted,
        risk_model,
        current_weights,
        constraints,
        optimizer,
        date,
        selected_universe,
        realized_vol,
        current_drawdown=0.0,
    ):
        del (
            predicted,
            risk_model,
            constraints,
            optimizer,
            selected_universe,
            realized_vol,
            current_drawdown,
        )
        observed_current[pd.Timestamp(date)] = current_weights.copy()
        return pd.Series({"A": 0.5, "B": 0.5})

    monkeypatch.setattr(backtester, "_refit_alpha", mark_model_ready)
    monkeypatch.setattr(
        backtester,
        "_predict_returns",
        lambda *args, **kwargs: pd.Series(0.0, index=universe),
    )
    monkeypatch.setattr(backtester, "_optimize_period", capture_optimize)

    backtester.run(
        data_manager=object(),
        factor_engine=object(),
        processor=object(),
        factor_names=["f"],
        universe_schedule={dates[0]: universe},
        rebalance_dates=dates,
        alpha_model=object(),
        risk_model=object(),
        optimizer=object(),
        constraints=[],
    )

    assert observed_current[dates[1]].to_dict() == pytest.approx(
        {"A": 0.55, "B": 0.45}
    )


def test_backtester_does_not_apply_target_on_following_untradable_close(
    monkeypatch,
):
    from backtest.engine import Backtester

    dates = pd.date_range("2025-01-02", periods=4, freq="B")
    universe = pd.Index(["A"])
    daily_returns = pd.DataFrame(
        {"A": [np.nan, 0.0, -0.20, 0.0]}, index=dates
    )
    backtester = Backtester(cost_model=None)

    def prepare(*args, **kwargs):
        del args, kwargs
        backtester._close_tradable = pd.DataFrame(
            {"A": [True, False, True, True]}, index=dates
        )
        return (
            dates,
            universe,
            dates,
            {},
            pd.DataFrame(0.0, index=dates, columns=universe),
            daily_returns,
        )

    monkeypatch.setattr(backtester, "_prepare_backtest_data", prepare)

    def mark_model_ready(*args, **kwargs):
        del args, kwargs
        backtester._last_fit_factors = {"f"}
        return dates[0], 0

    monkeypatch.setattr(backtester, "_refit_alpha", mark_model_ready)
    monkeypatch.setattr(
        backtester,
        "_predict_returns",
        lambda *args, **kwargs: pd.Series(0.0, index=universe),
    )
    monkeypatch.setattr(
        backtester,
        "_optimize_period",
        lambda *args, **kwargs: pd.Series({"A": 1.0}),
    )

    result = backtester.run(
        data_manager=object(),
        factor_engine=object(),
        processor=object(),
        factor_names=["f"],
        universe_schedule={dates[0]: universe},
        rebalance_dates=dates,
        alpha_model=object(),
        risk_model=object(),
        optimizer=object(),
        constraints=[],
    )

    effective = result.research_ledger.effective_weights["A"]
    assert effective.loc[dates[1]] == pytest.approx(0.0)
    assert effective.loc[dates[2]] == pytest.approx(0.0)
    assert effective.loc[dates[3]] == pytest.approx(1.0)
    assert result.research_ledger.daily.loc[
        dates[2], "gross_return"
    ] == pytest.approx(0.0)


def test_backtester_records_non_rebalance_roll_on_concrete_contracts(monkeypatch):
    from backtest.engine import Backtester

    class CostModel:
        def estimate_cost(self, target, current, date):
            del date
            return float((target - current).abs().sum()) * 0.01

    dates = pd.date_range("2025-01-02", periods=4, freq="B")
    universe = pd.Index(["A"])
    daily_returns = pd.DataFrame({"A": [np.nan, 0.0, 0.0, 0.0]}, index=dates)
    contract_schedule = pd.DataFrame(
        {"A": ["A2501", "A2501", "A2505", "A2505"]}, index=dates
    )
    backtester = Backtester(cost_model=CostModel())

    def prepare(*args, **kwargs):
        del args, kwargs
        backtester._contract_schedule = contract_schedule
        backtester._close_tradable = pd.DataFrame(True, index=dates, columns=universe)
        return (
            pd.DatetimeIndex([dates[0], dates[-1]]),
            universe,
            dates,
            {},
            pd.DataFrame(0.0, index=dates, columns=universe),
            daily_returns,
        )

    monkeypatch.setattr(backtester, "_prepare_backtest_data", prepare)
    def mark_model_ready(*args, **kwargs):
        del args, kwargs
        backtester._last_fit_factors = {"f"}
        return dates[0], 0

    monkeypatch.setattr(backtester, "_refit_alpha", mark_model_ready)
    monkeypatch.setattr(
        backtester,
        "_predict_returns",
        lambda *args, **kwargs: pd.Series(0.0, index=universe),
    )
    monkeypatch.setattr(
        backtester,
        "_optimize_period",
        lambda *args, **kwargs: pd.Series({"A": 1.0}),
    )

    result = backtester.run(
        data_manager=object(),
        factor_engine=object(),
        processor=object(),
        factor_names=["f"],
        universe_schedule={dates[0]: universe},
        rebalance_dates=pd.DatetimeIndex([dates[0], dates[-1]]),
        alpha_model=object(),
        risk_model=object(),
        optimizer=object(),
        constraints=[],
    )

    assert result.decision_turnover.loc[dates[0]] == pytest.approx(1.0)
    assert result.turnover.loc[dates[1]] == pytest.approx(1.0)
    expected_roll = 2.0 / 0.99
    assert result.decision_turnover.loc[dates[1]] == pytest.approx(expected_roll)
    assert result.turnover.loc[dates[2]] == pytest.approx(expected_roll)
    assert result.research_ledger.daily.loc[
        dates[1], "roll_turnover"
    ] == pytest.approx(expected_roll)
    assert result.costs.loc[dates[1]] == pytest.approx(0.01)
    assert result.costs.loc[dates[2]] == pytest.approx(expected_roll * 0.01)
    assert result.research_ledger.daily.loc[
        dates[2], "executed_roll_turnover"
    ] == pytest.approx(expected_roll)
    assert result.metrics["total_roll_turnover"] == pytest.approx(expected_roll)
    assert result.research_ledger.metadata["contract_schedule_policy"] == (
        "point_in_time_schedule"
    )
