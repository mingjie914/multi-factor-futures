import json

import numpy as np
import pandas as pd
import pytest

from backtest.engine import BacktestResult
from backtest.research_ledger import (
    MissingActiveReturnError,
    ResearchLedgerError,
    ResearchReturnLedger,
    align_transition_weights,
    close_marked_step,
    default_research_ledger_metadata,
)


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
        signals_history=[],
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


def test_research_ledger_rejects_broken_nav_identity():
    ledger = _ledger()
    ledger.daily.loc[ledger.daily.index[-1], "nav_after"] += 0.01
    with pytest.raises(ResearchLedgerError, match="NAV identity"):
        ledger.validate()


def test_backtester_emits_close_marked_ledger_and_uses_drifted_weights(monkeypatch):
    from backtest.engine import Backtester

    class CostModel:
        def estimate_cost(self, target, current, date):
            del date
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
    monkeypatch.setattr(
        backtester,
        "_refit_alpha",
        lambda *args, **kwargs: (None, 0),
    )
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
        factor_names=[],
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
    assert result.turnover.loc[dates[2]] == pytest.approx(0.10 / 0.99)
    assert result.turnover.loc[dates[-1]] == pytest.approx(0.0)
    assert result.costs.sum() == pytest.approx(0.01 + (0.10 / 0.99) * 0.01)
