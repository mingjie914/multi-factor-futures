from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


class _StaticRiskModel:
    def __init__(self, annual_covariance: pd.DataFrame):
        self.annual_covariance = annual_covariance

    def covariance(self, date, universe):
        del date
        return self.annual_covariance.reindex(
            index=universe, columns=universe
        ) / 252.0


def _optimizer(**overrides):
    from optimization.hierarchical_asset_risk_parity import (
        HierarchicalAssetRiskParityOptimizer,
    )

    params = {
        "target_volatility": 0.0,
        "max_leverage": 2.0,
        "covariance_shrinkage": 0.0,
    }
    params.update(overrides)
    return HierarchicalAssetRiskParityOptimizer(**params)


def _run(optimizer, forecasts, annual_covariance, constraints=None, previous=None):
    names = pd.Index(forecasts)
    return optimizer.optimize(
        pd.Series(forecasts, dtype=float),
        _StaticRiskModel(
            pd.DataFrame(annual_covariance, index=names, columns=names)
        ),
        pd.Series(previous or {}, dtype=float),
        list(constraints or []),
        None,
        pd.Timestamp("2025-01-02"),
        names,
    )


def test_taxonomy_exposes_three_level_asset_hierarchy():
    from core.sectors import hierarchy_for

    assert hierarchy_for("IF") == ("stock", "stock")
    assert hierarchy_for("TF") == ("bond", "bond")
    assert hierarchy_for("RB") == ("commodity", "ferrous")
    assert hierarchy_for("LC") == ("commodity", "nonferrous")
    assert hierarchy_for("UR") == ("commodity", "energy")


def test_leaf_signal_divides_by_volatility_exactly_once():
    covariance = np.diag([0.20**2, 0.10**2])
    weights = _run(_optimizer(), {"RB": 2.0, "I": 1.0}, covariance)

    assert weights.abs().sum() == pytest.approx(1.0)
    assert weights["RB"] == pytest.approx(0.5)
    assert weights["I"] == pytest.approx(0.5)


def test_hierarchical_optimizer_rejects_missing_forecast():
    optimizer = _optimizer()
    names = pd.Index(["RB", "I"])
    with pytest.raises(ValueError, match="expected returns"):
        optimizer.optimize(
            pd.Series({"RB": 1.0}),
            _StaticRiskModel(pd.DataFrame(np.eye(2), index=names, columns=names)),
            pd.Series(dtype=float),
            [],
            None,
            pd.Timestamp("2025-01-02"),
            names,
        )


def test_asset_selector_rejects_missing_forecast():
    from optimization.asset_selection import SectorForecastSelector

    with pytest.raises(ValueError, match="missing or non-finite"):
        SectorForecastSelector(mode="hard_top_n").apply(
            pd.Series({"RB": 1.0, "I": np.nan})
        )


def test_commodity_sector_layer_uses_covariance_risk_parity():
    covariance = np.diag([0.20**2, 0.10**2])
    weights = _run(_optimizer(), {"RB": 1.0, "CU": 1.0}, covariance)

    assert weights["RB"] == pytest.approx(1.0 / 3.0, abs=5e-4)
    assert weights["CU"] == pytest.approx(2.0 / 3.0, abs=5e-4)


def test_top_asset_layer_equalizes_risk_contributions():
    volatility = np.array([0.20, 0.05, 0.10])
    forecasts = {"IF": 1.0, "T": 1.0, "RB": 1.0}
    weights = _run(_optimizer(), forecasts, np.diag(volatility**2))
    expected = (1.0 / volatility) / (1.0 / volatility).sum()

    np.testing.assert_allclose(weights.to_numpy(), expected, atol=5e-4)
    contributions = weights.to_numpy() * (
        np.diag(volatility**2) @ weights.to_numpy()
    )
    np.testing.assert_allclose(
        contributions / contributions.sum(),
        np.ones(3) / 3.0,
        atol=5e-4,
    )


def test_target_volatility_scales_exactly_and_respects_leverage_cap():
    covariance = np.array([[0.20**2]])
    target_ten = _run(
        _optimizer(target_volatility=0.10), {"RB": 1.0}, covariance
    )
    capped = _run(
        _optimizer(target_volatility=0.50, max_leverage=2.0),
        {"RB": 1.0},
        covariance,
    )

    assert target_ten["RB"] == pytest.approx(0.5)
    assert target_ten["RB"] * 0.20 == pytest.approx(0.10)
    assert capped["RB"] == pytest.approx(2.0)


def test_hard_limits_reduce_risk_without_redistributing_exposure():
    from optimization.constraints import LeverageConstraint, PositionLimitConstraint

    covariance = np.diag([0.10**2, 0.10**2])
    weights = _run(
        _optimizer(target_volatility=0.20),
        {"RB": 1.0, "I": 1.0},
        covariance,
        constraints=[
            PositionLimitConstraint(lower=-0.4, upper=0.4),
            LeverageConstraint(limit=0.6),
        ],
    )

    assert weights.abs().sum() <= 0.6 + 1e-12
    assert weights.abs().max() <= 0.4 + 1e-12
    assert weights["RB"] == pytest.approx(weights["I"])


def test_hard_limits_take_precedence_when_previous_weights_are_invalid():
    from optimization.constraints import (
        LeverageConstraint,
        NetExposureConstraint,
        PositionLimitConstraint,
        SectorExposureConstraint,
        TurnoverConstraint,
    )

    names = ["RB", "I", "IF"]
    covariance = np.diag([0.10**2] * len(names))
    weights = _run(
        _optimizer(),
        dict.fromkeys(names, 1.0),
        covariance,
        previous=dict.fromkeys(names, 0.8),
        constraints=[
            PositionLimitConstraint(lower=-0.25, upper=0.25),
            SectorExposureConstraint(limit=0.30),
            NetExposureConstraint(lower=-0.20, upper=0.20),
            LeverageConstraint(limit=0.50),
            TurnoverConstraint(limit=0.01),
        ],
    )

    assert weights.abs().max() <= 0.25 + 1e-12
    assert weights[["RB", "I"]].sum() <= 0.30 + 1e-12
    assert weights.abs().sum() <= 0.50 + 1e-12
    assert weights.sum() <= 0.20 + 1e-12


def test_unsupported_constraint_cannot_be_silently_ignored():
    from optimization.constraints import WeightSumConstraint

    with pytest.raises(ValueError, match="does not support constraints: weight_sum"):
        _run(
            _optimizer(),
            {"RB": 1.0},
            np.array([[0.10**2]]),
            constraints=[WeightSumConstraint(target=1.0)],
        )


def test_default_config_enables_three_layer_optimizer_at_ten_percent():
    from core.config import FrameworkConfig, load_config

    config = load_config("config/default.yaml")
    assert config.optimization.type == "hierarchical_asset_risk_parity"
    assert (
        config.optimization.hierarchical_asset_risk_parity.target_volatility
        == pytest.approx(0.10)
    )
    assert config.meta_optimizer.target_volatility == pytest.approx(0.10)
    assert (
        FrameworkConfig().optimization.type
        == "hierarchical_asset_risk_parity"
    )
    turnover = [
        item
        for item in config.optimization.constraints
        if item.get("type") == "turnover"
    ]
    assert turnover == [{"type": "turnover", "limit": 0.30}]


def test_runner_builds_three_layer_optimizer_from_config():
    from core.config import load_config
    from optimization.hierarchical_asset_risk_parity import (
        HierarchicalAssetRiskParityOptimizer,
    )
    from pipeline.runner import PipelineRunner

    runner = PipelineRunner.__new__(PipelineRunner)
    runner.config = load_config("config/default.yaml")
    runner._build_optimization_layer()

    assert isinstance(runner.optimizer, HierarchicalAssetRiskParityOptimizer)
    assert runner.optimizer.target_volatility == pytest.approx(0.10)


def test_switching_to_mean_variance_does_not_consume_hierarchy_settings():
    from core.config import load_config
    from optimization.mean_variance import MeanVarianceOptimizer
    from pipeline.runner import PipelineRunner

    runner = PipelineRunner.__new__(PipelineRunner)
    runner.config = load_config("config/default.yaml")
    runner.config.optimization.type = "mean_variance"
    runner.config.optimization.risk_aversion = 7.0
    runner.config.optimization.cost_penalty = 0.25
    runner._build_optimization_layer()

    assert isinstance(runner.optimizer, MeanVarianceOptimizer)
    assert runner.optimizer.risk_aversion == pytest.approx(7.0)
    assert runner.optimizer.cost_penalty == pytest.approx(0.25)
    assert runner.optimizer.deployment_status == "research_only"


def test_obsolete_sector_optimizer_is_not_registered():
    import optimization  # noqa: F401
    from core.registry import list_registered
    from optimization.hierarchical_asset_risk_parity import (
        HierarchicalAssetRiskParityOptimizer,
    )

    optimizers = list_registered("optimizer")["optimizer"]
    assert "hierarchical_sector" not in optimizers
    assert (
        HierarchicalAssetRiskParityOptimizer.deployment_status
        == "formal_default"
    )


def test_hierarchical_optimizer_rejects_invalid_risk_budgets():
    from optimization.hierarchical_asset_risk_parity import (
        HierarchicalAssetRiskParityOptimizer,
    )

    with pytest.raises(ValueError, match="finite and non-negative"):
        HierarchicalAssetRiskParityOptimizer(asset_class_budgets={"stock": -0.1})
    with pytest.raises(ValueError, match="must be finite"):
        HierarchicalAssetRiskParityOptimizer(target_volatility=float("nan"))


def test_hierarchical_optimizer_rejects_invalid_covariance():
    from optimization.hierarchical_asset_risk_parity import (
        HierarchicalAssetRiskParityOptimizer,
    )

    optimizer = HierarchicalAssetRiskParityOptimizer()
    with pytest.raises(ValueError, match="square"):
        optimizer._ensure_psd(np.ones((2, 3)))
    with pytest.raises(ValueError, match="finite"):
        optimizer._ensure_psd(np.array([[1.0, np.nan], [np.nan, 1.0]]))
