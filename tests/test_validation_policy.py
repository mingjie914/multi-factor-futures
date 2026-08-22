from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
import pytest


def test_default_validation_policy_is_hashable_and_scorecard_cannot_self_calibrate():
    from core.config import load_config
    from research.validation import (
        policy_dict,
        validate_policy,
        validation_policy_sha256,
    )

    policy = load_config("config/default.yaml").validation_policy
    assert validate_policy(policy)["discovery_q"] == pytest.approx(0.10)
    assert "macro_trend" in policy.dual_track_families
    assert len(validation_policy_sha256(policy)) == 64

    invalid = copy.deepcopy(policy_dict(policy))
    invalid["scorecard"]["enforced"] = True
    with pytest.raises(ValueError, match="must be calibrated"):
        validate_policy(invalid)


def test_macro_sensitivity_family_uses_dual_track_governance():
    from research.governance import factor_family

    assert factor_family("macro_beta_pmi_new_orders_36m") == "macro_trend"


def test_hierarchical_fdr_uses_selection_adjusted_local_level_and_reports_fwer():
    from research.validation import apply_hierarchical_fdr

    factors = {
        "signal": [{"p_value": 0.001}, {"p_value": 0.020}],
        "noise": [{"p_value": 0.80}],
    }
    audit = apply_hierarchical_fdr(factors, q=0.10, fwer_alpha=0.05)

    assert audit["factor_family_count"] == 2
    assert audit["selected_factor_count"] == 1
    assert audit["local_alpha"] == pytest.approx(0.05)
    assert all(row["hierarchical_fdr_significant"] for row in factors["signal"])
    assert factors["signal"][0]["evidence_level"] == "FWER"
    assert factors["signal"][1]["evidence_level"] == "FDR"
    assert not factors["noise"][0]["hierarchical_fdr_significant"]


def test_intraday_turnover_is_aggregated_by_trading_day():
    from testing.turnover import TurnoverTest

    index = pd.to_datetime([
        "2026-01-05 09:01", "2026-01-05 09:02", "2026-01-05 09:03",
        "2026-01-06 09:01", "2026-01-06 09:02", "2026-01-06 09:03",
    ])
    factor = pd.DataFrame(
        [
            [1, 2, 3, 4], [4, 3, 2, 1], [1, 3, 2, 4],
            [4, 2, 3, 1], [1, 2, 4, 3], [3, 4, 1, 2],
        ],
        index=index,
        columns=list("ABCD"),
        dtype=float,
    )
    result = TurnoverTest().run(factor)

    assert len(result.turnover_series) == len(index) - 1
    assert len(result.daily_turnover_series) == 2
    assert result.daily_turnover == pytest.approx(
        result.daily_turnover_series.mean()
    )
    assert result.monthly_turnover == pytest.approx(result.daily_turnover * 21)
    assert sum(result.mean_absolute_weights.values()) == pytest.approx(1.0)


def test_validation_cost_scales_with_turnover_and_roll_is_deferred():
    from optimization.costs import factor_cost_coverage

    low_turnover = factor_cost_coverage(
        gross_annual_alpha=0.001,
        annual_half_turnover=0.1,
        annual_roll_cost=0.00105,
        turnover_cost_rate=0.0002,
        include_roll_cost=False,
        cost_stage="factor_validation",
        safety_margin=1.5,
    )
    high_turnover = factor_cost_coverage(
        gross_annual_alpha=0.001,
        annual_half_turnover=1000.0,
        annual_roll_cost=0.00105,
        turnover_cost_rate=0.0002,
        include_roll_cost=False,
        cost_stage="factor_validation",
        safety_margin=1.5,
    )

    assert low_turnover["annual_trading_cost"] == pytest.approx(0.00004)
    assert high_turnover["annual_trading_cost"] == pytest.approx(0.4)
    assert low_turnover["annual_roll_cost_charged"] == pytest.approx(0.0)
    assert low_turnover["transaction_cost_mode"] == (
        "full_traded_notional_times_rate"
    )
    assert low_turnover["complete"] is True
    assert low_turnover["passes"] is True

    post_screen = factor_cost_coverage(
        gross_annual_alpha=0.01,
        annual_half_turnover=1.0,
        turnover_cost_rate=0.0002,
        annual_roll_cost=0.00105,
        include_roll_cost=True,
        safety_margin=1.5,
    )
    assert post_screen["annual_roll_cost_charged"] == pytest.approx(0.00105)
    assert post_screen["total_annual_cost"] == pytest.approx(0.00145)


def test_calendar_year_robustness_uses_natural_years_and_minimum_history():
    from testing.robustness import CalendarYearRobustnessTest

    rng = np.random.default_rng(20260728)
    dates = pd.DatetimeIndex(np.concatenate([
        pd.date_range(f"{year}-01-02", periods=30, freq="B").to_numpy()
        for year in range(2021, 2026)
    ]))
    columns = [f"A{i:02d}" for i in range(12)]
    factor = pd.DataFrame(
        rng.normal(size=(len(dates), len(columns))), index=dates, columns=columns
    )
    returns = factor * 0.01
    result = CalendarYearRobustnessTest(
        minimum_years=5,
        minimum_days_per_year=20,
        bootstrap_samples=39,
    ).run(factor, returns)

    assert result.valid_years == [2021, 2022, 2023, 2024, 2025]
    assert result.observation_channel is False
    assert result.ic_sign_consistency == pytest.approx(1.0)
    assert result.effect_year_ratio == pytest.approx(1.0)
    assert result.block_bootstrap_ic_ci[0] > 0.0


def test_single_instrument_ts_channel_uses_unique_trading_days_and_conservative_p():
    from workflows.factor_adaptivity import _compute_ic_by_sector

    rng = np.random.default_rng(7)
    dates = pd.date_range("2021-01-04", periods=760, freq="B")
    x = rng.normal(size=len(dates))
    factor = pd.DataFrame({"SI": x}, index=dates)
    returns = pd.DataFrame(
        {"SI": 0.02 * x + rng.normal(scale=0.01, size=len(dates))},
        index=dates,
    )
    record = _compute_ic_by_sector(
        factor,
        returns,
        {"SI": "nonferrous"},
        forward_period=1,
        single_min_trading_days=750,
        single_bootstrap_samples=49,
    )["nonferrous"]

    assert record["n_trading_days"] == 760
    assert record["sufficient_history"] is True
    assert record["observation_channel"] is False
    assert record["p_value"] == max(
        record["hac_p_value"], record["wild_bootstrap_p_value"]
    )


def test_frequency_sample_policy_enforces_daily_and_one_minute_boundaries():
    from core.config import ValidationPolicyConfig
    from research.sample_policy import assess_sample_counts, minimum_training_days

    policy = ValidationPolicyConfig()
    assert assess_sample_counts(
        750, 250, policy=policy, frequency="daily"
    ).sufficient
    assert not assess_sample_counts(
        749, 250, policy=policy, frequency="daily"
    ).sufficient
    assert assess_sample_counts(
        14400, 4800, policy=policy, frequency="1min"
    ).sufficient
    too_short = assess_sample_counts(
        14399, 4800, policy=policy, frequency="1min"
    )
    assert not too_short.sufficient
    assert "insufficient_train_bars" in too_short.reasons
    assert minimum_training_days(policy, "daily") == 750
    assert minimum_training_days(policy, "1min") == 60
    dense_but_too_short = assess_sample_counts(
        14400, 4800, policy=policy, frequency="1min",
        train_days=59, test_days=20,
    )
    assert not dense_but_too_short.sufficient
    assert "insufficient_train_days" in dense_but_too_short.reasons


def test_adaptivity_oos_split_marks_short_data_as_observation():
    from core.config import ValidationPolicyConfig
    from workflows.factor_adaptivity import _compute_oos_consistency

    policy = ValidationPolicyConfig()
    short = pd.Series(
        np.linspace(-0.01, 0.01, 999),
        index=pd.date_range("2020-01-01", periods=999, freq="B"),
    )
    result = _compute_oos_consistency(short, policy=policy, frequency="daily")
    assert result["observation_channel"] is True
    assert result["train_bars"] == 749
    assert result["test_bars"] == 250
    assert "insufficient_train_bars" in result["observation_reasons"]


def test_intraday_single_instrument_gate_uses_frequency_specific_days(monkeypatch):
    import workflows.factor_adaptivity as adaptivity
    from core.config import ValidationPolicyConfig

    observed = []

    def fake_sector(*args, **kwargs):
        observed.append(kwargs["single_min_trading_days"])
        return {}

    monkeypatch.setattr(adaptivity, "_compute_ic_by_sector", fake_sector)
    index = pd.date_range("2024-01-02 09:00", periods=10, freq="min")
    frame = pd.DataFrame({"A": np.arange(10.0)}, index=index)
    adaptivity._analyze_factor_across_periods(
        frame, {1: frame}, [1], {"A": "sector"},
        validation_policy=ValidationPolicyConfig(), frequency="1min",
    )

    assert observed == [60]


def test_taxonomy_change_requires_full_p0_replay():
    from core.sectors import SECTOR_MAP, taxonomy_diff

    previous = dict(SECTOR_MAP)
    previous["SI"] = "other"
    audit = taxonomy_diff(previous)
    assert audit["requires_full_p0_replay"] is True
    assert audit["changes"] == [
        {"instrument": "SI", "previous": "other", "current": "nonferrous"}
    ]


def test_taxonomy_replay_comparison_records_silent_factor_status_changes():
    from research.validation import compare_taxonomy_replay

    previous = {
        "config": {"taxonomy_sha256": "a" * 64},
        "all_results": [{"name": "signal", "best_period": 5, "best_ic": 0.02}],
        "final_factors": ["signal"],
    }
    current = {
        "config": {"taxonomy_sha256": "b" * 64},
        "all_results": [{"name": "signal", "best_period": 0, "best_ic": 0.0}],
        "final_factors": [],
    }
    audit = compare_taxonomy_replay(previous, current)
    assert audit["taxonomy_changed"] is True
    assert audit["removed_final_factors"] == ["signal"]
    assert audit["factor_differences"][0]["current_final"] is False


def test_observation_factor_cap_scales_only_that_factor_contribution():
    from alpha.ols import SectorGroupedOLSModel

    dates = pd.date_range("2024-01-02", periods=40, freq="B")
    values = np.tile(np.asarray([-1.0, 1.0]), (len(dates), 1))
    factor = pd.DataFrame(values, index=dates, columns=["RB", "HC"])
    returns = factor * 0.02
    uncapped = SectorGroupedOLSModel(min_samples_per_sector=1)
    capped = SectorGroupedOLSModel(
        min_samples_per_sector=1, factor_weight_caps={"signal": 0.5}
    )
    uncapped.fit({"signal": factor}, returns)
    capped.fit({"signal": factor}, returns)

    base = uncapped.predict({"signal": factor}, pd.Index(factor.columns), dates[-1])
    limited = capped.predict({"signal": factor}, pd.Index(factor.columns), dates[-1])
    np.testing.assert_allclose(limited.to_numpy(), base.to_numpy() * 0.5)


def test_sector_model_does_not_hide_unexpected_fit_failure(monkeypatch):
    from alpha.ols import SectorGroupedOLSModel

    dates = pd.date_range("2024-01-02", periods=40, freq="B")
    factor = pd.DataFrame(
        np.tile(np.asarray([-1.0, 1.0]), (len(dates), 1)),
        index=dates,
        columns=["RB", "HC"],
    )
    model = SectorGroupedOLSModel(
        min_samples_per_sector=1,
        fallback_to_global=True,
        ridge_alphas=[0.0, 0.1],
    )

    def fail(*args, **kwargs):
        raise RuntimeError("unexpected sector fit failure")

    monkeypatch.setattr(model, "_choose_ridge_alpha", fail)
    with pytest.raises(RuntimeError, match="unexpected sector fit failure"):
        model.fit({"signal": factor}, factor * 0.02)


def test_fold_survival_cannot_claim_promotion_before_new_locked_oos():
    from workflows.walkforward import summarize_factor_fold_survival

    folds = [
        {"factor_oos": {"signal": {
            "same_direction": True,
            "oriented_oos_ic": 0.02,
            "n_observations": 100,
        }}},
        {"factor_oos": {"signal": {
            "same_direction": True,
            "oriented_oos_ic": 0.01,
            "n_observations": 50,
        }}},
    ]
    result = summarize_factor_fold_survival(folds)["signal"]
    assert result["combined_oriented_oos_ic"] == pytest.approx(1.0 / 60.0)
    assert result["combined_oos_same_direction"] is True
    assert result["passes_two_consecutive_fold_gate"] is True
    assert result["observation_transition_ready"] is False
    assert result["requires_positive_new_locked_oos"] is True
    assert result["locked_oos_status"] == "frozen_oos_ends_2026-05-14"
    assert result["simulated_live_status"] == "active_since_2026-05-15"
    assert result["production_approved"] is False


def test_fold_survival_rejects_negative_combined_oos_despite_fold_majority():
    from workflows.walkforward import summarize_factor_fold_survival

    folds = [
        {"factor_oos": {"signal": {
            "same_direction": True,
            "oriented_oos_ic": 0.01,
            "n_observations": 100,
        }}},
        {"factor_oos": {"signal": {
            "same_direction": True,
            "oriented_oos_ic": 0.01,
            "n_observations": 100,
        }}},
        {"factor_oos": {"signal": {
            "same_direction": False,
            "oriented_oos_ic": -0.10,
            "n_observations": 100,
        }}},
    ]
    result = summarize_factor_fold_survival(folds)["signal"]
    assert result["fold_sign_ratio"] == pytest.approx(2.0 / 3.0)
    assert result["combined_oriented_oos_ic"] < 0.0
    assert result["passes_fold_gate"] is False


def test_threshold_sensitivity_reports_factor_names_and_jaccard():
    from types import SimpleNamespace

    from workflows.research import _build_threshold_sensitivity

    policy = SimpleNamespace(
        discovery_q=0.10,
        fwer_report_alpha=0.05,
        expected_directions={},
        min_abs_ic=0.01,
        min_abs_t=2.0,
        monthly_turnover_reference=0.50,
        annual_direction_ratio=0.60,
        annual_effect_ratio=0.65,
        cost_safety_margin=1.5,
    )
    results = [{
        "name": "signal",
        "all_periods": {"1": {
            "ols_p_value": 0.001,
            "estimable": True,
            "ic": 0.02,
            "ols_hac_t": 3.0,
            "period": 1,
            "preprocessing_variant": "neutralized",
        }},
    }]
    report = _build_threshold_sensitivity(results, policy)

    for scenario in report["scenarios"].values():
        assert scenario["local_fdr_factor_names"] == ["signal"]
        assert scenario["economic_factor_names"] == ["signal"]
        assert scenario["local_fdr_jaccard_vs_baseline"] == pytest.approx(1.0)
        assert scenario["economic_jaccard_vs_baseline"] == pytest.approx(1.0)


def test_deployment_adaptivity_loads_only_frozen_discovery_contract(tmp_path):
    from core.config import load_config
    from core.sectors import taxonomy_sha256
    from research.validation import validation_policy_sha256
    from workflows.factor_adaptivity import load_discovery_contract

    policy = load_config("config/default.yaml").validation_policy
    payload = {
        "config": {
            "validation_policy_sha256": validation_policy_sha256(policy),
            "taxonomy_sha256": taxonomy_sha256(),
        },
        "final_factors": ["signal"],
        "significant_factors": [{
            "name": "signal",
            "best_variant": "raw",
            "observation_channel": True,
            "observation_reasons": ["locked_oos_pending"],
            "promotion_status": "observation",
            "weight_cap": 0.5,
        }],
    }
    path = tmp_path / "ic_by_window_period.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    names, variants, metadata = load_discovery_contract(
        str(path),
        expected_policy_sha256=validation_policy_sha256(policy),
        expected_taxonomy_sha256=taxonomy_sha256(),
    )
    assert names == ["signal"]
    assert variants == {"signal": "raw"}
    assert metadata["signal"]["weight_cap"] == pytest.approx(0.5)

    with pytest.raises(ValueError, match="taxonomy hash mismatch"):
        load_discovery_contract(
            str(path),
            expected_policy_sha256=validation_policy_sha256(policy),
            expected_taxonomy_sha256="0" * 64,
        )

    del payload["significant_factors"][0]["best_variant"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid preprocessing variants"):
        load_discovery_contract(
            str(path),
            expected_policy_sha256=validation_policy_sha256(policy),
            expected_taxonomy_sha256=taxonomy_sha256(),
        )


def test_adaptivity_rejects_neutralized_contract_without_neutralize_step():
    from types import SimpleNamespace

    from core.config import load_config
    from workflows.factor_adaptivity import run_adaptivity_analysis

    config = load_config("config/default.yaml")
    assert "neutralize" not in {str(step.type) for step in config.processing}

    with pytest.raises(ValueError, match="requires neutralized preprocessing"):
        run_adaptivity_analysis(
            [], "config/default.yaml", "2026-01-01", "2026-01-01", "2026-01-02",
            config=config,
            runner=SimpleNamespace(config=config),
            preprocessing_variants={"signal": "neutralized"},
        )
