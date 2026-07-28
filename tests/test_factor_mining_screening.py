from __future__ import annotations

import pandas as pd

from factor_mining.api import CandidateSpec, FeatureConfig, TargetSpec
from factor_mining.cli import _record_prescreen_outcome
from factor_mining.data import make_synthetic_panels
from factor_mining.operators import Expr
from factor_mining.repository import CandidateRepository
from factor_mining.screening import ScreeningConfig, screen_candidates


def _candidate(candidate_id: str, expression: Expr) -> CandidateSpec:
    return CandidateSpec(
        candidate_id=candidate_id,
        framework_name=f"mined_{candidate_id}",
        kind="symbolic",
        category="auto_mined",
        frequency="1min",
        target=TargetSpec(name="forward_5p", horizon_bars=5, cost_bps=1.0),
        dependencies=("close",),
        lookback_bars=10,
        payload={
            "expression": expression.to_dict(),
            "expression_sha256": expression.sha256,
            "decision_lag_bars": 1,
            "postprocess": {
                "mad_clip": 5.0,
                "neutralize_volatility": False,
                "volatility_feature": None,
            },
        },
        feature_config=FeatureConfig(
            raw_fields=("close",),
            feature_horizons=(1, 5),
            lag_steps=(1,),
            rolling_windows=(3, 5),
        ),
    )


def test_prescreen_keeps_correlated_valid_candidates_and_rejects_constants():
    panels = make_synthetic_panels(periods=180, symbols=8, seed=41)
    base = Expr.operation("cs_rank", Expr.terminal("return_1p"))
    equivalent = Expr.operation("add", base, Expr.constant(0.0))
    candidates = (
        _candidate("gp_screen_a", base),
        _candidate("gp_screen_b", equivalent),
        _candidate("gp_screen_constant", Expr.constant(1.0)),
    )

    outcome = screen_candidates(
        candidates,
        panels,
        config=ScreeningConfig(
            min_time_observations=20,
            min_coverage=0.20,
            min_variable_row_fraction=0.05,
            max_correlation_observations=2_000,
        ),
    )

    assert outcome.passed_candidate_ids == ("gp_screen_a", "gp_screen_b")
    by_id = {item["candidate_id"]: item for item in outcome.results}
    assert "high_peer_correlation" in by_id["gp_screen_a"]["soft_flags"]
    assert set(by_id["gp_screen_a"]["predictive_ic_decay"]["curve"]) == {
        "1", "3", "5", "10", "20", "40"
    }
    persistence = by_id["gp_screen_a"]["signal_rank_persistence"]
    assert persistence["lag_unit"] == "decision_bars"
    assert set(persistence["curve"]) == {"1", "3", "5", "10", "20", "40"}
    assert (
        by_id["gp_screen_a"]["diagnostic_universe_policy"]
        == "static_declared_universe"
    )
    assert "no_terminal_dependency" in by_id["gp_screen_constant"]["hard_reasons"]
    assert "insufficient_cross_sectional_variation" in by_id[
        "gp_screen_constant"
    ]["hard_reasons"]


def test_prescreen_applies_point_in_time_eligibility_and_excludes_warmup():
    panels = make_synthetic_panels(periods=180, symbols=8, seed=43)
    candidate = _candidate("gp_dynamic_pool", Expr.terminal("return_1p"))
    eligibility = pd.DataFrame(
        False,
        index=panels["close"].index,
        columns=panels["close"].columns,
    )
    eligibility.iloc[90:, :6] = True

    outcome = screen_candidates(
        (candidate,),
        panels,
        eligibility=eligibility,
        config=ScreeningConfig(
            min_time_observations=20,
            min_coverage=0.20,
            min_variable_row_fraction=0.05,
        ),
    )

    result = outcome.results[0]
    assert result["hard_pass"] is True
    assert result["diagnostic_universe_policy"] == "point_in_time_eligibility"
    assert 20 <= result["time_observations"] < 90


def test_prescreen_retains_dynamic_pool_warmup_for_signal_processing():
    panels = make_synthetic_panels(periods=180, symbols=8, seed=45)
    candidate = _candidate(
        "gp_dynamic_warmup",
        Expr.operation("ts_mean", Expr.terminal("return_1p"), window=5),
    )
    evaluation = pd.DataFrame(
        False,
        index=panels["close"].index,
        columns=panels["close"].columns,
    )
    evaluation.iloc[90:, :6] = True
    processing = pd.DataFrame(
        False,
        index=panels["close"].index,
        columns=panels["close"].columns,
    )
    processing.iloc[:, :6] = True
    config = ScreeningConfig(min_time_observations=20, min_coverage=0.20)

    retained = screen_candidates(
        (candidate,),
        panels,
        eligibility=evaluation,
        processing_eligibility=processing,
        config=config,
    ).results[0]
    truncated = screen_candidates(
        (candidate,),
        panels,
        eligibility=evaluation,
        processing_eligibility=evaluation,
        config=config,
    ).results[0]

    assert retained["time_observations"] > truncated["time_observations"]


def test_prescreen_excludes_ineligible_symbols_before_neutralization():
    panels = make_synthetic_panels(periods=180, symbols=8, seed=44)
    candidate = _candidate(
        "gp_dynamic_neutralization",
        Expr.operation(
            "square", Expr.operation("cs_zscore", Expr.terminal("return_1p"))
        ),
    )
    eligibility = pd.DataFrame(
        True,
        index=panels["close"].index,
        columns=panels["close"].columns,
    )
    eligibility.iloc[:, 6:] = False
    config = ScreeningConfig(min_time_observations=20, min_coverage=0.20)

    baseline = screen_candidates(
        (candidate,), panels, eligibility=eligibility, config=config
    ).results[0]

    perturbed = {name: frame.copy() for name, frame in panels.items()}
    perturbed["close"].iloc[:, 6:] *= 10_000.0
    result = screen_candidates(
        (candidate,), perturbed, eligibility=eligibility, config=config
    ).results[0]

    assert result["mean_rank_ic"] == baseline["mean_rank_ic"]
    assert result["coverage"] == baseline["coverage"]


def test_recording_prescreen_evidence_does_not_promote_candidate(tmp_path):
    repository = CandidateRepository(tmp_path / "candidates.sqlite3")
    candidate = _candidate(
        "gp_screen_repository", Expr.terminal("return_1p")
    )
    repository.add_candidates((candidate,))

    evaluation_id = repository.record_evaluation(
        candidate.candidate_id,
        stage="mining_prescreen",
        metrics={"hard_pass": True},
        evidence={"valid": True, "scope": "mining_prescreen_not_formal_evidence"},
        run_id="screen_test",
    )

    assert evaluation_id > 0
    assert repository.get_candidate(candidate.candidate_id).status == "mined_candidate"


def test_prescreen_hard_failure_is_terminally_rejected(tmp_path):
    repository = CandidateRepository(tmp_path / "candidates.sqlite3")
    candidate = _candidate(
        "gp_screen_rejected", Expr.terminal("return_1p")
    )
    repository.add_candidates((candidate,))

    _record_prescreen_outcome(
        repository,
        candidate.candidate_id,
        {
            "hard_pass": False,
            "hard_reasons": ["insufficient_coverage"],
        },
        {"scope": "mining_prescreen_not_formal_evidence"},
        run_id="screen_test",
    )

    assert repository.get_candidate(candidate.candidate_id).status == "rejected"
