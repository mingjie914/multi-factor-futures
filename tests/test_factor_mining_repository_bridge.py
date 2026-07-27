from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from core.registry import get
from factor_mining.api import (
    CandidateSpec,
    FeatureConfig,
    MiningRunSpec,
    RunMode,
    TargetSpec,
)
from factor_mining.bridge import (
    compute_symbolic_candidate,
    register_snapshot,
    register_snapshot_from_environment,
)
from factor_mining.data import make_synthetic_panels
from factor_mining.operators import Expr
from factor_mining.repository import CandidateRepository, load_snapshot
from factor_mining.validation import ValidationConfig, prepare_signal


class FrameProvider:
    def __init__(self, frames):
        self.frames = frames

    def get(self, field, dates, universe):
        return self.frames.get(field, pd.DataFrame()).reindex(
            index=dates, columns=universe
        )


def _candidate(name="mined_bridge_contract_test"):
    expression = Expr.operation("cs_rank", Expr.terminal("return_1p"))
    return CandidateSpec(
        candidate_id="gp_bridge_contract_test",
        framework_name=name,
        kind="symbolic",
        category="auto_mined",
        frequency="1min",
        target=TargetSpec(name="forward_5p", horizon_bars=5),
        dependencies=("close",),
        lookback_bars=2,
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
            raw_fields=("close",), feature_horizons=(1, 5),
            lag_steps=(1,), rolling_windows=(3, 5),
        ),
        lineage={"engine": "gp", "run_id": "repo_test"},
    )


def _repository_with_candidate(tmp_path):
    repository = CandidateRepository(tmp_path / "candidates.sqlite3")
    candidate = _candidate()
    repository.add_run(MiningRunSpec(
        run_id="repo_test", mode=RunMode.DEV, seed=1,
        start="2024-01-01", end="2024-02-01", universe=("A", "B"),
        target=candidate.target, feature_config=candidate.feature_config,
    ))
    repository.add_candidates((candidate,), run_id="repo_test")
    return repository, candidate


def test_sqlite_catalog_and_immutable_snapshot_can_coexist(tmp_path):
    repository, candidate = _repository_with_candidate(tmp_path)
    snapshot = repository.write_snapshot(
        tmp_path / "selected.json", candidate_ids=(candidate.candidate_id,)
    )

    loaded = load_snapshot(snapshot)

    assert loaded == (CandidateSpec.from_dict(candidate.to_dict()),)
    with pytest.raises(FileExistsError):
        repository.write_snapshot(
            snapshot, candidate_ids=(candidate.candidate_id,)
        )


def test_repository_can_freeze_candidate_selection_by_run(tmp_path):
    repository, candidate = _repository_with_candidate(tmp_path)

    assert repository.list_candidates(run_ids=("repo_test",)) == (
        CandidateSpec.from_dict(candidate.to_dict()),
    )
    assert repository.list_candidates(run_ids=("missing_run",)) == ()


def test_candidate_status_requires_evidence_and_cannot_move_backwards(tmp_path):
    repository, candidate = _repository_with_candidate(tmp_path)
    with pytest.raises(ValueError, match="requires audit evidence"):
        repository.promote(candidate.candidate_id, "development_candidate")

    repository.promote(
        candidate.candidate_id,
        "development_candidate",
        evidence={"valid": True, "study_id": "test"},
    )
    with pytest.raises(ValueError, match="invalid candidate status transition"):
        repository.promote(
            candidate.candidate_id,
            "mined_candidate",
            evidence={"valid": True},
        )


def test_snapshot_tampering_is_rejected(tmp_path):
    repository, candidate = _repository_with_candidate(tmp_path)
    snapshot = repository.write_snapshot(
        tmp_path / "selected.json", candidate_ids=(candidate.candidate_id,)
    )
    value = json.loads(snapshot.read_text(encoding="utf-8"))
    value["candidates"][0]["framework_name"] = "tampered"
    snapshot.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="hash mismatch"):
        load_snapshot(snapshot)


def test_bridge_registers_normal_factor_and_preserves_point_in_time(tmp_path):
    repository, candidate = _repository_with_candidate(tmp_path)
    snapshot = repository.write_snapshot(
        tmp_path / "selected.json", candidate_ids=(candidate.candidate_id,)
    )
    names = register_snapshot(snapshot)
    factor = get("factor", candidate.framework_name)()
    panels = make_synthetic_panels(periods=80, symbols=5)
    dates = panels["close"].index
    universe = panels["close"].columns

    baseline = factor.compute(FrameProvider(panels), dates, universe)
    changed = {field: frame.copy() for field, frame in panels.items()}
    changed["close"].iloc[-1] *= 4.0
    revised = factor.compute(FrameProvider(changed), dates, universe)

    assert names == (candidate.framework_name,)
    assert factor.dependencies() == ["close"]
    assert factor.frequency == "1min"
    assert baseline.index.equals(dates)
    assert baseline.columns.equals(universe)
    assert np.isfinite(baseline.iloc[5:].to_numpy()).any()
    pd.testing.assert_series_equal(baseline.iloc[-1], revised.iloc[-1])


def test_bridge_combines_signal_and_target_entry_lags():
    candidate = _candidate("mined_bridge_lag_alignment_test")
    panels = make_synthetic_panels(periods=50, symbols=4)
    dates = panels["close"].index
    universe = panels["close"].columns

    actual = compute_symbolic_candidate(
        candidate, FrameProvider(panels), dates, universe
    )
    raw = panels["close"].pct_change(fill_method=None).rank(
        axis=1, method="average", pct=True
    ).to_numpy(dtype=np.float32)
    expected = prepare_signal(
        raw,
        ValidationConfig(decision_lag_bars=2, neutralize_volatility=False),
    )

    np.testing.assert_allclose(actual.to_numpy(), expected, equal_nan=True)


def test_bridge_fails_closed_on_missing_dependency(monkeypatch):
    candidate = _candidate("mined_bridge_missing_test")
    panels = make_synthetic_panels(periods=40, symbols=4)
    dates = panels["close"].index
    universe = panels["close"].columns

    result = compute_symbolic_candidate(
        candidate, FrameProvider({}), dates, universe
    )

    assert result.isna().all().all()
    monkeypatch.delenv("MF_MINED_CANDIDATE_SNAPSHOT", raising=False)
    assert register_snapshot_from_environment() == ()
