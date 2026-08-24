from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from core.config import load_config
from workflows.walkforward import (
    _build_fold_bundle,
    _build_rolling_folds,
    _candidate_factor_names,
    _warmup_start,
    rolling_walk_forward,
)
from workflows.research import _research_period_protocol


def test_intraday_walkforward_pool_is_the_registered_588_contract():
    names = _candidate_factor_names("factors.library.intraday")

    assert len(names) == 588


def test_intraday_walkforward_uses_configured_90_day_warmup():
    config = load_config("config/default.yaml")

    assert _warmup_start(config, "daily_intraday", "2026-08-20") == pd.Timestamp(
        "2026-05-22"
    )


def test_fold_bundle_creates_artifact_directory_before_checkpoint(monkeypatch, tmp_path):
    output_dir = tmp_path / "fold" / "artifacts"
    monkeypatch.setattr("pipeline.runner.PipelineRunner", lambda config: object())

    def stop_after_directory_check(*args, **kwargs):
        assert output_dir.is_dir()
        raise RuntimeError("stop after directory check")

    monkeypatch.setattr(
        "workflows.research._run_multi_period_screening",
        stop_after_directory_check,
    )

    with pytest.raises(RuntimeError, match="stop after directory check"):
        _build_fold_bundle(
            load_config("config/default.yaml"),
            name="fold-1",
            train_start="2025-01-01",
            train_end="2025-07-01",
            output_dir=output_dir,
            candidate_factors=["intraday_probe"],
            build_correlation=False,
            fdr_method="hierarchical",
            frequency="daily_intraday",
        )


def test_only_declared_training_contracts_are_selection_eligible():
    factor_validation = _research_period_protocol(
        "factor_validation_is", "2026-01-01", "2026-04-01", "2026-08-01"
    )
    rolling = _research_period_protocol(
        "rolling_walkforward_train", "2026-01-01", "2026-04-01", "2026-08-01"
    )
    control = _research_period_protocol(
        "long_history_replay", "2016-03-31", "2016-03-31", "2026-08-20"
    )

    assert factor_validation["selection_eligible"] is True
    assert rolling["selection_eligible"] is True
    assert control["selection_eligible"] is False


def test_intraday_walkforward_defaults_to_intraday_module(monkeypatch, tmp_path):
    captured = []
    monkeypatch.setattr(
        "workflows.walkforward._candidate_factor_names",
        lambda prefix: captured.append(prefix) or ["intraday_probe"],
    )
    monkeypatch.setattr(
        "workflows.walkforward._build_fold_bundle",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("stop test")),
    )
    assessment = MagicMock(sufficient=True)
    monkeypatch.setattr(
        "research.sample_policy.assess_sample_counts",
        lambda *args, **kwargs: assessment,
    )

    rolling_walk_forward(
        load_config("config/default.yaml"),
        run_root=tmp_path / "run",
        build_correlation=False,
        max_folds=1,
        is_intraday=True,
        calendar=pd.bdate_range("2025-01-01", periods=180),
    )

    assert captured == ["factors.library.intraday"]


def test_intraday_walkforward_uses_only_five_most_recent_folds(monkeypatch, tmp_path):
    observed = []
    monkeypatch.setattr(
        "workflows.walkforward._build_fold_bundle",
        lambda *args, **kwargs: (
            observed.append(kwargs["name"]),
            (_ for _ in ()).throw(RuntimeError("stop test")),
        )[1],
    )
    assessment = MagicMock(sufficient=True)
    monkeypatch.setattr(
        "research.sample_policy.assess_sample_counts",
        lambda *args, **kwargs: assessment,
    )

    calendar = pd.bdate_range("2017-01-01", "2026-08-20")
    results = rolling_walk_forward(
        load_config("config/default.yaml"),
        run_root=tmp_path / "run",
        candidate_factors=["intraday_probe"],
        build_correlation=False,
        is_intraday=True,
        calendar=calendar,
    )

    assert len(results) == len(observed) == 5
    assert results[-1]["test_end"] == "2026-05-15"


def test_short_walkforward_calendar_retries_without_name_error():
    calendar = pd.bdate_range("2024-01-01", periods=12)

    folds = _build_rolling_folds(
        calendar, train_bars=5, test_bars=4, step_bars=4
    )

    assert len(folds) == 2
