from __future__ import annotations

import sys

import pandas as pd
import pytest

from core.factor_contract import bind_factor_contract, validate_factor_contract
from core.interfaces import Factor
from core.period import iter_overlapping_chunks
from core.registry import _REGISTRIES
from workflows.research import (
    _compute_factor_date_chunks,
    _load_adaptivity_data,
    _load_research_checkpoint,
    _passes_post_bonferroni_quality,
    _parse_requested_factors,
    _select_registered_factors,
    _validate_requested_factors,
    _write_json_atomic,
)


def test_requested_factor_parser_is_stable_and_deduplicated():
    assert _parse_requested_factors(" beta,alpha,beta ") == ["beta", "alpha"]
    with pytest.raises(ValueError, match="at least one"):
        _parse_requested_factors(" , ")


def test_research_checkpoint_roundtrip_and_contract_guard(tmp_path):
    path = tmp_path / ".multi_period_checkpoint.json"
    contract = {"version": 1, "factors": ["alpha"]}
    results = [{"name": "alpha", "best_ic": 0.01}]

    _write_json_atomic(str(path), {"contract": contract, "results": results})

    assert _load_research_checkpoint(str(path), contract) == results
    with pytest.raises(RuntimeError, match="contract"):
        _load_research_checkpoint(
            str(path), {"version": 1, "factors": ["beta"]}
        )
    assert not list(tmp_path.glob("*.tmp"))


def test_date_chunk_compute_preserves_sparse_early_history(monkeypatch):
    import factors.engine as engine_module

    dates = pd.date_range("2024-01-01", periods=6, freq="B")
    universe = pd.Index(["RB"])

    class Data:
        frequency = "daily"

        def prefetch(self, *args, **kwargs):
            return None

    class SparseFactor(Factor):
        name = "sparse_factor"
        frequency = "daily"
        validation_horizons = (3, 5, 10)

        def dependencies(self):
            return []

        def compute(self, data, requested_dates, requested_universe):
            values = pd.DataFrame(
                1.0, index=requested_dates, columns=requested_universe
            )
            values.loc[values.index < dates[3]] = float("nan")
            return values

    monkeypatch.setattr(
        engine_module, "registry_get", lambda kind, name: SparseFactor
    )
    expected = engine_module.FactorEngine(Data()).compute_factors(
        ["sparse_factor"], dates, universe
    )["sparse_factor"]
    actual, invalid = _compute_factor_date_chunks(
        Data(), ["sparse_factor"], list(iter_overlapping_chunks(dates, 3, 0)),
        universe, 1, tolerate_failures=False, clear_intraday_caches=False,
    )

    assert invalid == []
    pd.testing.assert_frame_equal(
        actual["sparse_factor"], expected, check_freq=False
    )


def test_explicit_adaptivity_file_fails_closed(tmp_path):
    bad = tmp_path / "adaptivity.csv"
    bad.write_text("wrong\nvalue\n", encoding="utf-8")
    with pytest.raises(ValueError, match="factor"):
        _load_adaptivity_data(str(bad))


def test_registered_factor_contract_rejects_formal_horizon_override():
    class LegacyFactor:
        name = "contract_probe_5d"
        frequency = "daily"

    bind_factor_contract(LegacyFactor, LegacyFactor.name)
    factor = LegacyFactor()
    assert factor.validation_horizons == (3, 5, 10)
    with pytest.raises(ValueError, match="requested horizons.*do not match"):
        validate_factor_contract(
            factor, provider_frequency="daily", requested_horizons=[5]
        )


def test_requested_factor_validation_rejects_unknown_names():
    assert _validate_requested_factors(["known"], {"known"}) == ["known"]
    with pytest.raises(ValueError, match="missing"):
        _validate_requested_factors(["known", "missing"], {"known"})


def test_all_factor_selection_respects_frequency_and_module_family():
    daily_intraday = type(
        "DailyIntraday", (), {"frequency": "daily", "__module__": "factors.library.intraday"}
    )
    minute_intraday = type(
        "MinuteIntraday", (), {"frequency": "1min", "__module__": "factors.library.intraday"}
    )
    daily_other = type(
        "DailyOther", (), {"frequency": "daily", "__module__": "factors.library.technical"}
    )
    registry = {"z": daily_other, "b": minute_intraday, "a": daily_intraday}

    assert _select_registered_factors(
        registry, "daily", "factors.library.intraday"
    ) == ["a"]
    assert _select_registered_factors(registry, "1min") == ["b"]


def test_post_discovery_quality_uses_ic_and_t_not_stock_ir_cutoff():
    assert _passes_post_bonferroni_quality({
        "best_ic": 0.036,
        "best_t": 2.10,
        "best_ic_pos_ratio": 0.566,
        "best_ir": 0.075,
    })
    assert not _passes_post_bonferroni_quality({
        "best_ic": 0.009,
        "best_t": 3.0,
        "best_ic_pos_ratio": 0.60,
        "best_ir": 0.80,
    })
    assert _passes_post_bonferroni_quality({
        "best_ic": -0.03,
        "best_t": -2.2,
        "best_ic_pos_ratio": 0.45,
        "best_ir": -0.06,
    })
    assert not _passes_post_bonferroni_quality({
        "best_ic": 0.03,
        "best_t": 1.99,
        "best_ir": 0.80,
    })


def test_user_factor_discovery_is_sorted_and_duplicate_safe(tmp_path, monkeypatch):
    import factors.user as user_factors

    module_names = ["skill_test_b", "skill_test_a"]
    factor_names = ["skill_test_factor_b", "skill_test_factor_a"]
    for module_name, factor_name in zip(module_names, factor_names):
        (tmp_path / f"{module_name}.py").write_text(
            "from factors.user import register_user_factor\n"
            f"@register_user_factor({factor_name!r}, category='test')\n"
            "class SyntheticFactor:\n"
            "    pass\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(user_factors, "__path__", [str(tmp_path)])
    try:
        assert user_factors.load_user_factors() == tuple(sorted(module_names))
        registered = _REGISTRIES.get("factor", {})
        assert all(name in registered for name in factor_names)
        with pytest.raises(ValueError, match="already registered"):
            user_factors.register_user_factor(factor_names[0], category="test")
    finally:
        for module_name in module_names:
            sys.modules.pop(f"factors.user.{module_name}", None)
        for factor_name in factor_names:
            _REGISTRIES.get("factor", {}).pop(factor_name, None)
