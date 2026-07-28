from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_mining.api import FeatureConfig, TargetSpec
from factor_mining.data import LocalParquetData, LocalParquetSpec, make_synthetic_panels
from factor_mining.features import FeatureEngine
from factor_mining.validation import PreparedTarget, ValidationConfig, prepare_signal


def test_one_minute_feature_vocabulary_includes_one_bar_information():
    panels = make_synthetic_panels(periods=80, symbols=5)
    features = FeatureEngine(FeatureConfig(
        feature_horizons=(1, 5), lag_steps=(1,), rolling_windows=(3, 5)
    )).build(panels)

    assert "return_1p" in features.values
    assert "intrabar_return_1p" in features.values
    assert "close_lag_1p" in features.values
    assert "close_mean_3p" in features.values
    assert "close_mean_1p" not in features.values
    assert features.values["return_1p"].dtype == np.float32


def test_selected_feature_build_materializes_only_runtime_terminals():
    panels = make_synthetic_panels(periods=80, symbols=5)
    config = FeatureConfig(
        feature_horizons=(1, 5), lag_steps=(1,), rolling_windows=(3, 5, 60)
    )

    features = FeatureEngine(config).build(
        panels, required_features=("return_1p", "realized_vol_60p")
    )

    assert set(features.values) == {"return_1p", "realized_vol_60p"}


def test_feature_memory_budget_fails_before_unbounded_growth():
    panels = make_synthetic_panels(periods=3500, symbols=20)
    config = FeatureConfig(max_feature_memory_mb=64)

    with pytest.raises(MemoryError, match="max_feature_memory_mb"):
        FeatureEngine(config).build(panels)


def test_feature_and_decision_lag_do_not_consume_changed_current_bar():
    panels = make_synthetic_panels(periods=80, symbols=5)
    config = FeatureConfig(
        feature_horizons=(1, 5), lag_steps=(1,), rolling_windows=(3, 5)
    )
    baseline = FeatureEngine(config).build(panels)
    changed = {name: frame.copy() for name, frame in panels.items()}
    changed["close"].iloc[-1] *= 3.0
    revised = FeatureEngine(config).build(changed)

    before = prepare_signal(
        baseline.values["return_1p"],
        ValidationConfig(neutralize_volatility=False),
    )
    after = prepare_signal(
        revised.values["return_1p"],
        ValidationConfig(neutralize_volatility=False),
    )
    np.testing.assert_allclose(before[-1], after[-1], equal_nan=True)


def test_missing_open_is_not_silently_proxied_with_close():
    panels = make_synthetic_panels(periods=50, symbols=4)
    panels.pop("open")
    features = FeatureEngine(FeatureConfig(
        feature_horizons=(1, 5), lag_steps=(1,), rolling_windows=(3, 5)
    )).build(panels)

    assert np.isnan(features.values["open_gap_1p"]).all()
    assert np.isnan(features.values["intrabar_return_1p"]).all()


def test_target_uses_delayed_entry_and_forward_bar_horizon():
    index = pd.date_range("2024-01-02 09:00", periods=8, freq="1min")
    close = pd.DataFrame({"A": np.arange(100.0, 108.0)}, index=index)
    spec = TargetSpec(name="fwd_2p", horizon_bars=2, entry_delay_bars=1)

    target = PreparedTarget.from_close(close, spec)

    assert np.isclose(target.values[0, 0], 103.0 / 101.0 - 1.0)
    assert np.isnan(target.values[-3:, 0]).all()


def test_target_horizon_uses_each_instruments_own_valid_bars():
    index = pd.date_range("2024-01-02 09:00", periods=5, freq="1min")
    close = pd.DataFrame({"A": [100.0, 101.0, np.nan, 103.0, 104.0]}, index=index)
    spec = TargetSpec(name="fwd_1p", horizon_bars=1, entry_delay_bars=1)

    target = PreparedTarget.from_close(close, spec)

    assert target.values[0, 0] == pytest.approx(103.0 / 101.0 - 1.0)


def test_local_adapter_explicitly_disables_mysql(tmp_path, monkeypatch):
    captured = {}

    class FakeSource:
        def __init__(self, parquet_config, mysql_config=None):
            captured["config"] = parquet_config
            captured["mysql"] = mysql_config

    monkeypatch.setattr("data.parquet_source.ParquetFuturesSource", FakeSource)
    LocalParquetData(LocalParquetSpec(tmp_path))

    assert captured["mysql"] is None
    assert captured["config"]["root_path"] == str(tmp_path.resolve())
