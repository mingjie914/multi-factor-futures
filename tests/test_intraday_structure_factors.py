"""Standalone definitions against the frozen expression semantics and timing."""
import json
import numpy as np
import pandas as pd
import pytest

from factors.library import intraday
from factor_mining.api import CandidateSpec, FeatureConfig, TargetSpec
from factor_mining.bridge import compute_symbolic_candidate
from factor_mining.daily import daily_end
from factor_mining.operators import Expr
from data.manager import FrequencyDataProvider

CASES = json.loads(r'''
[["IntradayAmountBandwidthOi5min","5min",1,1,["amount","close","curve_top2_oi"],{"args":[{"name":"curve_top2_oi","op":"terminal","output_type":"numeric"},{"args":[{"name":"amount_lag_4p","op":"terminal","output_type":"numeric"},{"args":[{"name":"boll_width_30p","op":"terminal","output_type":"numeric"}],"op":"ts_std","output_type":"numeric","window":30}],"op":"mul","output_type":"numeric"}],"op":"max","output_type":"numeric"}],["IntradayOiBreadthDeviation5min","5min",1,-1,["close","curve_oi_breadth","curve_oi_hhi","curve_total_oi"],{"args":[{"name":"curve_oi_breadth","op":"terminal","output_type":"numeric"},{"args":[{"name":"curve_oi_hhi","op":"terminal","output_type":"numeric"},{"args":[{"name":"curve_total_oi","op":"terminal","output_type":"numeric"}],"op":"ts_zscore","output_type":"numeric","window":60}],"op":"sub","output_type":"numeric"}],"op":"mul","output_type":"numeric"}],["IntradayOiBreadthConcentrationClip5min","5min",1,-1,["close","curve_oi_breadth","curve_oi_hhi","curve_total_oi"],{"args":[{"args":[{"name":"curve_oi_hhi","op":"terminal","output_type":"numeric"},{"args":[{"name":"curve_total_oi","op":"terminal","output_type":"numeric"}],"op":"ts_zscore","output_type":"numeric","window":60}],"op":"sub","output_type":"numeric"},{"name":"curve_oi_breadth","op":"terminal","output_type":"numeric"}],"op":"min","output_type":"numeric"}],["IntradayAmountPriceDeviation5min","5min",-1,-1,["amount","close"],{"args":[{"args":[{"args":[{"name":"amount_mean_12p","op":"terminal","output_type":"numeric"},{"args":[{"name":"close_ma_gap_12p","op":"terminal","output_type":"numeric"}],"op":"ts_zscore","output_type":"numeric","window":24}],"op":"div","output_type":"numeric"}],"op":"neg","output_type":"numeric"}],"op":"abs","output_type":"numeric"}],["IntradayIlliquidityTotalOi5min","5min",1,-1,["amount","close","curve_total_oi"],{"args":[{"name":"amihud_24p","op":"terminal","output_type":"numeric"},{"name":"curve_total_oi","op":"terminal","output_type":"numeric"}],"op":"div","output_type":"numeric"}],["IntradayAmountConcentrationSpread5min","5min",1,1,["amount","close","curve_oi_concentration"],{"args":[{"args":[{"name":"amount_lag_8p","op":"terminal","output_type":"numeric"}],"op":"cs_demean","output_type":"numeric"},{"args":[{"name":"curve_oi_concentration","op":"terminal","output_type":"numeric"}],"op":"ts_max","output_type":"numeric","window":3}],"op":"sub","output_type":"numeric"}],["IntradayTrendOiComposite5min","5min",1,1,["close","oi","volume"],{"args":[{"name":"close_ma_gap_48p","op":"terminal","output_type":"numeric"},{"args":[{"args":[{"name":"volume_lag_12p","op":"terminal","output_type":"numeric"},{"name":"close_ma_gap_60p","op":"terminal","output_type":"numeric"}],"op":"min","output_type":"numeric"},{"args":[{"name":"close_ema_gap_48p","op":"terminal","output_type":"numeric"},{"name":"oi_change_15p","op":"terminal","output_type":"numeric"}],"op":"add","output_type":"numeric"}],"op":"add","output_type":"numeric"}],"op":"max","output_type":"numeric"}],["IntradayRelativeOiTrendClip1min","1min",-1,-1,["close","oi"],{"args":[{"name":"oi_relative_48p","op":"terminal","output_type":"numeric"},{"args":[{"args":[{"args":[{"name":"close_ema_gap_60p","op":"terminal","output_type":"numeric"},{"name":"close_ma_gap_48p","op":"terminal","output_type":"numeric"}],"op":"min","output_type":"numeric"},{"name":"close_ma_gap_48p","op":"terminal","output_type":"numeric"}],"op":"min","output_type":"numeric"}],"op":"neg","output_type":"numeric"}],"op":"max","output_type":"numeric"}],["IntradayOiChangeConcentrationAdjusted5min","5min",-1,-1,["close","curve_oi_hhi","oi"],{"args":[{"name":"oi_change_30p","op":"terminal","output_type":"numeric"},{"args":[{"name":"oi_relative_48p","op":"terminal","output_type":"numeric"},{"args":[{"name":"curve_oi_hhi","op":"terminal","output_type":"numeric"},{"args":[{"name":"curve_oi_hhi","op":"terminal","output_type":"numeric"}],"op":"signed_sqrt","output_type":"numeric"}],"op":"add","output_type":"numeric"}],"op":"div","output_type":"numeric"}],"op":"sub","output_type":"numeric"}],["IntradayVolumeAmountConcentrationClip5min","5min",-1,1,["amount","close","curve_oi_hhi","volume"],{"args":[{"args":[{"name":"curve_oi_hhi","op":"terminal","output_type":"numeric"}],"op":"signed_sqrt","output_type":"numeric"},{"args":[{"name":"volume_mean_48p","op":"terminal","output_type":"numeric"},{"name":"amount_lag_12p","op":"terminal","output_type":"numeric"}],"op":"div","output_type":"numeric"}],"op":"min","output_type":"numeric"}],["IntradayOiBreadthPriceDivergence5min","5min",-1,-1,["amount","close","curve_oi_breadth","oi"],{"args":[{"args":[{"name":"amount_lag_1p","op":"terminal","output_type":"numeric"},{"name":"curve_oi_breadth","op":"terminal","output_type":"numeric"}],"op":"min","output_type":"numeric"},{"args":[{"name":"oi_change_30p","op":"terminal","output_type":"numeric"},{"name":"return_30p","op":"terminal","output_type":"numeric"}],"op":"sub","output_type":"numeric"}],"op":"mul","output_type":"numeric"}],["IntradayRelativeOiPriceClip1min","1min",1,-1,["close","oi"],{"args":[{"args":[{"name":"oi_relative_48p","op":"terminal","output_type":"numeric"}],"op":"neg","output_type":"numeric"},{"name":"close_ma_gap_60p","op":"terminal","output_type":"numeric"}],"op":"min","output_type":"numeric"}],["IntradayAmountOiChangeIntensity5min","5min",-1,-1,["amount","close","curve_contract_count","oi_change"],{"args":[{"args":[{"args":[{"args":[{"name":"amount_relative_48p","op":"terminal","output_type":"numeric"}],"op":"abs","output_type":"numeric"},{"args":[{"name":"oi_change","op":"terminal","output_type":"numeric"},{"name":"amount_mean_12p","op":"terminal","output_type":"numeric"}],"op":"div","output_type":"numeric"}],"op":"mul","output_type":"numeric"},{"name":"curve_contract_count","op":"terminal","output_type":"numeric"}],"op":"mul","output_type":"numeric"}],"op":"signed_sqrt","output_type":"numeric"}]]
''')


@pytest.mark.parametrize("definition", CASES, ids=[c[0] for c in CASES])
@pytest.mark.parametrize("case", ["ordinary", "missing", "constant"])
def test_minute_structure_matches_expression_and_prefix(definition, case):
    cls, frequency, raw_direction, daily_direction, dependencies, expression = definition
    factor = getattr(intraday, cls)()
    days = pd.bdate_range("2025-01-06", periods=6)
    bars = pd.DatetimeIndex([day + pd.Timedelta(minutes=540+i*int(frequency[0]))
                             for day in days for i in range(90)])
    columns = pd.Index([f"S{i}" for i in range(12)])
    rng = np.random.default_rng(612)
    shape = (len(bars), len(columns))
    fields = set(dependencies) | {"close"}
    panels = {f: pd.DataFrame(rng.uniform(.2, 2., shape), index=bars, columns=columns) for f in fields}
    panels["close"] = pd.DataFrame(100 + rng.normal(0, .1, shape).cumsum(axis=0), index=bars, columns=columns)
    for field, scale in {"amount": 1e8, "volume": 1e4, "oi": 1e5,
                         "curve_top2_oi": 1e5, "curve_total_oi": 1e6}.items():
        if field in panels:
            panels[field] *= scale
    if case == "missing":
        for frame in panels.values():
            frame.iloc[50:65, 0] = np.nan
            frame.iloc[179, 1] = np.nan
            frame.iloc[::41, 2] = np.nan
    if case == "constant":
        for frame in panels.values():
            frame.iloc[:, 0] = 1.
            frame.iloc[:, 1] = 0.
    calls = []
    class Source:
        def fetch_price_at_frequency(self, universe, start, end, fields, *, frequency):
            calls.append(frequency)
            return {f: panels[f].loc[(bars.normalize() >= start) & (bars.normalize() <= end), universe]
                    for f in fields}
        def trading_session_index(self, index):
            return index
    class Data:
        source = Source()
        _factor_eligibility = pd.DataFrame(True, index=days, columns=columns)
    data = Data()
    data._factor_eligibility.iloc[2, 3] = False
    expr = Expr.from_dict(expression)
    candidate = CandidateSpec(candidate_id="reference", framework_name="reference", kind="symbolic",
        category=factor.category, frequency=frequency, target=TargetSpec("target", frequency),
        dependencies=tuple(dependencies), lookback_bars=61,
        payload={"expression": expression, "expression_sha256": expr.sha256,
                 "decision_lag_bars": 1, "postprocess": {"mad_clip": 5.,
                 "neutralize_volatility": True, "volatility_feature": "realized_vol_60p"}},
        feature_config=FeatureConfig(source_frequency=frequency, decision_frequency=frequency,
            raw_fields=tuple(fields), rolling_windows=(12,24,30,48,60),
            lag_steps=(1,4,8,12), feature_horizons=(1,15,30)), expected_direction=raw_direction)
    provider = FrequencyDataProvider(data, frequency, days[0], days[-1], columns)
    provider._factor_eligibility = data._factor_eligibility.reindex(bars.normalize())
    provider._factor_eligibility.index = bars
    reference = compute_symbolic_candidate(candidate, provider, bars, columns)
    expected = daily_end(reference, panels["close"], bars) * daily_direction
    intraday.clear_transient_data_caches()
    actual = factor.compute(data, days, columns)
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5, equal_nan=True)
    prefix = factor.compute(data, days[:4], columns)
    pd.testing.assert_frame_equal(actual.loc[days[:4]], prefix)
    assert set(calls) == {frequency}
    assert factor.frequency == factor.signal_frequency == "daily"
    assert factor.validation_horizons == (5, 10, 20)
    assert factor.prefetch_dependencies() == []
    assert factor.training_days > 0
    assert factor.expected_direction == 1
    assert factor.category in {"momentum", "term_structure"}
    intraday.clear_transient_data_caches()


def test_structure_registry_has_only_two_one_minute_definitions():
    factors = [getattr(intraday, c[0])() for c in CASES]
    assert sum(f.input_bar_frequency == "1min" for f in factors) == 2
    assert len({f.name for f in factors}) == 13
    for factor in factors:
        assert factor.name.startswith("intraday_")
        assert factor.name.endswith("_" + factor.input_bar_frequency)
        result = factor.compute(None, pd.DatetimeIndex([]), ["A"])
        assert result.empty
