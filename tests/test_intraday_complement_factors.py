"""Daily complement factors: independent formula, causality and source contracts."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import factors.library.intraday as intraday


NAMES = (
    "oi_path_efficiency", "oi_stable_price_trend",
    "seat_concentration_pressure", "seat_crowding_release",
    "liquidation_reversal", "oi_price_response_asymmetry",
    "sector_dispersion_repair", "sector_dispersion_breakout",
    "sector_direction_persistence", "sector_downside_decoupling",
    "sector_delayed_response", "sector_leader_confirmation",
    "calendar_seasonal_excess", "calendar_seasonal_consensus",
)


@pytest.mark.parametrize("name", [
    "intraday_calendar_seasonal_excess_20d",
    "intraday_calendar_seasonal_consensus_20d",
    "intraday_historical_price_shape_return_5d",
    "intraday_historical_price_shape_weighted_5d",
    "intraday_historical_price_shape_sign_consensus_5d",
    "intraday_historical_intraday_shape_return_30m",
    "intraday_historical_intraday_shape_positive_weight_30m",
    "intraday_historical_intraday_shape_sign_consensus_30m",
])
def test_declared_history_matches_full_compute_without_extending_other_factors(name, monkeypatch):
    from core.registry import get
    from core.period import iter_overlapping_chunks
    from workflows.research import _compute_factor_date_chunks

    minute = "intraday_shape" in name
    data = DailyData(periods=600 if minute else 3100)
    universe = data.universe[:2]
    dates = data.dates
    if minute:
        rng = np.random.default_rng(682)
        index = pd.DatetimeIndex([d + pd.Timedelta(minutes=540 + m)
                                  for d in dates for m in range(150)])
        close = pd.DataFrame(100 * np.exp(rng.normal(0, .001, (len(index), 2)).cumsum(0)),
                             index=index, columns=universe)
    else:
        close = data.fields["close"][universe]
    panel = {"open": close * .999, "high": close * 1.01,
             "low": close * .99, "close": close}
    monkeypatch.setattr(intraday, "_get_minute_panel", lambda d, days, u, freq:
                        {k: v.loc[v.index.normalize().isin(days), u] for k, v in panel.items()})
    f = get("factor", name)()
    targets = dates[-100:]
    expected = f.compute(data, dates, universe).reindex(targets)
    assert expected.notna().any().any()
    ordinary = "intraday_oi_path_efficiency_20d"
    chunks = list(iter_overlapping_chunks(targets, 50, 128))
    profile = {}
    actual, invalid = _compute_factor_date_chunks(
        data, [name, ordinary], chunks, universe, 2,
        tolerate_failures=False, clear_intraday_caches=False, performance=profile,
        history_calendar=dates, history_days={name: f.warmup_trading_days},
    )
    assert not invalid
    pd.testing.assert_frame_equal(actual[name], expected, check_freq=False, atol=1e-12, rtol=1e-12)
    ordinary_requests = [r for r in profile["factor_timings"] if r["factor"] == ordinary]
    assert [r["request_dates"] for r in ordinary_requests] == [len(r) for _, r in chunks]


def factor(name):
    from core.registry import get
    return get("factor", "intraday_" + name + "_20d")()


class DailyData:
    def __init__(self, periods=1400):
        self.dates = pd.bdate_range("2018-01-02", periods=periods)
        self.universe = ["RB", "HC", "I", "J", "CU", "AL", "NI"]
        rng = np.random.default_rng(614)
        shape = (periods, len(self.universe))
        self.fields = {
            "close": pd.DataFrame(100 * np.exp(rng.normal(.0002, .012, shape).cumsum(0)),
                                  index=self.dates, columns=self.universe),
            "oi": pd.DataFrame(1000 * np.exp(rng.normal(0, .02, shape).cumsum(0)),
                               index=self.dates, columns=self.universe),
            "volume": pd.DataFrame(rng.lognormal(8, .5, shape), index=self.dates,
                                   columns=self.universe),
        }
        self._factor_eligibility = pd.DataFrame(True, index=self.dates, columns=self.universe)
        self.calls = []

    def get(self, field, dates, universe):
        self.calls.append(field)
        return self.fields[field].reindex(index=dates, columns=universe)

    def prefetch(self, *args, **kwargs):
        pass


@pytest.fixture
def seat_data(monkeypatch):
    data = DailyData()
    rows = []
    for j, root in enumerate(data.universe):
        for seat in range(4):
            for i, day in enumerate(data.dates):
                rows.append((day, root, str(seat),
                             100 + 30 * np.sin(i / (7 + seat) + j) + 10 * seat,
                             110 + 25 * np.cos(i / (9 + seat) + j) + 4 * seat))
    raw = pd.DataFrame(rows, columns=["_ts", "_root", "seat_name", "long_position", "short_position"])
    monkeypatch.setattr(intraday, "_get_seat_table", lambda d, dates, u, t:
                        raw.loc[raw._ts.isin(dates) & raw._root.isin(u)].copy())
    return data, raw


@pytest.mark.parametrize("name", NAMES)
def test_complement_metadata_prefix_and_no_input_mutation(name, seat_data):
    data, raw = seat_data
    f = factor(name)
    assert f.input_bar_frequency == f.signal_frequency == f.frequency == "daily"
    assert f.validation_horizons == (5, 10, 20)
    assert f.expected_direction in (-1, 1)
    assert getattr(f, "training_days", 0) == 0  # Warmup is not GP training evidence.
    originals = {k: v.copy() for k, v in data.fields.items()}
    original_seats = raw.copy()
    result = f.compute(data, data.dates, data.universe)
    assert result.notna().to_numpy().sum() > 30
    assert not np.isinf(result.to_numpy()).any()
    pd.testing.assert_frame_equal(result.iloc[:1300], f.compute(data, data.dates[:1300], data.universe))
    for k, v in originals.items():
        pd.testing.assert_frame_equal(v, data.fields[k])
    pd.testing.assert_frame_equal(raw, original_seats)
    assert f.compute(None, [], data.universe).empty
    assert f.compute(None, data.dates, []).empty


def test_oi_efficiency_matches_scalar_formula_and_constant():
    data = DailyData(90)
    f = factor("oi_path_efficiency")
    g = np.log(data.fields["oi"]).diff()
    expected = (g.rolling(20).sum() / g.abs().rolling(20).sum()).shift(1)
    actual = f.compute(data, data.dates, data.universe)
    np.testing.assert_allclose(actual, expected, atol=2e-12, equal_nan=True)
    data.fields["oi"].iloc[:, :] = 1000.
    assert (f.compute(data, data.dates, data.universe).iloc[21:] == 0).all().all()


def test_seat_pressure_uses_separate_nonnegative_shares(seat_data):
    data, raw = seat_data
    values = raw.groupby(["_ts", "_root"])[["long_position", "short_position"]]
    hhi = values.agg(lambda s: (s / s.sum()).pow(2).sum())
    diff = (hhi.long_position - hhi.short_position).unstack()
    expected = diff.diff().rolling(20).mean().reindex(columns=data.universe).shift(1)
    actual = factor("seat_concentration_pressure").compute(data, data.dates, data.universe)
    np.testing.assert_allclose(actual, expected, atol=2e-12, equal_nan=True)


@pytest.mark.parametrize("name", NAMES[:12])
def test_missing_eligibility_and_scale_invariance(name, seat_data):
    data, _ = seat_data
    dates = data.dates[:100]
    f = factor(name)
    before = f.compute(data, dates, data.universe)
    for field in data.fields:
        data.fields[field] *= 7
    after = f.compute(data, dates, data.universe)
    np.testing.assert_allclose(before, after, atol=2e-10, rtol=2e-10, equal_nan=True)
    data._factor_eligibility.loc[dates[60], "RB"] = False
    result = f.compute(data, dates, data.universe)
    assert pd.isna(result.loc[dates[60], "RB"])


def test_peers_exclude_self_unknown_groups_and_ineligible_values():
    data = DailyData(110)
    f = factor("sector_direction_persistence")
    data._factor_eligibility["J"] = False
    before = f.compute(data, data.dates, data.universe)
    data.fields["close"]["J"] *= np.exp(np.arange(110) / 7)
    after = f.compute(data, data.dates, data.universe)
    pd.testing.assert_frame_equal(before, after)
    assert f.compute(data, data.dates, ["RB"]).isna().all().all()


@pytest.mark.parametrize("name", NAMES[-2:])
def test_seasonality_excludes_current_year_and_needs_three_prior_years(name):
    data = DailyData()
    f = factor(name)
    a = f.compute(data, data.dates, data.universe)
    assert a.loc[a.index.year <= 2020].isna().all().all()
    current = data.dates.year == 2023
    data.fields["close"].loc[current] *= np.linspace(1, 2, current.sum())[:, None]
    b = f.compute(data, data.dates, data.universe)
    pd.testing.assert_frame_equal(a, b)


@pytest.mark.parametrize("mode", ["reference", "shadow", "native"])
def test_daily_rust_expression_modes_preserve_values(mode, monkeypatch):
    data = DailyData(80)
    f = factor("oi_stable_price_trend")
    monkeypatch.setenv("MF_FACTOR_KERNEL_MODE", "reference")
    expected = f.compute(data, data.dates, data.universe)
    monkeypatch.setenv("MF_FACTOR_KERNEL_MODE", mode)
    pd.testing.assert_frame_equal(expected, f.compute(data, data.dates, data.universe))


@pytest.mark.parametrize("name", NAMES[:12])
def test_all_daily_formulas_against_independent_pandas_oracle(name, seat_data):
    data, raw = seat_data
    dates = data.dates[:120]
    close, oi, volume = (data.fields[k].loc[dates] for k in ("close", "oi", "volume"))
    r, g = np.log(close).diff(), np.log(oi).diff()
    mean = lambda x: x.rolling(20).mean()
    rms = lambda x: np.sqrt(mean(x.pow(2)))
    ratio = lambda n, d: n.div(d.where(d > 1e-12)).mask(d.le(1e-12) & n.notna() & d.ge(0), 0.)
    if name == "oi_path_efficiency":
        expected = ratio(mean(g), mean(g.abs()))
    elif name == "oi_stable_price_trend":
        expected = ratio(mean(r), rms(r)) / (1 + oi.rolling(20).std(ddof=0) / mean(oi))
    elif name.startswith("seat_"):
        grouped = raw.loc[raw._ts.isin(dates)].groupby(["_ts", "_root"])
        l, s = grouped.long_position.sum(), grouped.short_position.sum()
        hl = grouped.long_position.agg(lambda x: np.sum((x / x.sum())**2)).unstack()
        hs = grouped.short_position.agg(lambda x: np.sum((x / x.sum())**2)).unstack()
        if name == "seat_concentration_pressure":
            expected = mean(hl.diff() - hs.diff())
        else:
            expected = mean(((l - s) / (l + s)).unstack().shift(1) * -((hl + hs) / 2).diff())
    elif name == "liquidation_reversal":
        a = (volume / mean(volume).shift(1)).clip(0, 3)
        expected = -ratio(mean(np.sign(r) * (-g).clip(lower=0) * a), mean(g.abs() * a))
    elif name == "oi_price_response_asymmetry":
        up, down = g.gt(0), g.lt(0)
        nu, nd = up.rolling(20).sum(), down.rolling(20).sum()
        diff = r.where(up, 0).rolling(20).sum() / nu - r.where(down, 0).rolling(20).sum() / nd
        expected = ratio(diff, rms(r)).where((nu >= 3) & (nd >= 3) & (g.notna().rolling(20).sum() == 20))
    else:
        m, s = r * np.nan, r * np.nan
        for col in data.universe:
            peers = [p for p in data.universe if p != col and intraday._SECTOR_MAP[p] == intraday._SECTOR_MAP[col]]
            m[col], s[col] = r[peers].mean(axis=1), r[peers].std(axis=1, ddof=0)
        if name in ("sector_dispersion_repair", "sector_dispersion_breakout"):
            z = ratio(r - m, s + rms(r))
            change = ratio(s - s.shift(1), s + s.shift(1))
            expected = (mean(z * change.clip(lower=0)) if name.endswith("breakout")
                        else -mean(z * (-change).clip(lower=0)))
        elif name == "sector_direction_persistence":
            expected = mean(np.sign(r) * np.sign(m)) * ratio(mean(r), rms(r))
        elif name == "sector_downside_decoupling":
            w = (-m).clip(lower=0)
            expected = ratio(mean(r * w), mean(w) * rms(r)).where(m.lt(0).rolling(20).sum() >= 5)
        elif name == "sector_delayed_response":
            expected = mean(r.rolling(20).corr(m.shift(1)).shift(1) * ratio(m - r, rms(r)))
        else:
            expected = mean(m.rolling(20).corr(r.shift(1)).shift(1) * ratio(r, rms(r)))
    actual = factor(name).compute(data, dates, data.universe)
    expected = expected.reindex(index=dates, columns=data.universe).shift(1)
    np.testing.assert_array_equal(actual.isna(), expected.isna())
    np.testing.assert_allclose(actual, expected, atol=2e-10, rtol=2e-10, equal_nan=True)


@pytest.mark.parametrize("name", NAMES[-2:])
def test_seasonal_value_against_completed_year_oracle(name):
    data = DailyData()
    f = factor(name)
    r = np.log(data.fields["close"]).diff()
    means = r.groupby([r.index.year, r.index.month]).mean()
    values = pd.DataFrame(np.nan, index=data.dates, columns=data.universe)
    for year in data.dates.year.unique():
        history = means.loc[(means.index.get_level_values(0) >= year - 5)
                            & (means.index.get_level_values(0) < year)]
        if history.empty:
            continue
        for month in range(1, 13):
            sample = history.loc[history.index.get_level_values(1) == month]
            if len(sample) < 3:
                continue
            score = (sample.mean() / np.sqrt(sample.pow(2).mean()) * np.sign(sample).mean().abs()
                     if f.CONSENSUS else sample.mean() - history.mean())
            values.loc[(values.index.year == year) & (values.index.month == month)] = score.to_numpy()
    expected = values.rolling(20).mean().shift(1)
    np.testing.assert_allclose(f.compute(data, data.dates, data.universe), expected,
                               atol=2e-12, equal_nan=True)


def test_missing_inputs_and_zero_volume_do_not_create_signal():
    data = DailyData(90)
    data.fields["volume"].iloc[:, :] = 0.
    assert factor("liquidation_reversal").compute(data, data.dates, data.universe).isna().all().all()
    for frame in data.fields.values():
        frame.iloc[50] = np.nan
    for name in ("oi_path_efficiency", "oi_stable_price_trend", "oi_price_response_asymmetry"):
        out = factor(name).compute(data, data.dates, data.universe)
        assert out.iloc[51:72].isna().all().all()


GAP_NAMES = (
    "oi_pullback_recovery", "oi_cost_retention",
    "seat_concentration_followthrough", "seat_concentration_rotation",
    "sector_downside_oi_resilience", "sector_upside_oi_confirmation",
    "sector_dispersion_lag_response", "sector_residual_sign_balance",
    "overnight_oi_repair", "session_oi_transfer",
    "drawdown_oi_resilience", "recovery_oi_retention",
)


@pytest.fixture
def gap_data(seat_data):
    data, raw = seat_data
    data.dates = data.dates[:220]
    rng = np.random.default_rng(732)
    data.fields["open"] = data.fields["close"].shift(1) * np.exp(
        rng.normal(0, .008, data.fields["close"].shape))
    data.fields["open"].iloc[0] = data.fields["close"].iloc[0]
    return data, raw


def gap_oracle(name, data, raw):
    dates, universe = data.dates, data.universe
    fields = {k: v.reindex(index=dates, columns=universe).where(
        data._factor_eligibility.reindex(index=dates, columns=universe))
        for k, v in data.fields.items()}
    fields = {k: v.where(np.isfinite(v) & (v > 0)) for k, v in fields.items()}
    p, g = np.log(fields["close"]), np.log(fields["oi"]).diff()
    r = p.diff()
    total = lambda v: v.rolling(20).sum()

    def divide(n, d):
        out = n / d.where(d > 1e-12)
        return out.mask(
            np.isfinite(n) & np.isfinite(d) & (d >= 0) & (d <= 1e-12), 0.)

    def response(v, w):
        w = w.where(np.isfinite(v) & np.isfinite(w) & (w >= 0))
        d = (total(w) * total(w * v.pow(2))).clip(lower=0).pow(.5)
        return divide(total(w * v), d).where(total(w.gt(0).where(w.notna()).astype(float)) >= 3)

    if name.startswith("seat_"):
        subset = raw.loc[raw._ts.isin(dates)]
        seats = subset.groupby(["_ts", "_root", "seat_name"])[["long_position", "short_position"]].sum()
        sums = seats.groupby(level=[0, 1]).sum()
        square = seats.pow(2).groupby(level=[0, 1]).sum()
        lh = (square.long_position / sums.long_position.pow(2)).unstack().reindex(columns=universe)
        sh = (square.short_position / sums.short_position.pow(2)).unstack().reindex(columns=universe)
        b = ((sums.long_position - sums.short_position) / (sums.long_position + sums.short_position)).unstack().reindex(columns=universe)
        a = lh - sh
        if name.endswith("followthrough"):
            out = response(b.diff(), a.diff().shift(1).clip(lower=0))
        else:
            u, v = a.shift(1) * b.diff(), b.shift(1) * a.diff()
            out = divide(total(u - v), total(u.abs() + v.abs()))
    elif name.startswith("sector_"):
        mean, std = r * np.nan, r * np.nan
        for root in universe:
            sector = intraday._SECTOR_MAP.get(root)
            peers = [s for s in universe if s != root and sector not in (None, "other")
                     and intraday._SECTOR_MAP.get(s) == sector]
            if len(peers) >= 2:
                valid = r[peers].count(axis=1) >= 2
                mean[root] = r[peers].mean(axis=1).where(valid)
                std[root] = r[peers].std(axis=1, ddof=0).where(valid)
        e = r - mean
        if name == "sector_downside_oi_resilience":
            out = response(e, (-mean).clip(lower=0) * g.clip(lower=0))
        elif name == "sector_upside_oi_confirmation":
            out = response(e, mean.clip(lower=0) * g.clip(lower=0))
        elif name == "sector_dispersion_lag_response":
            out = response(e, std.diff().shift(1).clip(lower=0) * (-e.shift(1)).clip(lower=0))
        else:
            out = divide(e, std + e.abs()).rolling(20).mean()
    elif name == "oi_pullback_recovery":
        out = response(r, (-r.shift(1)).clip(lower=0) * g.shift(1).clip(lower=0))
    elif name == "oi_cost_retention":
        a = g.clip(lower=0)
        anchor = total(a * p) / total(a)
        out = divide(p - anchor, r.pow(2).rolling(20).mean().pow(.5)).where(total(a.gt(0).where(a.notna()).astype(float)) >= 3)
    elif name in ("overnight_oi_repair", "session_oi_transfer"):
        n = np.log(fields["open"]) - p.shift(1)
        d = p - np.log(fields["open"])
        out = (response(d, (-n).clip(lower=0) * g.clip(lower=0)) if name == "overnight_oi_repair"
               else divide(total(g * (d - n)), total(g.abs() * (d.abs() + n.abs()))))
    elif name == "drawdown_oi_resilience":
        out = response(r, (p.rolling(20).max() - p).shift(1) * g.clip(lower=0))
    else:
        oi = fields["oi"]
        w = (p - p.rolling(20).min()).shift(1) * (oi / oi.rolling(20).max()).shift(1)
        out = response(r, w)
    return out.shift(1).reindex(index=dates, columns=universe).where(data._factor_eligibility)


@pytest.mark.parametrize("name", GAP_NAMES)
def test_gap_formulas_and_frequencies_against_independent_oracle(name, gap_data):
    data, raw = gap_data
    f = factor(name)
    assert f.input_bar_frequency == f.signal_frequency == f.frequency == "daily"
    assert f.validation_horizons == (5, 10, 20)
    assert f.expected_direction == 1
    assert all(section in f.__doc__ for section in ("【用法说明】", "【公式】", "【含义】", "方向:", "⚠"))
    actual = f.compute(data, data.dates, data.universe)
    expected = gap_oracle(name, data, raw)
    assert actual.notna().sum().sum() > 20
    np.testing.assert_array_equal(actual.isna(), expected.isna())
    np.testing.assert_allclose(actual, expected, atol=2e-9, rtol=2e-9, equal_nan=True)
    if name != "oi_cost_retention":
        assert np.nanmax(np.abs(actual)) <= 1 + 1e-10


@pytest.mark.parametrize("name", GAP_NAMES)
def test_gap_prefix_shift_chunk_and_no_mutation(name, gap_data, monkeypatch):
    data, raw = gap_data
    f = factor(name)
    originals = {k: v.copy() for k, v in data.fields.items()}
    actual = f.compute(data, data.dates, data.universe)
    prefix = f.compute(data, data.dates[:150], data.universe)
    pd.testing.assert_frame_equal(actual.iloc[:150], prefix)
    chunk = f.compute(data, data.dates[-180:], data.universe).iloc[-50:]
    np.testing.assert_allclose(actual.iloc[-50:], chunk, atol=2e-9, rtol=2e-9, equal_nan=True)
    for key, value in originals.items():
        pd.testing.assert_frame_equal(data.fields[key], value)
        data.fields[key].loc[data.dates[150]:] *= 1.7
    raw.loc[raw._ts >= data.dates[150], "long_position"] *= 1.7
    changed = f.compute(data, data.dates, data.universe)
    pd.testing.assert_frame_equal(actual.iloc[:151], changed.iloc[:151])
    assert f.compute(None, [], data.universe).empty
    assert f.compute(None, data.dates, []).empty
    for mode in ("reference", "shadow", "native"):
        monkeypatch.setenv("MF_FACTOR_KERNEL_MODE", mode)
        pd.testing.assert_frame_equal(changed, f.compute(data, data.dates, data.universe))


@pytest.mark.parametrize("name", [n for n in GAP_NAMES if not n.startswith("seat_")])
def test_gap_units_missing_and_eligibility(name, gap_data):
    data, raw = gap_data
    f = factor(name)
    expected = f.compute(data, data.dates, data.universe)
    for key, value in data.fields.items():
        value *= 13 if key in ("open", "close") else 7
    scaled = f.compute(data, data.dates, data.universe)
    np.testing.assert_allclose(expected, scaled, atol=2e-8, rtol=2e-8, equal_nan=True)
    data.fields["close"].loc[data.dates[110], "RB"] = np.nan
    data._factor_eligibility.loc[:, "I"] = False
    actual = f.compute(data, data.dates, data.universe)
    assert actual["I"].isna().all()
    assert actual.loc[data.dates[112:131], "RB"].isna().all()
    oracle = gap_oracle(name, data, raw)
    np.testing.assert_array_equal(actual.isna(), oracle.isna())
    np.testing.assert_allclose(actual, oracle, atol=2e-8, rtol=2e-8, equal_nan=True)


def test_gap_event_missing_not_zero_and_sparse_not_admitted():
    import polars as pl
    response = pl.col("r")
    w = pl.col("w")
    for weights in ([0.] * 25, [1.] * 2 + [0.] * 23):
        out = pl.DataFrame({"r": [1.] * 25, "w": weights}).select(intraday._supplement_event_response(response, w))
        assert out.to_series().is_null().all()
    frame = pl.DataFrame({"r": [0.] * 25, "w": [1.] * 25})
    assert frame.select(intraday._supplement_event_response(response, w)).to_series()[-1] == 0
    frame = pl.DataFrame({"r": [1.] * 24 + [None], "w": [1.] * 25})
    assert frame.select(intraday._supplement_event_response(response, w)).to_series()[-1] is None


def test_gap_open_invalid_and_seat_missing(gap_data):
    data, raw = gap_data
    data.fields["open"].loc[data.dates[100], "RB"] = 0
    raw.loc[(raw._ts == data.dates[100]) & (raw._root == "RB"), "long_position"] = np.nan
    for name in ("overnight_oi_repair", "session_oi_transfer", *GAP_NAMES[2:4]):
        out = factor(name).compute(data, data.dates, data.universe)
        assert out.loc[data.dates[102:121], "RB"].isna().all()


def test_intraday_numbers_cover_each_registration_once():
    import ast
    import re
    from pathlib import Path
    from core.registry import get, list_registered
    source = Path(intraday.__file__).read_text(encoding="utf-8-sig")
    registrations = []
    for node in ast.parse(source).body:
        if isinstance(node, ast.ClassDef):
            for decorator in node.decorator_list:
                if (isinstance(decorator, ast.Call)
                        and isinstance(decorator.func, ast.Name)
                        and decorator.func.id == "register_factor"):
                    registrations.append((decorator.lineno, decorator.args[0].value, node))
    headers = list(re.finditer(r"^# (\d+)\. (\S+)", source, re.M))
    assert sorted(int(m[1]) for m in headers) == list(range(1, len(registrations) + 1))
    assert not re.search(r"^# [KV]\d+\.|登记\d+", source, re.M)
    mapped = []
    for header in headers:
        line = source.count("\n", 0, header.start()) + 1
        _, name, node = next(row for row in registrations if row[0] > line)
        mapped.append(name)
        if 700 <= int(header[1]) <= 733:
            assert header[2] == name
            doc = ast.get_docstring(node)
            for marker in ("【用法说明】", "【公式】", "【含义】", "方向:", "⚠"):
                assert marker in doc
            assert getattr(get("factor", name), "expected_direction", None) is None
    registered = {name for name in list_registered("factor")["factor"]
                  if get("factor", name).__module__ == intraday.__name__}
    assert len(mapped) == len(set(mapped))
    assert set(mapped) == registered
    for name in GAP_NAMES:
        assert "intraday_" + name + "_20d" in registered
