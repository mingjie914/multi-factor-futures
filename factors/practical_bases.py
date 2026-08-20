"""Economically motivated daily futures factor primitives.

These primitives deliberately use only daily OHLCV and open interest so they
can support the project's close-to-next-session workflow. They expand the
search space through distinct hypotheses rather than aliases of one indicator.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


PRACTICAL_BASES = frozenset({
    "log_momentum", "momentum_tstat", "trend_r2", "linear_trend_slope",
    "directional_consistency", "up_down_balance", "ema_acceleration",
    "price_acceleration", "breakout_distance", "return_autocorr",
    "upside_vol", "semivariance_balance", "vol_of_vol", "vol_term_spread",
    "atr_term_spread", "range_expansion", "gap_vol", "intraday_vol",
    "jump_intensity", "drawdown_speed", "volume_momentum",
    "volume_surprise", "volume_volatility", "oi_momentum", "oi_surprise",
    "oi_volatility", "turnover_oi", "turnover_trend",
    "price_oi_confirmation", "signed_volume_pressure", "median_return",
    "absolute_return_mean", "max_return", "min_return", "tail_spread",
    "quantile_asymmetry", "zero_return_ratio", "gap_reversal",
    "intraday_reversal", "candle_pressure", "volume_weighted_clv",
    "range_skew",
})


def _min_periods(window: int) -> int:
    return max(2, min(window, max(5, window // 2)))


def _ema(frame: pd.DataFrame, span: int) -> pd.DataFrame:
    return frame.ewm(
        span=span, adjust=False, min_periods=_min_periods(span)
    ).mean()


def _atr(
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    window: int,
) -> pd.DataFrame:
    previous = close.shift(1)
    true_range = pd.DataFrame(
        np.maximum.reduce([
            (high - low).to_numpy(dtype=float),
            (high - previous).abs().to_numpy(dtype=float),
            (low - previous).abs().to_numpy(dtype=float),
        ]),
        index=close.index,
        columns=close.columns,
    )
    return true_range.rolling(
        window, min_periods=_min_periods(window)
    ).mean()


def _rolling_time_correlation(
    frame: pd.DataFrame, window: int
) -> pd.DataFrame:
    """Rolling correlation with time, vectorized across instruments."""
    time_index = pd.Series(
        np.arange(len(frame), dtype=float), index=frame.index
    )
    min_periods = _min_periods(window)
    mean_x = time_index.rolling(window, min_periods=min_periods).mean()
    mean_x2 = time_index.pow(2).rolling(
        window, min_periods=min_periods
    ).mean()
    mean_y = frame.rolling(window, min_periods=min_periods).mean()
    mean_y2 = frame.pow(2).rolling(window, min_periods=min_periods).mean()
    mean_xy = frame.mul(time_index, axis=0).rolling(
        window, min_periods=min_periods
    ).mean()
    covariance = mean_xy - mean_y.mul(mean_x, axis=0)
    variance_x = (mean_x2 - mean_x.pow(2)).clip(lower=0.0)
    variance_y = (mean_y2 - mean_y.pow(2)).clip(lower=0.0)
    denominator = np.sqrt(
        variance_y.mul(variance_x, axis=0)
    ).replace(0.0, np.nan)
    return covariance / denominator


def compute_practical_base(
    base: str,
    params: dict,
    ohlcv: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Compute one practical primitive as a dates-by-instruments matrix."""
    if base not in PRACTICAL_BASES:
        raise ValueError(f"unknown practical base: {base}")

    close = ohlcv["close"].astype(float).where(lambda x: x > 0)
    open_ = ohlcv["open"].astype(float).where(lambda x: x > 0)
    high = ohlcv["high"].astype(float)
    low = ohlcv["low"].astype(float)
    volume = ohlcv["volume"].astype(float).where(lambda x: x > 0)
    window = int(params.get("window", 20))
    short = max(2, window // 3)
    long = max(window * 2, 20)
    min_periods = _min_periods(window)

    returns = ohlcv.get("_return_1d")
    if returns is None:
        returns = close.pct_change(fill_method=None)
    log_close = np.log(close)
    log_returns = log_close.diff()
    momentum = log_close.diff(window)
    rolling_std = log_returns.rolling(
        window, min_periods=min_periods
    ).std(ddof=0)

    if base == "log_momentum":
        return momentum
    if base == "momentum_tstat":
        count = log_returns.rolling(window, min_periods=min_periods).count()
        mean = log_returns.rolling(window, min_periods=min_periods).mean()
        return mean / rolling_std.replace(0.0, np.nan) * np.sqrt(count)
    if base in {"trend_r2", "linear_trend_slope"}:
        correlation = _rolling_time_correlation(log_close, window)
        if base == "trend_r2":
            return np.sign(momentum) * correlation.pow(2)
        time_std = pd.Series(
            np.arange(len(close), dtype=float), index=close.index
        ).rolling(window, min_periods=min_periods).std(ddof=0)
        price_std = log_close.rolling(
            window, min_periods=min_periods
        ).std(ddof=0)
        return correlation * price_std.div(time_std, axis=0)
    if base == "directional_consistency":
        return np.sign(log_returns).rolling(
            window, min_periods=min_periods
        ).mean()
    if base == "up_down_balance":
        positive = log_returns.clip(lower=0.0).rolling(
            window, min_periods=min_periods
        ).sum()
        negative = (-log_returns.clip(upper=0.0)).rolling(
            window, min_periods=min_periods
        ).sum()
        return (positive - negative) / (positive + negative).replace(0.0, np.nan)
    if base == "ema_acceleration":
        fast = _ema(log_close, short)
        middle = _ema(log_close, max(short + 1, window // 2))
        slow = _ema(log_close, window)
        return fast - 2.0 * middle + slow
    if base == "price_acceleration":
        return log_close.diff(short).diff(short)
    if base == "breakout_distance":
        upper = high.shift(1).rolling(
            window, min_periods=min_periods
        ).max()
        lower = low.shift(1).rolling(
            window, min_periods=min_periods
        ).min()
        upside = (close / upper.replace(0.0, np.nan) - 1.0).clip(lower=0.0)
        downside = (close / lower.replace(0.0, np.nan) - 1.0).clip(upper=0.0)
        return upside + downside
    if base == "return_autocorr":
        return log_returns.rolling(
            window, min_periods=min_periods
        ).corr(log_returns.shift(1))

    if base == "upside_vol":
        return log_returns.where(log_returns > 0.0, 0.0).rolling(
            window, min_periods=min_periods
        ).std(ddof=0)
    if base == "semivariance_balance":
        upside = log_returns.clip(lower=0.0).pow(2).rolling(
            window, min_periods=min_periods
        ).mean()
        downside = log_returns.clip(upper=0.0).pow(2).rolling(
            window, min_periods=min_periods
        ).mean()
        return (upside - downside) / (upside + downside).replace(0.0, np.nan)
    if base == "vol_of_vol":
        return log_returns.abs().rolling(
            window, min_periods=min_periods
        ).std(ddof=0)
    if base == "vol_term_spread":
        short_vol = log_returns.rolling(
            short, min_periods=_min_periods(short)
        ).std(ddof=0)
        long_vol = log_returns.rolling(
            long, min_periods=_min_periods(long)
        ).std(ddof=0)
        return short_vol / long_vol.replace(0.0, np.nan) - 1.0
    if base == "atr_term_spread":
        short_atr = _atr(high, low, close, short)
        long_atr = _atr(high, low, close, long)
        return short_atr / long_atr.replace(0.0, np.nan) - 1.0
    daily_range = (high - low) / close.replace(0.0, np.nan)
    if base == "range_expansion":
        short_range = daily_range.rolling(
            short, min_periods=_min_periods(short)
        ).mean()
        long_range = daily_range.rolling(
            long, min_periods=_min_periods(long)
        ).mean()
        return short_range / long_range.replace(0.0, np.nan) - 1.0
    gap_return = open_ / close.shift(1).replace(0.0, np.nan) - 1.0
    intraday_return = close / open_.replace(0.0, np.nan) - 1.0
    if base == "gap_vol":
        return gap_return.rolling(window, min_periods=min_periods).std(ddof=0)
    if base == "intraday_vol":
        return intraday_return.rolling(
            window, min_periods=min_periods
        ).std(ddof=0)
    if base == "jump_intensity":
        threshold = rolling_std.shift(1) * 2.0
        jumps = log_returns.abs().gt(threshold).astype(float)
        jumps = jumps.where(log_returns.notna() & threshold.notna())
        return jumps.rolling(window, min_periods=min_periods).mean()
    if base == "drawdown_speed":
        peak = close.rolling(window, min_periods=min_periods).max()
        drawdown = close / peak.replace(0.0, np.nan) - 1.0
        return drawdown.diff(short)

    if base == "volume_momentum":
        return np.log(volume).diff(window)
    if base == "volume_surprise":
        baseline = volume.shift(1).rolling(
            window, min_periods=min_periods
        ).median()
        return np.log(volume / baseline.replace(0.0, np.nan))
    volume_change = volume.pct_change(fill_method=None)
    if base == "volume_volatility":
        return volume_change.rolling(
            window, min_periods=min_periods
        ).std(ddof=0)

    if base in {
        "oi_momentum", "oi_surprise", "oi_volatility", "turnover_oi",
        "turnover_trend", "price_oi_confirmation",
    }:
        oi = ohlcv["oi"].astype(float).where(lambda x: x > 0)
        log_oi = np.log(oi)
        if base == "oi_momentum":
            return log_oi.diff(window)
        if base == "oi_surprise":
            baseline = oi.shift(1).rolling(
                window, min_periods=min_periods
            ).median()
            return np.log(oi / baseline.replace(0.0, np.nan))
        if base == "oi_volatility":
            return oi.pct_change(fill_method=None).rolling(
                window, min_periods=min_periods
            ).std(ddof=0)
        turnover = volume / oi.replace(0.0, np.nan)
        if base == "turnover_oi":
            return np.log(turnover).rolling(
                window, min_periods=min_periods
            ).mean()
        if base == "turnover_trend":
            recent = turnover.rolling(
                short, min_periods=_min_periods(short)
            ).mean()
            baseline = turnover.rolling(
                long, min_periods=_min_periods(long)
            ).mean()
            return recent / baseline.replace(0.0, np.nan) - 1.0
        return momentum * log_oi.diff(window)
    if base == "signed_volume_pressure":
        signed = np.sign(log_returns) * volume
        numerator = signed.rolling(window, min_periods=min_periods).sum()
        denominator = volume.rolling(window, min_periods=min_periods).sum()
        return numerator / denominator.replace(0.0, np.nan)

    if base == "median_return":
        return log_returns.rolling(window, min_periods=min_periods).median()
    if base == "absolute_return_mean":
        return log_returns.abs().rolling(window, min_periods=min_periods).mean()
    if base == "max_return":
        return log_returns.rolling(window, min_periods=min_periods).max()
    if base == "min_return":
        return log_returns.rolling(window, min_periods=min_periods).min()
    q10 = log_returns.rolling(window, min_periods=min_periods).quantile(0.10)
    q50 = log_returns.rolling(window, min_periods=min_periods).median()
    q90 = log_returns.rolling(window, min_periods=min_periods).quantile(0.90)
    if base == "tail_spread":
        return q90 + q10
    if base == "quantile_asymmetry":
        return (q90 + q10 - 2.0 * q50) / (q90 - q10).replace(0.0, np.nan)
    if base == "zero_return_ratio":
        zero = log_returns.abs().le(1e-8).astype(float).where(log_returns.notna())
        return zero.rolling(window, min_periods=min_periods).mean()
    if base == "gap_reversal":
        return -gap_return.rolling(window, min_periods=min_periods).sum()
    if base == "intraday_reversal":
        return -intraday_return.rolling(window, min_periods=min_periods).sum()
    price_range = (high - low).replace(0.0, np.nan)
    clv = (2.0 * close - high - low) / price_range
    if base == "candle_pressure":
        return clv.rolling(window, min_periods=min_periods).mean()
    if base == "volume_weighted_clv":
        numerator = (clv * volume).rolling(
            window, min_periods=min_periods
        ).sum()
        denominator = volume.rolling(window, min_periods=min_periods).sum()
        return numerator / denominator.replace(0.0, np.nan)
    if base == "range_skew":
        return daily_range.rolling(window, min_periods=min_periods).skew()

    raise AssertionError(f"unhandled practical base: {base}")
