"""Practical daily futures factor SPECs from robust factor families."""
from __future__ import annotations


_WINDOWS = (3, 5, 10, 20, 60, 120)
_TRANSFORMS = (
    "raw", "smooth", "rank", "vol_scaled", "stability", "compress",
)

_TSM = "Moskowitz, Ooi and Pedersen (2012), JFE, Time Series Momentum"
_VOL = "Yang and Zhang (2000), Journal of Business, OHLC volatility estimation"
_DOWNSIDE = "Ang, Chen and Xing (2006), RFS, Downside Risk"
_TAIL = "Harvey and Siddique (2000), Journal of Finance, Conditional Skewness"
_OI = "Hong and Yogo (2012), JFE, Futures Market Interest"
_LIQUIDITY = "Amihud (2002), Journal of Financial Markets, Illiquidity"
_COST = "Patton and Weller (2020), JFE, Costs of Trading Market Anomalies"

_BASES = {
    "log_momentum": ("trend", "Log-price momentum", _TSM),
    "momentum_tstat": ("trend", "Momentum mean divided by sampling risk", _TSM),
    "trend_r2": ("trend", "Signed R-squared of log price against time", _TSM),
    "linear_trend_slope": ("trend", "Rolling log-price trend slope", _TSM),
    "directional_consistency": ("trend", "Positive versus negative return frequency", _TSM),
    "up_down_balance": ("trend", "Positive versus negative cumulative return", _TSM),
    "ema_acceleration": ("trend", "Curvature across fast, medium and slow EMAs", _TSM),
    "price_acceleration": ("trend", "Change in short-horizon price velocity", _TSM),
    "breakout_distance": ("trend", "Distance beyond the prior price channel", _TSM),
    "return_autocorr": ("trend", "First-order return autocorrelation", _TSM),
    "upside_vol": ("risk_distribution", "Realized volatility of positive returns", _DOWNSIDE),
    "semivariance_balance": ("risk_distribution", "Upside-minus-downside semivariance", _DOWNSIDE),
    "vol_of_vol": ("risk_distribution", "Volatility of absolute daily returns", _VOL),
    "vol_term_spread": ("risk_distribution", "Short versus long realized volatility", _VOL),
    "atr_term_spread": ("risk_distribution", "Short versus long ATR spread", _VOL),
    "range_expansion": ("risk_distribution", "Recent versus long daily range", _VOL),
    "gap_vol": ("risk_distribution", "Volatility of close-to-open gaps", _VOL),
    "intraday_vol": ("risk_distribution", "Volatility of open-to-close returns", _VOL),
    "jump_intensity": ("risk_distribution", "Frequency of two-sigma price jumps", _VOL),
    "drawdown_speed": ("risk_distribution", "Drawdown deterioration or recovery speed", _DOWNSIDE),
    "volume_momentum": ("liquidity_flow", "Log change in trading volume", _COST),
    "volume_surprise": ("liquidity_flow", "Volume relative to its lagged median", _COST),
    "volume_volatility": ("liquidity_flow", "Volatility of volume growth", _COST),
    "oi_momentum": ("positioning_participation", "Log change in open interest", _OI),
    "oi_surprise": ("positioning_participation", "Open interest relative to lagged median", _OI),
    "oi_volatility": ("positioning_participation", "Volatility of open-interest growth", _OI),
    "turnover_oi": ("positioning_participation", "Volume-to-open-interest turnover", _OI),
    "turnover_trend": ("positioning_participation", "Recent versus long turnover", _OI),
    "price_oi_confirmation": ("positioning_participation", "Price and open-interest confirmation", _OI),
    "signed_volume_pressure": ("liquidity_flow", "Return-signed volume pressure", _COST),
    "median_return": ("risk_distribution", "Rolling median return", _TAIL),
    "absolute_return_mean": ("risk_distribution", "Mean absolute return", _VOL),
    "max_return": ("risk_distribution", "Largest rolling-window return", _TAIL),
    "min_return": ("risk_distribution", "Smallest rolling-window return", _TAIL),
    "tail_spread": ("risk_distribution", "Upper and lower return-tail balance", _TAIL),
    "quantile_asymmetry": ("risk_distribution", "Normalized return-quantile asymmetry", _TAIL),
    "zero_return_ratio": ("liquidity_flow", "Fraction of zero-return observations", _LIQUIDITY),
    "gap_reversal": ("reversal", "Contrarian close-to-open gap", _TSM),
    "intraday_reversal": ("reversal", "Contrarian open-to-close return", _TSM),
    "candle_pressure": ("pattern", "Average close location in the daily range", _TSM),
    "volume_weighted_clv": ("liquidity_flow", "Volume-weighted close location", _COST),
    "range_skew": ("risk_distribution", "Skewness of normalized daily ranges", _TAIL),
}

_OI_BASES = {
    "oi_momentum", "oi_surprise", "oi_volatility", "turnover_oi",
    "turnover_trend", "price_oi_confirmation",
}


def _params(window: int) -> dict:
    norm = (
        20 if window <= 10 else 60 if window <= 20
        else 120 if window <= 60 else 252
    )
    return {
        "window": window,
        "norm": norm,
        "lag": max(1, window // 5),
        "smooth": max(3, window // 5),
    }


def _make_specs() -> list[dict]:
    specs = []
    for base, (category, description, source) in _BASES.items():
        dependencies = ["open", "high", "low", "close", "volume"]
        if base in _OI_BASES:
            dependencies.append("oi")
        for window in _WINDOWS:
            for transform in _TRANSFORMS:
                specs.append({
                    "slug": f"{base}_{window}d_{transform}",
                    "name_cn": f"{window}周期{base}_{transform}",
                    "base": base,
                    "transform": transform,
                    "params": _params(window),
                    "category": category,
                    "frequency": "daily",
                    "research_tier": "candidate",
                    "expected_direction": "to_be_estimated",
                    "dependencies": dependencies,
                    "description": description,
                    "source": source,
                })
    return specs


SPECS = _make_specs()
