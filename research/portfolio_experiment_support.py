"""Shared causal inputs for historical portfolio experiments.

The project originally kept these helpers inside three dated ``scripts/exp*``
entry points.  Current research workflows import them as a library, so the
reusable definitions live here without command-line side effects or obsolete
experiment drivers.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from core.config import load_config
from core.period import iter_overlapping_chunks
from data.manager import DataManager
from factors.engine import FactorComputationError, FactorEngine
from optimization.costs import SimpleFuturesCost
from optimization.factor_weighting import rank_information_coefficients
from strategies.combined import FACTORS as PRODUCTION_10F, SECTOR_MAP, UNIVERSE38

# Historical six-factor baseline.  This is research evidence, not the current
# production factor set in strategies.combined.
BASELINE_6F = {
    "intraday_jump_intensity_20d": -1,
    "intraday_price_peak_count_20d": 1,
    "intraday_realised_skewness_20d": 1,
    "intraday_dtws_20d": 1,
    "intraday_drip_stone_20d": -1,
    "intraday_peak_ridge_ratio_20d": -1,
}

# Fixed audit sets shared by internal historical studies and external
# comparators.  Keeping the definitions here prevents research workflows from
# depending on an external-strategy command module.
FACTORS_6F = list(BASELINE_6F)
FACTORS_14F = FACTORS_6F + [
    "intraday_ma_count_bullish_20d",
    "intraday_torrent_down_20d",
    "intraday_lowest_time_20d",
    "intraday_term_slope_20d",
    "intraday_open_close_volume_ratio_20d",
    "intraday_seat_long_short_seat_ratio_20d",
    "intraday_turnover_velocity_20d",
    "intraday_price_delay_20d",
]
FACTORS_13F = [
    name for name in FACTORS_14F
    if name != "intraday_ma_count_bullish_20d"
]
FACTORS_10F = list(PRODUCTION_10F)
FACTORS_8F = [
    "intraday_ma_count_bullish_20d",
    "intraday_price_peak_count_20d",
    "intraday_lowest_time_20d",
    "intraday_basis_momentum_20d",
    "intraday_price_delay_20d",
    "intraday_torrent_down_20d",
    "intraday_open_close_volume_ratio_20d",
    "intraday_zero_ret_freq_20d",
]

VALIDATED_47 = [
    "intraday_zero_ret_freq_20d", "intraday_open_close_drift_20d",
    "intraday_volatility_clustering_20d", "intraday_oi_vol_corr_daily_20d",
    "intraday_oi_time_centroid_20d", "intraday_wash_trade_20d",
    "intraday_settle_position_20d", "intraday_cross_vol_20d",
    "intraday_amihud_vol_ratio_20d", "intraday_oi_skew_stability_20d",
    "intraday_depth_trend_20d", "intraday_open_close_volume_ratio_20d",
    "intraday_oi_quantile_range_20d", "intraday_settle_gap_20d",
    "intraday_amihud_trend_20d", "intraday_price_delay_20d",
    "intraday_overnight_absorption_20d", "intraday_session_symmetry_20d",
    "intraday_lowest_time_20d", "intraday_oi_peak_ridge_ratio_20d",
    "intraday_seat_long_short_seat_ratio_20d", "intraday_volume_rank_ratio_20d",
    "intraday_herding_20d", "intraday_volume_time_shape_20d",
    "intraday_extreme_freq_balance_20d", "intraday_term_vol_ratio_20d",
    "intraday_turnover_velocity_20d", "intraday_settle_close_basis_20d",
    "intraday_settle_drift_20d", "intraday_settle_vol_ratio_20d",
    "intraday_term_slope_20d", "intraday_oi_log_change_vol_20d",
    "intraday_term_roll_yield_20d", "intraday_term_vol_spread_20d",
    "intraday_oi_vol_price_corr_20d", "intraday_term_spread_vol_20d",
    "intraday_rollover_basis_gap_20d", "intraday_basis_momentum_20d",
    "intraday_roll_yield_dualscore_20d",
    "intraday_roll_dualscore_consistency_20d",
    "intraday_amihud_resid_vol_20d", "intraday_volume_oi_price_confirm_20d",
    "intraday_amihud_cross_z_20d", "intraday_overnight_gap_reaction_20d",
    "intraday_false_breakout_retrace_20d", "intraday_ma_count_bullish_20d",
    "intraday_rv_compression_breakout_20d",
]

FACTOR_DIRECTIONS = {
    "intraday_jump_intensity_20d": -1, "intraday_price_peak_count_20d": 1,
    "intraday_realised_skewness_20d": 1, "intraday_dtws_20d": 1,
    "intraday_drip_stone_20d": -1, "intraday_peak_ridge_ratio_20d": -1,
    "intraday_zero_ret_freq_20d": -1, "intraday_open_close_drift_20d": 1,
    "intraday_volatility_clustering_20d": -1,
    "intraday_oi_vol_corr_daily_20d": 1, "intraday_oi_time_centroid_20d": -1,
    "intraday_wash_trade_20d": 1, "intraday_settle_position_20d": -1,
    "intraday_cross_vol_20d": 1, "intraday_amihud_vol_ratio_20d": 1,
    "intraday_oi_skew_stability_20d": -1, "intraday_depth_trend_20d": 1,
    "intraday_open_close_volume_ratio_20d": -1,
    "intraday_oi_quantile_range_20d": -1, "intraday_settle_gap_20d": 1,
    "intraday_amihud_trend_20d": 1, "intraday_price_delay_20d": -1,
    "intraday_overnight_absorption_20d": 1,
    "intraday_session_symmetry_20d": -1, "intraday_lowest_time_20d": 1,
    "intraday_oi_peak_ridge_ratio_20d": -1,
    "intraday_seat_long_short_seat_ratio_20d": 1,
    "intraday_volume_rank_ratio_20d": 1, "intraday_herding_20d": 1,
    "intraday_volume_time_shape_20d": 1,
    "intraday_extreme_freq_balance_20d": -1,
    "intraday_term_vol_ratio_20d": 1, "intraday_turnover_velocity_20d": 1,
    "intraday_settle_close_basis_20d": -1, "intraday_settle_drift_20d": 1,
    "intraday_settle_vol_ratio_20d": 1, "intraday_term_slope_20d": 1,
    "intraday_oi_log_change_vol_20d": -1,
    "intraday_term_roll_yield_20d": 1, "intraday_term_vol_spread_20d": -1,
    "intraday_oi_vol_price_corr_20d": -1,
    "intraday_term_spread_vol_20d": 1,
    "intraday_rollover_basis_gap_20d": 1, "intraday_basis_momentum_20d": 1,
    "intraday_roll_yield_dualscore_20d": -1,
    "intraday_roll_dualscore_consistency_20d": -1,
    "intraday_amihud_resid_vol_20d": 1,
    "intraday_volume_oi_price_confirm_20d": 1,
    "intraday_amihud_cross_z_20d": 1,
    "intraday_overnight_gap_reaction_20d": 1,
    "intraday_false_breakout_retrace_20d": 1,
    "intraday_ma_count_bullish_20d": 1,
    "intraday_rv_compression_breakout_20d": -1,
}

NEW_VALIDATED_21 = [
    "intraday_lead_amount_surge_20d", "intraday_price_diff_autocorr_20d",
    "intraday_diff_autocorr_long_20d", "intraday_order_flow_autocorr_20d",
    "intraday_vp_corr_high_freq_20d", "intraday_peak_moment_count_20d",
    "intraday_order_flow_memory_20d", "intraday_jump_ret_follow_ratio_20d",
    "intraday_neg_ret_illiq_20d", "intraday_ret_extreme_magnitude_20d",
    "intraday_peak_ridge_coherence_20d", "intraday_smart_money_v4_vol_20d",
    "intraday_torrent_down_20d", "intraday_jump_amount_lagcorr_20d",
    "intraday_csad_sigma120_20d", "intraday_vol_bucket_entropy_20d",
    "intraday_trajectory_illiq_20d", "intraday_vol_flow_vol_20d",
    "intraday_price_peak_interval_std_20d",
    "intraday_following_price_confirm_20d", "intraday_flow_ret_resid_vol_20d",
]

NEW_FACTOR_DIRECTIONS = {
    "intraday_lead_amount_surge_20d": -1,
    "intraday_price_diff_autocorr_20d": -1,
    "intraday_diff_autocorr_long_20d": 1,
    "intraday_order_flow_autocorr_20d": 1,
    "intraday_vp_corr_high_freq_20d": 1,
    "intraday_peak_moment_count_20d": -1,
    "intraday_order_flow_memory_20d": 1,
    "intraday_jump_ret_follow_ratio_20d": 1,
    "intraday_neg_ret_illiq_20d": -1,
    "intraday_ret_extreme_magnitude_20d": 1,
    "intraday_peak_ridge_coherence_20d": -1,
    "intraday_smart_money_v4_vol_20d": 1,
    "intraday_torrent_down_20d": -1,
    "intraday_jump_amount_lagcorr_20d": 1,
    "intraday_csad_sigma120_20d": 1,
    "intraday_vol_bucket_entropy_20d": 1,
    "intraday_trajectory_illiq_20d": -1,
    "intraday_vol_flow_vol_20d": 1,
    "intraday_price_peak_interval_std_20d": 1,
    "intraday_following_price_confirm_20d": 1,
    "intraday_flow_ret_resid_vol_20d": 1,
}


def configured_futures_cost_model() -> SimpleFuturesCost:
    """Load the formal, stateless futures cost policy used by comparisons."""
    config_path = Path(__file__).resolve().parents[1] / "config" / "intraday_backtest.yaml"
    costs = load_config(str(config_path)).costs
    if str(costs.type) != "simple_futures":
        raise ValueError("production-style research requires simple_futures costs")
    return SimpleFuturesCost(
        turnover_cost_rate=float(costs.turnover_cost_rate),
        annual_fee=float(costs.annual_fee),
        annual_roll_cost=float(costs.annual_roll_cost),
        periods_per_year=float(costs.periods_per_year),
        cost_stage=str(costs.cost_stage),
    )


def latest_local_date(data_manager=None) -> pd.Timestamp:
    """Read the newest date through the configured formal Parquet source."""
    if data_manager is None:
        data_manager = DataManager.from_config(
            load_config("config/intraday_backtest.yaml")
        )
    fetcher = getattr(data_manager.source, "fetch_latest_trade_date", None)
    if not callable(fetcher):
        raise NotImplementedError(
            "configured data source does not expose its latest trade date"
        )
    return pd.Timestamp(fetcher()).normalize()


class ExperimentEnvironment:
    """Shared data, factor, calendar and causal-risk caches for research."""

    def __init__(self, factors: dict[str, int] | None = None):
        self.cfg = load_config("config/intraday_backtest.yaml")
        self.data_manager = DataManager.from_config(self.cfg)
        self.cal = pd.DatetimeIndex(
            self.data_manager.get_calendar(
                pd.Timestamp("2015-12-01"),
                latest_local_date(self.data_manager),
            )
        )
        self.u = list(UNIVERSE38)
        self.engine = FactorEngine(self.data_manager)
        self.close = self.data_manager.get("close", self.cal, self.u)
        self.daily_ret, self.close_tradable = (
            self.data_manager.prepare_close_data(self.close)
        )
        self.factors = factors if factors is not None else dict(BASELINE_6F)
        self.sector_of = {
            symbol: sector
            for sector, members in SECTOR_MAP.items()
            for symbol in members
            if symbol in self.u
        }

class FactorPanelRunner:
    """Build the fixed 74-factor rank and IC panels used by research flows."""

    FACTOR_CHUNK_SIZE = 400
    # 当前日内库最长跨日依赖为 120 个交易日；留 128 日预热保证边界一致。
    FACTOR_CHUNK_OVERLAP = 128

    @classmethod
    def _iter_factor_chunks(cls, calendar):
        return iter_overlapping_chunks(
            calendar, cls.FACTOR_CHUNK_SIZE, cls.FACTOR_CHUNK_OVERLAP
        )

    @staticmethod
    def _compute_part(engine, names, dates, universe, already_valid):
        """Allow only expected leading-empty chunks for late-start factors."""
        try:
            return engine.compute_factors(
                names, dates, universe, parallel=False
            )
        except FactorComputationError:
            result = {}
            for name in names:
                try:
                    result.update(engine.compute_factors(
                        [name], dates, universe, parallel=False
                    ))
                except FactorComputationError as exc:
                    leading_empty = (
                        name not in already_valid
                        and isinstance(exc.__cause__, ValueError)
                        and str(exc.__cause__) == "factor output contains no finite values"
                    )
                    if not leading_empty:
                        raise
            return result

    def __init__(self, factor_names: list[str] | None = None):
        self.env = ExperimentEnvironment(BASELINE_6F)
        self.cal = self.env.cal
        self.u = self.env.u
        self.daily_ret = self.env.daily_ret
        self.close_tradable = self.env.close_tradable
        self._contract_schedule: pd.DataFrame | None = None
        self._contract_schedule_loaded = False
        all_factors = list(
            dict.fromkeys(
                factor_names
                if factor_names is not None
                else list(BASELINE_6F) + VALIDATED_47 + NEW_VALIDATED_21
            )
        )
        comp: dict[str, pd.DataFrame] = {}
        for target_dates, request_dates in self._iter_factor_chunks(self.cal):
            for offset in range(0, len(all_factors), 10):
                part = self._compute_part(
                    self.env.engine,
                    all_factors[offset:offset + 10],
                    request_dates,
                    self.u,
                    comp,
                )
                for name, values in part.items():
                    if name not in comp:
                        comp[name] = values.reindex(self.cal)
                    else:
                        comp[name].loc[target_dates] = values.reindex(target_dates)
        missing = [name for name in all_factors if name not in comp]
        if missing:
            raise FactorComputationError(
                f"factors never produced finite values: {missing}"
            )
        self.ranks: dict[str, pd.DataFrame] = {}
        for name in all_factors:
            if name not in comp:
                continue
            rank = comp[name].rank(axis=1, pct=True)
            direction = FACTOR_DIRECTIONS.get(
                name, NEW_FACTOR_DIRECTIONS.get(name, 1)
            )
            self.ranks[name] = rank if direction == 1 else (1 - rank)
        # 因子值T用于决策T，并预测下一交易日T+1收益。
        self.ic = rank_information_coefficients(
            self.ranks,
            self.daily_ret,
            minimum_cross_section=3,
        )
        # Full minute panels are no longer needed once daily factor/IC matrices
        # have been materialized. Keep only the persistent v4 source cache.
        from factors.library.intraday import clear_transient_data_caches

        clear_transient_data_caches()

    def get_contract_schedule(self) -> pd.DataFrame | None:
        """Load the causal concrete-contract schedule only when accounting needs it."""
        if self._contract_schedule_loaded:
            return self._contract_schedule
        data_manager = getattr(self.env, "data_manager", None)
        if data_manager is None:
            raise RuntimeError(
                "contract schedule must be loaded before detaching the data environment"
            )
        source = getattr(data_manager, "source", None)
        self._contract_schedule_loaded = True
        if source is None or not hasattr(source, "fetch_contract_schedule"):
            return None
        self._contract_schedule = source.fetch_contract_schedule(
            self.u,
            self.cal.min(),
            self.cal.max(),
        )
        return self._contract_schedule
