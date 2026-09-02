"""Shared causal inputs for historical portfolio experiments.

The project originally kept these helpers inside three dated ``scripts/exp*``
entry points.  Current research workflows import them as a library, so the
reusable definitions live here without command-line side effects or obsolete
experiment drivers.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from collections.abc import Mapping

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

VALIDATED_47_FACTORS = [
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
    "intraday_torrent_down_20d": -1,
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
    config_path = Path(__file__).resolve().parents[1] / "config" / "default.yaml"
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
    """Read the newest date through the configured formal data source."""
    if data_manager is None:
        data_manager = DataManager.from_config(
            load_config("config/default.yaml")
        )
    fetcher = getattr(data_manager.source, "fetch_latest_trade_date", None)
    if not callable(fetcher):
        raise NotImplementedError(
            "configured data source does not expose its latest trade date"
        )
    return pd.Timestamp(fetcher()).normalize()


class ExperimentEnvironment:
    """Shared data, factor, calendar and causal-risk caches for research."""

    def __init__(
        self,
        factors: dict[str, int] | None = None,
        *,
        start: str | pd.Timestamp = "2015-12-01",
        end: str | pd.Timestamp | None = None,
    ):
        self.cfg = load_config("config/default.yaml")
        self.data_manager = DataManager.from_config(self.cfg)
        end_date = (
            pd.Timestamp(end).normalize()
            if end is not None
            else latest_local_date(self.data_manager)
        )
        self.cal = pd.DatetimeIndex(
            self.data_manager.get_calendar(
                pd.Timestamp(start).normalize(),
                end_date,
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
    """Build rank and IC panels for an explicitly supplied factor universe."""

    FACTOR_CHUNK_SIZE = 400
    # 当前日内库最长跨日依赖为 120 个交易日；留 128 日预热保证边界一致。
    FACTOR_CHUNK_OVERLAP = 128
    CHECKPOINT_SCHEMA_VERSION = 1

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

    @staticmethod
    def _source_tree_fingerprint() -> str:
        root = Path(__file__).resolve().parents[1]
        digest = hashlib.sha256()
        files = [Path(__file__).resolve()]
        for directory in ("core", "data", "factors"):
            files.extend(sorted((root / directory).rglob("*.py")))
        for path in sorted(set(files), key=lambda item: item.as_posix()):
            digest.update(path.relative_to(root).as_posix().encode("utf-8"))
            digest.update(path.read_bytes())
        return digest.hexdigest()

    def _checkpoint_contract(self, factor_names: list[str]) -> dict:
        source = self.env.data_manager.source
        dates = pd.DatetimeIndex(self.cal).asi8
        source_fingerprint = getattr(
            source, "checkpoint_source_fingerprint", None
        )
        return {
            "schema_version": self.CHECKPOINT_SCHEMA_VERSION,
            "factors": factor_names,
            "universe": list(self.u),
            "calendar_sha256": hashlib.sha256(dates.tobytes()).hexdigest(),
            "calendar_start": pd.Timestamp(self.cal.min()).isoformat(),
            "calendar_end": pd.Timestamp(self.cal.max()).isoformat(),
            "source_fingerprint": (
                str(source_fingerprint(self.cal.min(), self.cal.max()))
                if callable(source_fingerprint)
                else str(getattr(
                    source, "cache_namespace", source.__class__.__name__
                ))
            ),
            "source_tree_sha256": self._source_tree_fingerprint(),
        }

    @staticmethod
    def _checkpoint_factor_file(directory: Path, factor: str) -> Path:
        suffix = hashlib.sha256(factor.encode("utf-8")).hexdigest()[:16]
        return directory / f"factor_{suffix}.parquet"

    def _load_factor_checkpoint(
        self, directory: Path | None, factor_names: list[str]
    ) -> tuple[dict[str, pd.DataFrame], dict | None]:
        if directory is None:
            return {}, None
        directory.mkdir(parents=True, exist_ok=True)
        contract = self._checkpoint_contract(factor_names)
        manifest_path = directory / "manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("contract") != contract:
                raise RuntimeError("factor-panel checkpoint contract changed")
        else:
            manifest = {"contract": contract, "completed": {}}
            temporary = manifest_path.with_suffix(f".{os.getpid()}.tmp")
            temporary.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            temporary.replace(manifest_path)

        loaded = {}
        for factor, filename in dict(manifest.get("completed") or {}).items():
            if factor not in factor_names:
                raise RuntimeError("factor-panel checkpoint contains an unknown factor")
            path = directory / str(filename)
            if not path.is_file():
                raise RuntimeError(f"factor-panel checkpoint is missing {path.name}")
            frame = pd.read_parquet(path)
            frame.index = pd.DatetimeIndex(frame.index)
            if not frame.index.equals(self.cal) or list(frame.columns) != list(self.u):
                raise RuntimeError(
                    f"factor-panel checkpoint axes changed for {factor}"
                )
            frame.index = self.cal
            loaded[factor] = frame
        return loaded, manifest

    def _save_factor_checkpoint(
        self,
        directory: Path | None,
        manifest: dict | None,
        factors: Mapping[str, pd.DataFrame],
    ) -> None:
        if directory is None or manifest is None or not factors:
            return
        completed = dict(manifest.get("completed") or {})
        for factor, frame in factors.items():
            path = self._checkpoint_factor_file(directory, factor)
            temporary = path.with_suffix(f".{os.getpid()}.tmp")
            frame.to_parquet(temporary)
            temporary.replace(path)
            completed[factor] = path.name
        manifest["completed"] = completed
        manifest_path = directory / "manifest.json"
        temporary = manifest_path.with_suffix(f".{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(manifest_path)

    def __init__(
        self,
        factor_names: list[str] | None = None,
        *,
        start: str | pd.Timestamp = "2015-12-01",
        end: str | pd.Timestamp | None = None,
        factor_directions: Mapping[str, int] | None = None,
        ic_horizon: int = 1,
        checkpoint_dir: str | Path | None = None,
    ):
        self.ic_horizon = int(ic_horizon)
        if self.ic_horizon < 1:
            raise ValueError("ic_horizon must be a positive daily-bar horizon")
        self.env = ExperimentEnvironment(BASELINE_6F, start=start, end=end)
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
                else list(BASELINE_6F) + VALIDATED_47_FACTORS + NEW_VALIDATED_21
            )
        )
        checkpoint_path = (
            Path(checkpoint_dir).resolve() if checkpoint_dir is not None else None
        )
        comp, checkpoint_manifest = self._load_factor_checkpoint(
            checkpoint_path, all_factors
        )
        from factors.library.intraday import clear_transient_data_caches

        self.checkpoint_loaded_factor_count = len(comp)
        for offset in range(0, len(all_factors), 10):
            batch_names = [
                name for name in all_factors[offset:offset + 10]
                if name not in comp
            ]
            if not batch_names:
                continue
            batch: dict[str, pd.DataFrame] = {}
            for target_dates, request_dates in self._iter_factor_chunks(self.cal):
                try:
                    part = self._compute_part(
                        self.env.engine,
                        batch_names,
                        request_dates,
                        self.u,
                        {**comp, **batch},
                    )
                    for name, values in part.items():
                        if name not in batch:
                            batch[name] = values.reindex(self.cal)
                        else:
                            batch[name].loc[target_dates] = values.reindex(target_dates)
                finally:
                    # A later chunk never reuses an earlier chunk's minute
                    # panels. Release them here while retaining batch outputs.
                    clear_transient_data_caches()
            comp.update(batch)
            self._save_factor_checkpoint(
                checkpoint_path, checkpoint_manifest, batch
            )
        self.computed_factor_count = len(comp) - self.checkpoint_loaded_factor_count
        missing = [name for name in all_factors if name not in comp]
        if missing:
            raise FactorComputationError(
                f"factors never produced finite values: {missing}"
            )
        self.raw_ranks: dict[str, pd.DataFrame] = {
            name: comp[name].rank(axis=1, pct=True)
            for name in all_factors
            if name in comp
        }
        self.ranks = self._oriented_ranks(
            all_factors, factor_directions=factor_directions
        )
        self._ic_returns = self._build_ic_returns()
        self.ic = rank_information_coefficients(
            self.ranks,
            self._ic_returns,
            minimum_cross_section=3,
        )
        # Full minute panels are no longer needed once daily factor/IC matrices
        # have been materialized. Keep only the persistent v4 source cache.
        clear_transient_data_caches()

    def _oriented_ranks(
        self,
        factor_names,
        *,
        factor_directions: Mapping[str, int] | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Apply one strategy's directions without recomputing factor values."""
        explicit_directions = {
            str(name): int(direction)
            for name, direction in dict(factor_directions or {}).items()
        }
        invalid_directions = {
            name: direction
            for name, direction in explicit_directions.items()
            if direction not in {-1, 1}
        }
        if invalid_directions:
            raise ValueError(
                f"factor directions must be ±1: {invalid_directions}"
            )
        missing = [name for name in factor_names if name not in self.raw_ranks]
        if missing:
            raise FactorComputationError(
                f"shared factor panel does not contain: {missing}"
            )
        ranks: dict[str, pd.DataFrame] = {}
        for name in factor_names:
            rank = self.raw_ranks[name]
            direction = explicit_directions.get(
                name,
                FACTOR_DIRECTIONS.get(
                    name, NEW_FACTOR_DIRECTIONS.get(name, 1)
                ),
            )
            ranks[name] = rank if direction == 1 else (1 - rank)
        return ranks

    def _build_ic_returns(self) -> pd.DataFrame:
        """Return the frozen future-return panel used by every shared view."""
        # 因子值 T 用于决策 T。默认口径预测 T+1；显式敏感性分支可
        # 使用同一因子面板对齐 T→T+h 的未来日度收益，但不会改写因子注册。
        if self.ic_horizon == 1:
            return self.daily_ret
        forward = self.env.data_manager.get_forward_returns(
            self.cal, self.u, period=self.ic_horizon
        )
        # rank_information_coefficients shifts its input by -1 because
        # the historical default input is a return-at-T+1 panel.
        return forward.shift(1)

    def for_factors(
        self,
        factor_names,
        *,
        factor_directions: Mapping[str, int] | None = None,
    ) -> "FactorPanelRunner":
        """Create a cheap strategy view over one already-computed factor panel."""
        view = copy.copy(self)
        names = list(dict.fromkeys(str(name) for name in factor_names))
        view.ranks = self._oriented_ranks(
            names, factor_directions=factor_directions
        )
        view.ic = rank_information_coefficients(
            view.ranks,
            self._ic_returns,
            minimum_cross_section=3,
        )
        return view

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
