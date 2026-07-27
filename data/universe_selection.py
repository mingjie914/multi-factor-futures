"""Point-in-time liquidity universe selection for futures research."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from core.sectors import sector_for


class LaggedLiquidityUniverseSelector:
    """Select a stable, sector-balanced core universe using only lagged data."""

    SUPPORTED_MODE = "lagged_liquidity_sector_balanced"

    def __init__(self, config) -> None:
        self.config = config
        self._validate_config()

    def _validate_config(self) -> None:
        cfg = self.config
        if str(cfg.mode) != self.SUPPORTED_MODE:
            raise ValueError(f"unsupported universe selection mode: {cfg.mode}")
        if int(cfg.lookback) < 2:
            raise ValueError("universe_selection.lookback must be at least 2")
        if int(cfg.min_listing_days) < int(cfg.lookback):
            raise ValueError(
                "universe_selection.min_listing_days must be >= lookback"
            )
        if not 0.0 < float(cfg.min_data_coverage) <= 1.0:
            raise ValueError(
                "universe_selection.min_data_coverage must be in (0, 1]"
            )
        if not 0 <= int(cfg.min_count) <= int(cfg.target_count) <= int(cfg.max_count):
            raise ValueError(
                "universe_selection counts must satisfy "
                "0 <= min_count <= target_count <= max_count"
            )
        if int(cfg.exit_buffer) < 0:
            raise ValueError("universe_selection.exit_buffer cannot be negative")
        weights = {str(k): float(v) for k, v in cfg.score_weights.items()}
        if set(weights) - {"amount", "oi"}:
            raise ValueError("universe selection supports only amount and oi weights")
        if any(value < 0 for value in weights.values()) or sum(weights.values()) <= 0:
            raise ValueError("universe selection score weights must be non-negative")
        for sector, minimum in cfg.sector_minimums.items():
            maximum = int(cfg.sector_maximums.get(sector, cfg.max_count))
            if int(minimum) < 0 or int(minimum) > maximum:
                raise ValueError(
                    f"invalid universe selection limits for sector {sector!r}"
                )

    @staticmethod
    def _trading_days(index) -> pd.DatetimeIndex:
        """Map Chinese futures night bars to their following trading day.

        The local minute store timestamps bars by wall-clock time.  A simple
        ``normalize()`` therefore turns Friday night/Saturday early-morning
        bars into extra calendar days and understates cross-contract coverage.
        Shifting by four hours puts both halves of the night session on its
        intended date; rolling weekends forward joins Friday night to Monday.
        """

        timestamps = pd.DatetimeIndex(index)
        if timestamps.tz is not None:
            timestamps = timestamps.tz_localize(None)
        shifted = (timestamps + pd.Timedelta(hours=4)).normalize()
        weekday = shifted.weekday
        weekend_offset = np.where(weekday == 5, 2, np.where(weekday == 6, 1, 0))
        return shifted + pd.to_timedelta(weekend_offset, unit="D")

    @classmethod
    def _daily_panel(cls, frame: pd.DataFrame, method: str) -> pd.DataFrame:
        result = frame.copy()
        result.index = pd.DatetimeIndex(result.index)
        if result.index.tz is not None:
            result.index = result.index.tz_localize(None)
        result = result.sort_index()
        day = cls._trading_days(result.index)
        grouped = result.groupby(day, sort=True)
        if method == "sum":
            return grouped.sum(min_count=1)
        if method == "last":
            return grouped.last()
        raise ValueError(f"unsupported daily aggregation: {method}")

    def _load_daily_inputs(self, data, dates, universe):
        amount = data.get("amount", dates, universe)
        oi = data.get("oi", dates, universe)
        close = data.get("close", dates, universe)
        required = {"amount": amount, "oi": oi, "close": close}
        missing = [
            name for name, frame in required.items()
            if frame.empty or not frame.notna().any().any()
        ]
        if missing:
            raise RuntimeError(
                "universe selection missing required data: " + ", ".join(missing)
            )
        return (
            self._daily_panel(amount, "sum"),
            self._daily_panel(oi, "last"),
            self._daily_panel(close, "last"),
        )

    def _decision_dates(self, daily_dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
        frequency = str(self.config.rebalance_freq).lower()
        if frequency == "daily":
            return daily_dates
        if frequency == "weekly":
            periods = daily_dates.to_period("W-FRI")
        elif frequency == "monthly":
            periods = daily_dates.to_period("M")
        else:
            raise ValueError(
                "universe_selection.rebalance_freq must be daily, weekly, or monthly"
            )
        first = pd.Series(daily_dates, index=daily_dates).groupby(periods).min()
        return pd.DatetimeIndex(first.to_numpy()).sort_values()

    def _rank_scores(
        self,
        amount_metric: pd.Series,
        oi_metric: pd.Series,
        eligible: pd.Series,
    ) -> pd.Series:
        names = eligible.index[eligible.fillna(False)]
        if len(names) == 0:
            return pd.Series(dtype=float)
        weights = {str(k): float(v) for k, v in self.config.score_weights.items()}
        total_weight = sum(weights.values())
        score = pd.Series(0.0, index=names, dtype=float)
        if weights.get("amount", 0.0) > 0:
            score += (
                amount_metric.reindex(names).rank(pct=True, method="average")
                * weights["amount"] / total_weight
            )
        if weights.get("oi", 0.0) > 0:
            score += (
                oi_metric.reindex(names).rank(pct=True, method="average")
                * weights["oi"] / total_weight
            )
        return score.dropna().sort_values(ascending=False, kind="stable")

    def _select_one_date(
        self, scores: pd.Series, incumbents: pd.Index
    ) -> pd.Index:
        if scores.empty:
            return pd.Index([], dtype=object)

        cfg = self.config
        if len(scores) < int(cfg.min_count):
            return pd.Index([], dtype=object)
        target = min(int(cfg.target_count), len(scores))
        maximum_count = min(int(cfg.max_count), len(scores))
        sectors = pd.Series(
            {instrument: sector_for(instrument) for instrument in scores.index},
            dtype=object,
        )
        global_rank = scores.rank(ascending=False, method="first")
        sector_rank = scores.groupby(sectors).rank(ascending=False, method="first")
        incumbent_set = set(incumbents).intersection(scores.index)
        buffered_incumbents = {
            instrument for instrument in incumbent_set
            if global_rank[instrument] <= target + int(cfg.exit_buffer)
            and sector_rank[instrument]
            <= int(cfg.sector_maximums.get(
                sectors[instrument], cfg.max_count
            )) + int(cfg.exit_buffer)
        }

        selected: list[str] = []
        selected_set: set[str] = set()
        sector_counts: dict[str, int] = {}

        def add(instrument: str) -> bool:
            if instrument in selected_set or len(selected) >= maximum_count:
                return False
            sector = str(sectors[instrument])
            maximum = int(cfg.sector_maximums.get(sector, cfg.max_count))
            if sector_counts.get(sector, 0) >= maximum:
                return False
            selected.append(instrument)
            selected_set.add(instrument)
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
            return True

        # Reserve minimum representation first. Incumbents inside the rank
        # buffer receive priority, then the current score decides.
        for sector, minimum in cfg.sector_minimums.items():
            sector_names = [name for name in scores.index if sectors[name] == sector]
            sector_names.sort(
                key=lambda name: (name in buffered_incumbents, scores[name]),
                reverse=True,
            )
            for name in sector_names:
                if sector_counts.get(str(sector), 0) >= int(minimum):
                    break
                add(name)

        for name in scores.index:
            if name in buffered_incumbents:
                add(name)
        if len(selected) < target:
            for name in scores.index:
                add(name)
                if len(selected) >= target:
                    break

        if len(selected) < int(cfg.min_count):
            return pd.Index([], dtype=object)
        return pd.Index(selected, dtype=object)

    def build_schedule(
        self, data, dates, universe
    ) -> dict[pd.Timestamp, pd.Index]:
        dates = pd.DatetimeIndex(dates)
        universe = pd.Index(universe)
        if len(dates) == 0 or len(universe) == 0:
            return {}
        if dates.tz is not None:
            dates = dates.tz_localize(None)

        amount, oi, close = self._load_daily_inputs(data, dates, universe)
        daily_dates = close.index.union(amount.index).union(oi.index).sort_values()
        amount = amount.reindex(index=daily_dates, columns=universe)
        oi = oi.reindex(index=daily_dates, columns=universe)
        close = close.reindex(index=daily_dates, columns=universe)

        lookback = int(self.config.lookback)
        min_periods = max(
            int(math.ceil(lookback * float(self.config.min_data_coverage))), 1
        )
        # Every selection input is shifted before rolling. A same-day volume
        # or OI shock can therefore affect only the next decision or later.
        lagged_amount = amount.where(amount > 0).shift(1)
        lagged_oi = oi.where(oi > 0).shift(1)
        lagged_available = close.notna().shift(1, fill_value=False)
        amount_metric = lagged_amount.rolling(
            lookback, min_periods=min_periods
        ).median()
        oi_metric = lagged_oi.rolling(
            lookback, min_periods=min_periods
        ).median()
        coverage = lagged_available.rolling(
            lookback, min_periods=lookback
        ).mean()
        listing_days = lagged_available.cumsum()
        eligible = (
            (coverage >= float(self.config.min_data_coverage))
            & (listing_days >= int(self.config.min_listing_days))
            & amount_metric.notna()
            & oi_metric.notna()
        )

        decisions = self._decision_dates(daily_dates)
        schedule: dict[pd.Timestamp, pd.Index] = {}
        incumbents = pd.Index([], dtype=object)
        for decision in decisions:
            scores = self._rank_scores(
                amount_metric.loc[decision],
                oi_metric.loc[decision],
                eligible.loc[decision],
            )
            incumbents = self._select_one_date(scores, incumbents)
            schedule[pd.Timestamp(decision)] = incumbents
        return schedule

    def build_eligibility_mask(
        self, data, dates, universe
    ) -> pd.DataFrame:
        dates = pd.DatetimeIndex(dates)
        universe = pd.Index(universe)
        mask = pd.DataFrame(False, index=dates, columns=universe, dtype=bool)
        if len(dates) == 0 or len(universe) == 0:
            return mask
        normalized_dates = dates.tz_localize(None) if dates.tz is not None else dates
        schedule = self.build_schedule(data, normalized_dates, universe)
        if not schedule:
            return mask

        trading_days = self._trading_days(normalized_dates)
        daily_dates = trading_days.unique().sort_values()
        daily_mask = pd.DataFrame(False, index=daily_dates, columns=universe)
        current = pd.Index([], dtype=object)
        for day in daily_dates:
            if day in schedule:
                current = schedule[day]
            if len(current):
                daily_mask.loc[day, current.intersection(universe)] = True
        expanded = daily_mask.reindex(trading_days)
        expanded.index = dates
        return expanded.astype(bool)


def build_universe_eligibility(data, dates, universe, config) -> pd.DataFrame | None:
    """Build a dynamic mask when enabled; otherwise preserve legacy behavior."""

    if config is None or not bool(getattr(config, "enabled", False)):
        return None
    return LaggedLiquidityUniverseSelector(config).build_eligibility_mask(
        data, dates, universe
    )
