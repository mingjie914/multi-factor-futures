"""Stateful sector-aware forecast selection policies."""
from __future__ import annotations

from typing import Mapping, Optional

import numpy as np
import pandas as pd


class SectorForecastSelector:
    """Hard/lagged Top-N or soft sector quota on predicted returns."""

    def __init__(
        self,
        mode: str = "hysteresis_top_n",
        top_n_per_side: int = 2,
        exit_buffer: int = 1,
        min_abs_forecast: float = 0.0,
        sector_map: Optional[Mapping[str, str]] = None,
    ):
        if mode not in {"hard_top_n", "hysteresis_top_n", "soft_quota"}:
            raise ValueError(f"unsupported asset selection mode {mode!r}")
        if top_n_per_side < 1 or exit_buffer < 0:
            raise ValueError("top_n_per_side must be positive and exit_buffer non-negative")
        self.mode = mode
        self.top_n_per_side = int(top_n_per_side)
        self.exit_buffer = int(exit_buffer)
        self.min_abs_forecast = float(min_abs_forecast)
        if sector_map is None:
            from core.sectors import SECTOR_MAP

            sector_map = SECTOR_MAP
        self.sector_map = dict(sector_map)
        self._selected: dict[tuple[str, str], set[str]] = {}
        self.last_diagnostics: dict = {}

    def reset(self) -> None:
        self._selected.clear()
        self.last_diagnostics = {}

    def apply(self, predicted: pd.Series, date=None) -> pd.Series:
        forecast = pd.Series(predicted, dtype=float).replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0.0)
        sectors: dict[str, list[str]] = {}
        for instrument in forecast.index:
            sector = self.sector_map.get(str(instrument), "other")
            sectors.setdefault(sector, []).append(instrument)

        if self.mode == "soft_quota":
            result = pd.Series(0.0, index=forecast.index)
            active = [
                (sector, assets)
                for sector, assets in sectors.items()
                if float(forecast.loc[assets].abs().sum()) > 0
            ]
            budget = 1.0 / len(active) if active else 0.0
            for _, assets in active:
                values = forecast.loc[assets]
                result.loc[assets] = values / float(values.abs().sum()) * budget
            self.last_diagnostics = {
                "date": str(pd.Timestamp(date)) if date is not None else None,
                "mode": self.mode,
                "selected_count": int((result != 0).sum()),
                "active_sectors": len(active),
            }
            return result

        result = pd.Series(0.0, index=forecast.index)
        selected_count = 0
        for sector, assets in sectors.items():
            values = forecast.loc[assets]
            for side, eligible in (
                ("long", values[values > self.min_abs_forecast].sort_values(ascending=False)),
                ("short", values[values < -self.min_abs_forecast].sort_values(ascending=True)),
            ):
                ranked = list(eligible.index)
                entrants = set(ranked[:self.top_n_per_side])
                key = (sector, side)
                if self.mode == "hysteresis_top_n":
                    retention = set(
                        ranked[:self.top_n_per_side + self.exit_buffer]
                    )
                    survivors = self._selected.get(key, set()) & retention
                    chosen = entrants | survivors
                else:
                    chosen = entrants
                self._selected[key] = set(chosen)
                if chosen:
                    result.loc[list(chosen)] = forecast.loc[list(chosen)]
                    selected_count += len(chosen)
        self.last_diagnostics = {
            "date": str(pd.Timestamp(date)) if date is not None else None,
            "mode": self.mode,
            "selected_count": selected_count,
            "active_sectors": len(sectors),
        }
        return result
