"""Fast post-optimization volatility-aware futures exposure limits."""
from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd


class VolatilityRiskCapController:
    """Reduce positions that exceed instrument or standalone-sector risk caps.

    The alpha optimizer remains responsible for distributing risk. This final
    controller only de-risks an otherwise valid target; it never renormalizes
    released capital and therefore cannot recreate a limit violation.
    """

    def __init__(
        self,
        *,
        asset_vol_budget: float,
        sector_vol_budget: float,
        hard_asset_cap: float = 1.0,
        gross_cap: float = 0.0,
        net_cap: float = 0.0,
        sector_map: Optional[Mapping[str, str]] = None,
        periods_per_year: float = 252.0,
        atr_window: int = 20,
        tolerance: float = 1e-10,
    ):
        self.asset_vol_budget = float(asset_vol_budget)
        self.sector_vol_budget = float(sector_vol_budget)
        self.hard_asset_cap = float(hard_asset_cap)
        self.gross_cap = float(gross_cap)
        self.net_cap = float(net_cap)
        self.periods_per_year = float(periods_per_year)
        self.atr_window = max(int(atr_window), 2)
        self.tolerance = float(tolerance)
        self._atr_provider_id: Optional[int] = None
        self._atr_dates = pd.DatetimeIndex([])
        self._atr_assets = pd.Index([])
        self._annual_atr = pd.DataFrame()
        if sector_map is None:
            from core.sectors import SECTOR_MAP

            sector_map = SECTOR_MAP
        self.sector_map = dict(sector_map)

    @classmethod
    def from_config(cls, config, *, sector_map=None):
        return cls(
            asset_vol_budget=float(config.asset_vol_budget),
            sector_vol_budget=float(config.sector_vol_budget),
            hard_asset_cap=float(config.hard_asset_cap),
            gross_cap=float(getattr(config, "gross_cap", 0.0)),
            net_cap=float(getattr(config, "net_cap", 0.0)),
            atr_window=int(getattr(config, "atr_window", 20)),
            sector_map=sector_map,
        )

    def prepare_data(self, data_manager, dates, instruments) -> bool:
        """Prefetch coherent OHLC once and calculate point-in-time ATR rates."""
        dates = pd.DatetimeIndex(dates).sort_values().unique()
        names = pd.Index(instruments)
        if len(dates) == 0 or len(names) == 0:
            return False
        covered = (
            self._atr_provider_id == id(data_manager)
            and dates.isin(self._atr_dates).all()
            and names.isin(self._atr_assets).all()
        )
        if covered:
            return True
        source = getattr(data_manager, "source", None)
        fetch = getattr(source, "fetch_price", None)
        if fetch is None:
            return False
        panel = fetch(names, dates[0], dates[-1], ["high", "low", "close"])
        missing = [field for field in ("high", "low", "close") if field not in panel]
        if missing:
            return False
        high = panel["high"].reindex(index=dates, columns=names).astype(float)
        low = panel["low"].reindex(index=dates, columns=names).astype(float)
        close = panel["close"].reindex(index=dates, columns=names).astype(float)
        from factors.user.supertrend import _coherent_ohlc_columns

        coherent = _coherent_ohlc_columns(high, low, close)
        if not coherent.all():
            bad = ", ".join(str(item) for item in coherent.index[~coherent])
            raise ValueError(
                "dynamic risk limits require coherent continuous-contract OHLC; "
                f"offending instruments: {bad}"
            )
        previous_close = close.shift(1)
        true_range = pd.DataFrame(
            np.maximum.reduce([
                (high - low).to_numpy(),
                (high - previous_close).abs().to_numpy(),
                (low - previous_close).abs().to_numpy(),
            ]),
            index=dates,
            columns=names,
        )
        atr = true_range.ewm(
            alpha=1.0 / self.atr_window,
            adjust=False,
            min_periods=self.atr_window,
        ).mean()
        self._atr_provider_id = id(data_manager)
        self._atr_dates = dates
        self._atr_assets = names
        self._annual_atr = atr.div(close.replace(0.0, np.nan)) * np.sqrt(
            self.periods_per_year
        )
        return True

    def annual_volatility_asof(
        self, date, instruments
    ) -> Optional[pd.Series]:
        if self._annual_atr.empty:
            return None
        available = self._annual_atr.index[
            self._annual_atr.index <= pd.Timestamp(date)
        ]
        if len(available) == 0:
            return None
        values = self._annual_atr.loc[available[-1]].reindex(instruments)
        return values if values.notna().any() else None

    def apply(
        self,
        weights: pd.Series,
        covariance: pd.DataFrame | np.ndarray,
        instruments: Optional[Sequence[str]] = None,
        *,
        covariance_is_annualized: bool = False,
        annual_volatility: Optional[pd.Series | np.ndarray] = None,
    ) -> tuple[pd.Series, dict]:
        names = pd.Index(instruments if instruments is not None else weights.index)
        result = weights.reindex(names).fillna(0.0).astype(float).copy()
        original = result.to_numpy(dtype=float).copy()
        matrix = self._aligned_covariance(covariance, names)
        annual_cov = matrix if covariance_is_annualized else matrix * self.periods_per_year

        annual_vol = self._aligned_volatility(
            annual_volatility, names, annual_cov
        )
        valid_vol = np.isfinite(annual_vol) & (annual_vol > self.tolerance)
        caps = np.full(len(names), self.hard_asset_cap, dtype=float)
        if self.asset_vol_budget > 0:
            caps[valid_vol] = np.minimum(
                caps[valid_vol], self.asset_vol_budget / annual_vol[valid_vol]
            )
        caps = np.maximum(caps, 0.0)
        result.iloc[:] = np.clip(original, -caps, caps)

        sector_vols = self._apply_sector_caps(result, annual_cov, names)
        gross = float(result.abs().sum())
        if self.gross_cap > 0 and gross > self.gross_cap:
            result *= self.gross_cap / gross
        net = float(result.sum())
        if self.net_cap > 0 and abs(net) > self.net_cap:
            result *= self.net_cap / abs(net)

        final_values = result.to_numpy(dtype=float)
        sector_vols = self._sector_volatility(final_values, annual_cov, names)
        asset_risk = np.abs(final_values) * annual_vol
        return result, {
            "constraint_adjusted": not np.allclose(
                final_values, original, rtol=0.0, atol=self.tolerance
            ),
            "dynamic_asset_caps": dict(zip(names.astype(str), caps.astype(float))),
            "max_asset_vol_proxy": float(np.nanmax(asset_risk)) if asset_risk.size else 0.0,
            "max_sector_standalone_vol": max(sector_vols.values(), default=0.0),
            "sector_standalone_vol": sector_vols,
            "gross_exposure": float(result.abs().sum()),
            "net_exposure": float(result.sum()),
        }

    def scale_for_aggregate(
        self,
        exposure: np.ndarray,
        annual_covariance: pd.DataFrame | np.ndarray,
        instruments: Sequence[str],
        *,
        covariance_is_psd: bool = False,
        annual_volatility: Optional[pd.Series | np.ndarray] = None,
    ) -> tuple[float, dict]:
        """Return one conservative capital scale for an aggregate sleeve mix."""
        names = pd.Index(instruments)
        values = np.asarray(exposure, dtype=float).reshape(-1)
        if values.size != len(names) or not np.isfinite(values).all():
            raise ValueError("aggregate exposure is misaligned or non-finite")
        matrix = self._aligned_covariance(
            annual_covariance, names, ensure_psd=not covariance_is_psd
        )
        annual_vol = self._aligned_volatility(
            annual_volatility, names, matrix
        )
        valid_vol = np.isfinite(annual_vol) & (annual_vol > self.tolerance)
        caps = np.full(len(names), self.hard_asset_cap, dtype=float)
        if self.asset_vol_budget > 0:
            caps[valid_vol] = np.minimum(
                caps[valid_vol], self.asset_vol_budget / annual_vol[valid_vol]
            )

        scale = 1.0
        active = np.abs(values) > self.tolerance
        if active.any():
            scale = min(scale, float(np.min(caps[active] / np.abs(values[active]))))

        sector_vols = self._sector_volatility(values, matrix, names)
        if self.sector_vol_budget > 0:
            for vol in sector_vols.values():
                if vol > self.sector_vol_budget:
                    scale = min(scale, self.sector_vol_budget / vol)
        gross = float(np.abs(values).sum())
        if self.gross_cap > 0 and gross > self.gross_cap:
            scale = min(scale, self.gross_cap / gross)
        net = float(values.sum())
        if self.net_cap > 0 and abs(net) > self.net_cap:
            scale = min(scale, self.net_cap / abs(net))
        scale = float(np.clip(scale, 0.0, 1.0))
        return scale, {
            "dynamic_risk_scale": scale,
            "dynamic_asset_caps": dict(zip(names.astype(str), caps.astype(float))),
            "max_asset_vol_proxy": float(np.max(np.abs(values * scale) * annual_vol))
            if values.size else 0.0,
            "max_sector_standalone_vol": max(
                (value * scale for value in sector_vols.values()), default=0.0
            ),
            "sector_standalone_vol": {
                key: float(value * scale) for key, value in sector_vols.items()
            },
        }

    def _apply_sector_caps(
        self, result: pd.Series, annual_cov: np.ndarray, names: pd.Index
    ) -> dict[str, float]:
        initial = self._sector_volatility(result.to_numpy(dtype=float), annual_cov, names)
        if self.sector_vol_budget <= 0:
            return initial
        for sector, indices in self._sector_indices(names).items():
            vol = initial.get(sector, 0.0)
            if vol > self.sector_vol_budget:
                result.iloc[indices] *= self.sector_vol_budget / vol
        return self._sector_volatility(result.to_numpy(dtype=float), annual_cov, names)

    def _sector_volatility(
        self, values: np.ndarray, covariance: np.ndarray, names: pd.Index
    ) -> dict[str, float]:
        result = {}
        for sector, indices in self._sector_indices(names).items():
            local = values[indices]
            local_cov = covariance[np.ix_(indices, indices)]
            result[sector] = float(np.sqrt(max(float(local @ local_cov @ local), 0.0)))
        return result

    def _sector_indices(self, names: pd.Index) -> dict[str, list[int]]:
        result: dict[str, list[int]] = {}
        for index, name in enumerate(names):
            sector = self.sector_map.get(str(name), "other")
            result.setdefault(sector, []).append(index)
        return result

    @staticmethod
    def _aligned_volatility(
        volatility: Optional[pd.Series | np.ndarray],
        names: pd.Index,
        covariance: np.ndarray,
    ) -> np.ndarray:
        fallback = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        if volatility is None:
            return fallback
        if isinstance(volatility, pd.Series):
            values = volatility.reindex(names).to_numpy(dtype=float)
        else:
            values = np.asarray(volatility, dtype=float).reshape(-1)
        if values.size != len(names):
            raise ValueError("volatility shape does not match instruments")
        valid = np.isfinite(values) & (values > 0.0)
        return np.where(valid, values, fallback)

    @staticmethod
    def _aligned_covariance(
        covariance: pd.DataFrame | np.ndarray,
        names: pd.Index,
        *,
        ensure_psd: bool = True,
    ) -> np.ndarray:
        if isinstance(covariance, pd.DataFrame):
            matrix = covariance.reindex(index=names, columns=names).to_numpy(dtype=float)
        else:
            matrix = np.asarray(covariance, dtype=float)
        if matrix.shape != (len(names), len(names)):
            raise ValueError("covariance shape does not match instruments")
        if not np.isfinite(matrix).all():
            raise ValueError("covariance contains NaN/Inf")
        matrix = (matrix + matrix.T) / 2.0
        if not ensure_psd:
            return matrix
        values, vectors = np.linalg.eigh(matrix)
        values = np.maximum(values, 0.0)
        return (vectors * values) @ vectors.T


__all__ = ["VolatilityRiskCapController"]
