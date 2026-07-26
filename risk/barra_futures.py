from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from core.interfaces import DataProvider, RiskModel
from core.registry import register
from core.types import (
    Date,
    FactorMatrix,
    ReturnMatrix,
    SpecificRisk,
    Universe,
    UniverseSchedule,
    WeightVector,
)


@register("risk_model", "barra_futures")
class BarraFuturesModel(RiskModel):
    """Low-dimensional, point-in-time futures risk model.

    Alpha factors are intentionally not used as risk factors. Risk exposures are
    derived from historical daily returns and market data using a fixed set of
    styles (momentum, volatility, skewness, liquidity and carry), plus sector
    dummies. When the cross-sectional factor model cannot be estimated, the
    model falls back to a shrunk asset covariance matrix.

    ``forward_returns`` keeps its historical interface name for compatibility,
    but callers must pass one-period historical returns. The backtester does so
    and truncates the input at t-1 before each estimate.
    """

    _DEFAULT_STYLES = (
        "carry",
        "momentum",
        "volatility",
        "skewness",
        "liquidity",
    )
    _MIN_VARIANCE = 1e-8
    _FALLBACK_DAILY_VARIANCE = 1e-4

    def __init__(
        self,
        style_factors: list = None,
        estimation_window: int = 252,
        covariance_estimator: str = "shrinkage",
    ):
        requested = style_factors or list(self._DEFAULT_STYLES)
        self._configured_style_factors = tuple(str(name).lower() for name in requested)
        self._style_factors = list(self._configured_style_factors)
        self._window = max(int(estimation_window), 20)
        self._cov_method = str(covariance_estimator).lower()

        self._factor_returns: Optional[pd.DataFrame] = None
        self._factor_cov: Optional[pd.DataFrame] = None
        self._specific_var: Optional[pd.Series] = None
        self._specific_var_panel: Optional[pd.DataFrame] = None
        self._exposures: Dict[str, pd.DataFrame] = {}
        self._risk_factor_names: list[str] = []
        self._asset_returns: Optional[pd.DataFrame] = None
        self._asset_return_var: Optional[pd.Series] = None
        self._factor_cov_cache: Dict[pd.Timestamp, pd.DataFrame] = {}
        self._asset_cov_cache: Dict[pd.Timestamp, pd.DataFrame] = {}
        self._prefetched_provider_id: Optional[int] = None
        self._prefetched_dates = pd.DatetimeIndex([])
        self._prefetched_assets = pd.Index([])
        self._prefetched_market: Dict[str, object] = {}

    def prepare_data(
        self, data: DataProvider, dates, assets
    ) -> "BarraFuturesModel":
        """Read slow market fields once; estimates still slice point-in-time."""
        dates = self._normalise_index(dates).sort_values().unique()
        assets = pd.Index(assets)
        covered = (
            self._prefetched_provider_id == id(data)
            and dates.isin(self._prefetched_dates).all()
            and assets.isin(self._prefetched_assets).all()
        )
        if covered:
            return self

        market: Dict[str, object] = {}
        if "liquidity" in self._configured_style_factors:
            for field in ("volume", "close"):
                try:
                    market[field] = data.get(field, dates, assets).reindex(
                        index=dates, columns=assets
                    )
                except Exception:
                    market[field] = None
        if "carry" in self._configured_style_factors:
            try:
                pair = data.get_contract_pair("close", dates, assets)
                market["near"] = pair.get("near", pd.DataFrame()).reindex(
                    index=dates, columns=assets
                )
                market["far"] = pair.get("far", pd.DataFrame()).reindex(
                    index=dates, columns=assets
                )
            except Exception:
                market["near"] = None
                market["far"] = None
        try:
            market["industry"] = data.get_industry(dates, assets).reindex(
                index=dates, columns=assets
            )
        except Exception:
            market["industry"] = None

        self._prefetched_provider_id = id(data)
        self._prefetched_dates = dates
        self._prefetched_assets = assets
        self._prefetched_market = market
        return self

    def estimate(
        self,
        data: DataProvider,
        factor_exposures: Dict[str, FactorMatrix],
        forward_returns: ReturnMatrix,
        universe: UniverseSchedule = None,
    ) -> "BarraFuturesModel":
        """Estimate a risk-model snapshot from historical one-period returns.

        ``factor_exposures`` is accepted for interface compatibility only. It is
        never promoted wholesale into the risk model, which keeps Alpha and risk
        factor definitions independent.
        """
        del factor_exposures, universe
        self._reset_estimated_state()

        returns = self._clean_returns(forward_returns)
        if returns.empty:
            return self
        self._asset_returns = returns
        self._asset_return_var = self._variance_asof(returns, returns.index[-1])

        exposures = self._build_risk_exposures(data, returns)
        self._exposures = exposures
        self._risk_factor_names = list(exposures)
        self._style_factors = [
            name for name in self._configured_style_factors if name in exposures
        ]

        if exposures:
            factor_returns, residuals = self._fit_cross_sections(returns, exposures)
            if factor_returns is not None and not factor_returns.empty:
                self._factor_returns = factor_returns
                self._specific_var_panel = self._ewma_variance_panel(residuals)
                if self._specific_var_panel is not None and not self._specific_var_panel.empty:
                    self._specific_var = self._specific_var_panel.iloc[-1].dropna()
                latest_date = factor_returns.index[-1]
                self._factor_cov = self._factor_covariance_asof(latest_date)

        # Prime the robust fallback even when the factor regression succeeded.
        self._asset_covariance_asof(returns.index[-1], returns.columns)
        return self

    def _reset_estimated_state(self) -> None:
        self._factor_returns = None
        self._factor_cov = None
        self._specific_var = None
        self._specific_var_panel = None
        self._exposures = {}
        self._risk_factor_names = []
        self._asset_returns = None
        self._asset_return_var = None
        self._factor_cov_cache = {}
        self._asset_cov_cache = {}

    def _clean_returns(self, returns: ReturnMatrix) -> pd.DataFrame:
        if returns is None or returns.empty:
            return pd.DataFrame()
        frame = returns.copy()
        frame.index = self._normalise_index(frame.index)
        frame = frame[~frame.index.duplicated(keep="last")].sort_index()
        frame = frame.apply(pd.to_numeric, errors="coerce")
        frame = frame.replace([np.inf, -np.inf], np.nan)
        frame = frame.dropna(axis=1, how="all")
        if len(frame) > self._window:
            frame = frame.iloc[-self._window :]
        return frame

    @staticmethod
    def _normalise_index(index) -> pd.DatetimeIndex:
        result = pd.DatetimeIndex(pd.to_datetime(index))
        if result.tz is not None:
            result = result.tz_convert(None)
        return result

    def _build_risk_exposures(
        self, data: DataProvider, returns: pd.DataFrame
    ) -> Dict[str, pd.DataFrame]:
        dates = returns.index
        assets = returns.columns
        raw_styles: Dict[str, pd.DataFrame] = {}

        if "momentum" in self._configured_style_factors:
            raw_styles["momentum"] = (
                (1.0 + returns).rolling(20, min_periods=10).apply(np.prod, raw=True) - 1.0
            )
        if "volatility" in self._configured_style_factors:
            raw_styles["volatility"] = returns.rolling(20, min_periods=10).std()
        if "skewness" in self._configured_style_factors:
            raw_styles["skewness"] = returns.rolling(60, min_periods=20).skew()
        if "liquidity" in self._configured_style_factors:
            liquidity = self._load_liquidity(data, dates, assets)
            if liquidity is not None:
                raw_styles["liquidity"] = liquidity
        if "carry" in self._configured_style_factors:
            carry = self._load_carry(data, dates, assets)
            if carry is not None:
                raw_styles["carry"] = carry

        exposures: Dict[str, pd.DataFrame] = {}
        for name in self._configured_style_factors:
            panel = raw_styles.get(name)
            if panel is None:
                continue
            standardised = self._cross_sectional_standardise(
                panel.reindex(index=dates, columns=assets)
            )
            if np.isfinite(standardised.to_numpy()).any() and float(
                standardised.abs().to_numpy().sum()
            ) > 0:
                exposures[name] = standardised

        exposures.update(self._sector_exposures(data, dates, assets))
        return exposures

    @staticmethod
    def _cross_sectional_standardise(panel: pd.DataFrame) -> pd.DataFrame:
        panel = panel.apply(pd.to_numeric, errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        means = panel.mean(axis=1)
        stds = panel.std(axis=1, ddof=0).replace(0.0, np.nan)
        zscore = panel.sub(means, axis=0).div(stds, axis=0).clip(-5.0, 5.0)
        # Missing or unavailable exposures are neutral, not untradable.
        return zscore.fillna(0.0)

    def _load_liquidity(
        self, data: DataProvider, dates: pd.DatetimeIndex, assets: pd.Index
    ) -> Optional[pd.DataFrame]:
        try:
            cached_volume = self._prefetched_market.get("volume")
            cached_close = self._prefetched_market.get("close")
            if isinstance(cached_volume, pd.DataFrame) and isinstance(cached_close, pd.DataFrame):
                volume = cached_volume.reindex(index=dates, columns=assets)
                close = cached_close.reindex(index=dates, columns=assets)
            else:
                volume = data.get("volume", dates, assets).reindex(index=dates, columns=assets)
                close = data.get("close", dates, assets).reindex(index=dates, columns=assets)
            traded_value_proxy = close.abs() * volume.clip(lower=0)
            result = np.log1p(traded_value_proxy).rolling(20, min_periods=10).mean()
            return result if result.notna().any().any() else None
        except Exception:
            return None

    def _load_carry(
        self, data: DataProvider, dates: pd.DatetimeIndex, assets: pd.Index
    ) -> Optional[pd.DataFrame]:
        try:
            cached_near = self._prefetched_market.get("near")
            cached_far = self._prefetched_market.get("far")
            if isinstance(cached_near, pd.DataFrame) and isinstance(cached_far, pd.DataFrame):
                near = cached_near.reindex(index=dates, columns=assets)
                far = cached_far.reindex(index=dates, columns=assets)
            else:
                pair = data.get_contract_pair("close", dates, assets)
                near = pair.get("near", pd.DataFrame()).reindex(index=dates, columns=assets)
                far = pair.get("far", pd.DataFrame()).reindex(index=dates, columns=assets)
            valid = (near > 0) & (far > 0)
            carry = np.log(near.where(valid) / far.where(valid))
            carry = carry.rolling(5, min_periods=3).mean()
            return carry if carry.notna().any().any() else None
        except Exception:
            return None

    def _sector_exposures(
        self, data: DataProvider, dates: pd.DatetimeIndex, assets: pd.Index
    ) -> Dict[str, pd.DataFrame]:
        try:
            cached_industry = self._prefetched_market.get("industry")
            if isinstance(cached_industry, pd.DataFrame):
                industry = cached_industry.reindex(index=dates, columns=assets)
            else:
                industry = data.get_industry(dates, assets).reindex(index=dates, columns=assets)
        except Exception:
            industry = pd.DataFrame(index=dates, columns=assets, dtype=object)

        if industry.empty or industry.notna().sum().sum() == 0:
            try:
                from core.sectors import SECTOR_MAP

                labels = pd.Series(
                    [SECTOR_MAP.get(str(asset), "other") for asset in assets],
                    index=assets,
                )
                industry = pd.DataFrame(
                    np.tile(labels.to_numpy(), (len(dates), 1)),
                    index=dates,
                    columns=assets,
                )
            except Exception:
                return {}

        industry = industry.astype(object).where(industry.notna(), "other")
        industry = industry.apply(lambda col: col.map(lambda value: str(value).strip() or "other"))
        first_labels = industry.iloc[0]
        first_counts = first_labels.value_counts()
        if first_counts.empty:
            return {}
        # A fixed baseline prevents intercept collinearity and is independent of
        # classifications introduced later in the estimation window.
        baseline = "other" if "other" in first_counts.index else str(first_counts.index[0])
        sectors = sorted({str(value) for value in industry.to_numpy().ravel()})
        result: Dict[str, pd.DataFrame] = {}
        for sector in sectors:
            if sector == baseline:
                continue
            result[f"sector:{sector}"] = industry.eq(sector).astype(float)
        return result

    def _fit_cross_sections(
        self,
        returns: pd.DataFrame,
        exposures: Dict[str, pd.DataFrame],
    ) -> tuple[Optional[pd.DataFrame], pd.DataFrame]:
        names = list(exposures)
        n_dates, n_assets = returns.shape
        coefficient_values = np.full((n_dates, len(names)), np.nan, dtype=float)
        residual_values = np.full((n_dates, n_assets), np.nan, dtype=float)
        min_observations = max(len(names) + 2, 8)
        y_values = returns.to_numpy(dtype=float)
        x_values = np.stack([
            exposures[name].reindex(
                index=returns.index, columns=returns.columns
            ).to_numpy(dtype=float)
            for name in names
        ], axis=2)
        x_values = np.nan_to_num(x_values, nan=0.0, posinf=0.0, neginf=0.0)

        for row in range(n_dates):
            valid = np.isfinite(y_values[row])
            if int(valid.sum()) < min_observations:
                continue
            x_valid = x_values[row, valid]
            y_valid = y_values[row, valid]
            design = np.column_stack([np.ones(len(x_valid)), x_valid])
            try:
                beta, _, _, _ = np.linalg.lstsq(design, y_valid, rcond=None)
            except np.linalg.LinAlgError:
                continue
            if not np.isfinite(beta).all():
                continue
            coefficient_values[row] = beta[1:]
            residual_values[row, valid] = y_valid - design @ beta

        coefficients = pd.DataFrame(
            coefficient_values, index=returns.index, columns=names
        )
        residuals = pd.DataFrame(
            residual_values, index=returns.index, columns=returns.columns
        )

        coefficients = coefficients.dropna(how="all")
        if coefficients.empty:
            return None, residuals
        return coefficients, residuals

    def _factor_covariance_asof(self, date: Date) -> Optional[pd.DataFrame]:
        if self._factor_returns is None or self._factor_returns.empty:
            return None
        date = self._timestamp(date)
        available = self._factor_returns.index[self._factor_returns.index <= date]
        if len(available) < 2:
            return None
        use_date = available[-1]
        if use_date in self._factor_cov_cache:
            return self._factor_cov_cache[use_date]
        history = self._factor_returns.loc[:use_date].iloc[-self._window :]
        covariance = self._shrunk_covariance(history)
        if covariance is not None:
            self._factor_cov_cache[use_date] = covariance
        return covariance

    def _asset_covariance_asof(
        self, date: Date, universe: Universe
    ) -> pd.DataFrame:
        universe = pd.Index(universe)
        if self._asset_returns is None or self._asset_returns.empty:
            return pd.DataFrame(
                np.eye(len(universe)) * self._FALLBACK_DAILY_VARIANCE,
                index=universe,
                columns=universe,
            )
        date = self._timestamp(date)
        available = self._asset_returns.index[self._asset_returns.index <= date]
        if len(available) == 0:
            return pd.DataFrame(
                np.eye(len(universe)) * self._FALLBACK_DAILY_VARIANCE,
                index=universe,
                columns=universe,
            )
        use_date = available[-1]
        if use_date not in self._asset_cov_cache:
            history = self._asset_returns.loc[:use_date].iloc[-self._window :]
            estimated = self._shrunk_covariance(history)
            if estimated is not None:
                self._asset_cov_cache[use_date] = estimated
        base = self._asset_cov_cache.get(use_date)
        return self._reindex_covariance(base, universe)

    def _shrunk_covariance(self, frame: pd.DataFrame) -> Optional[pd.DataFrame]:
        if frame is None or frame.empty or len(frame) < 2:
            return None
        clean = frame.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")
        if clean.empty:
            return None
        means = clean.mean(axis=0).fillna(0.0)
        values = clean.fillna(means).fillna(0.0).to_numpy(dtype=float)
        if values.shape[0] < 2:
            return None
        if values.shape[1] == 1:
            variance = max(float(np.var(values[:, 0], ddof=1)), self._MIN_VARIANCE)
            return pd.DataFrame([[variance]], index=clean.columns, columns=clean.columns)

        sample = np.cov(values, rowvar=False, ddof=1)
        sample = np.asarray(sample, dtype=float)
        sample = np.nan_to_num(sample, nan=0.0, posinf=0.0, neginf=0.0)
        if self._cov_method == "shrinkage":
            delta = self._ledoit_wolf_intensity(values, sample)
            sample = (1.0 - delta) * sample + delta * np.diag(np.diag(sample))
        sample = self._ensure_psd(sample)
        diagonal = np.maximum(np.diag(sample), self._MIN_VARIANCE)
        np.fill_diagonal(sample, diagonal)
        return pd.DataFrame(sample, index=clean.columns, columns=clean.columns)

    def _reindex_covariance(
        self, covariance: Optional[pd.DataFrame], universe: pd.Index
    ) -> pd.DataFrame:
        if covariance is None or covariance.empty:
            return pd.DataFrame(
                np.eye(len(universe)) * self._FALLBACK_DAILY_VARIANCE,
                index=universe,
                columns=universe,
            )
        aligned = covariance.reindex(index=universe, columns=universe)
        known_diag = np.diag(covariance.to_numpy(dtype=float))
        known_diag = known_diag[np.isfinite(known_diag) & (known_diag > 0)]
        fill_variance = (
            float(np.median(known_diag))
            if known_diag.size
            else self._FALLBACK_DAILY_VARIANCE
        )
        values = aligned.to_numpy(dtype=float)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        missing_assets = ~pd.Index(universe).isin(covariance.index)
        diagonal = np.maximum(np.diag(values), self._MIN_VARIANCE)
        diagonal[missing_assets] = fill_variance
        np.fill_diagonal(values, diagonal)
        values = self._ensure_psd(values)
        return pd.DataFrame(values, index=universe, columns=universe)

    @staticmethod
    def _ewma_variance_panel(residuals: pd.DataFrame) -> Optional[pd.DataFrame]:
        if residuals is None or residuals.empty:
            return None
        result = residuals.ewm(span=60, min_periods=10).var().clip(lower=1e-8)
        return result if result.notna().any().any() else None

    @staticmethod
    def _variance_asof(frame: pd.DataFrame, date: Date) -> pd.Series:
        date = BarraFuturesModel._timestamp(date)
        history = frame.loc[frame.index <= date]
        if history.empty:
            return pd.Series(dtype=float)
        return history.var().clip(lower=BarraFuturesModel._MIN_VARIANCE)

    def _specific_variance_asof(self, date: Date, universe: Universe) -> pd.Series:
        universe = pd.Index(universe)
        date = self._timestamp(date)
        if self._specific_var_panel is not None and not self._specific_var_panel.empty:
            available = self._specific_var_panel.index[
                self._specific_var_panel.index <= date
            ]
            if len(available):
                row = self._specific_var_panel.loc[available[-1]].reindex(universe)
                valid = row[np.isfinite(row) & (row > 0)]
                if not valid.empty:
                    return row.fillna(float(valid.median())).clip(lower=self._MIN_VARIANCE)
        if self._asset_returns is not None and not self._asset_returns.empty:
            variance = self._variance_asof(self._asset_returns, date).reindex(universe)
            valid = variance[np.isfinite(variance) & (variance > 0)]
            if not valid.empty:
                return variance.fillna(float(valid.median())).clip(lower=self._MIN_VARIANCE)
        return pd.Series(self._FALLBACK_DAILY_VARIANCE, index=universe, dtype=float)

    @staticmethod
    def _timestamp(date: Date) -> pd.Timestamp:
        result = pd.Timestamp(date)
        if result.tzinfo is not None:
            result = result.tz_convert(None)
        return result

    @staticmethod
    def _ledoit_wolf_intensity(X: np.ndarray, S: np.ndarray) -> float:
        """Estimate diagonal-target Ledoit-Wolf shrinkage intensity."""
        n, p = X.shape
        if n < 3 or p < 2:
            return 1.0
        Xc = X - X.mean(axis=0, keepdims=True)
        off_diagonal = ~np.eye(p, dtype=bool)
        d2 = float(np.sum(S[off_diagonal] ** 2))
        if d2 < 1e-18:
            return 1.0
        row_sq = Xc**2
        a_term = float(np.sum(row_sq.sum(axis=1) ** 2))
        b_term = float(np.sum(row_sq**2))
        c_term = n * float(np.sum(S**2))
        d_term = n * float(np.sum(np.diag(S) ** 2))
        numerator = (a_term - b_term) - 2.0 * (c_term - d_term) + n * d2
        b_bar2 = max(numerator / (n**2), 0.0)
        return float(np.clip(b_bar2 / d2, 0.0, 1.0))

    def covariance(self, date: Date, universe: Universe) -> pd.DataFrame:
        """Return a daily N x N covariance matrix in ``universe`` order."""
        universe = pd.Index(universe)
        if len(universe) == 0:
            return pd.DataFrame(index=universe, columns=universe, dtype=float)

        exposure_date = self._latest_exposure_date(date)
        factor_covariance = (
            self._factor_covariance_asof(exposure_date)
            if exposure_date is not None
            else None
        )
        if factor_covariance is None or factor_covariance.empty:
            return self._asset_covariance_asof(date, universe)

        names = list(factor_covariance.columns)
        try:
            exposure_matrix = pd.concat(
                [self._exposures[name].loc[exposure_date].rename(name) for name in names],
                axis=1,
            ).reindex(universe).fillna(0.0)
        except (KeyError, ValueError):
            return self._asset_covariance_asof(date, universe)

        x_values = exposure_matrix.to_numpy(dtype=float)
        f_values = factor_covariance.to_numpy(dtype=float)
        if not np.isfinite(x_values).all() or not np.isfinite(f_values).all():
            return self._asset_covariance_asof(date, universe)
        covariance = x_values @ self._ensure_psd(f_values) @ x_values.T
        covariance += np.diag(
            self._specific_variance_asof(exposure_date, universe).to_numpy(dtype=float)
        )
        covariance = np.nan_to_num(covariance, nan=0.0, posinf=0.0, neginf=0.0)
        covariance = self._ensure_psd(covariance)
        diagonal = np.maximum(np.diag(covariance), self._MIN_VARIANCE)
        np.fill_diagonal(covariance, diagonal)
        return pd.DataFrame(covariance, index=universe, columns=universe)

    def _latest_exposure_date(self, date: Date) -> Optional[pd.Timestamp]:
        if not self._exposures:
            return None
        target = self._timestamp(date)
        common: Optional[pd.DatetimeIndex] = None
        for panel in self._exposures.values():
            eligible = panel.index[panel.index <= target]
            common = eligible if common is None else common.intersection(eligible)
        if common is None or len(common) == 0:
            return None
        return common[-1]

    def _diag_prior(self, universe: Universe, date: Date = None) -> np.ndarray:
        target = date
        if target is None:
            target = (
                self._asset_returns.index[-1]
                if self._asset_returns is not None and not self._asset_returns.empty
                else pd.Timestamp.min
            )
        return self._specific_variance_asof(target, universe).to_numpy(dtype=float)

    @staticmethod
    def _ensure_psd(matrix: np.ndarray, tol: float = 1e-10) -> np.ndarray:
        values = np.asarray(matrix, dtype=float)
        if values.size == 0:
            return values
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        values = (values + values.T) / 2.0
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(values)
            if eigenvalues.min() < tol:
                eigenvalues = np.maximum(eigenvalues, tol)
                values = (eigenvectors * eigenvalues) @ eigenvectors.T
                values = (values + values.T) / 2.0
        except np.linalg.LinAlgError:
            diagonal = np.maximum(np.diag(values), tol)
            values = np.diag(diagonal)
        return values

    def specific_risk(self, date: Date, universe: Universe) -> SpecificRisk:
        """Return daily idiosyncratic variances for the requested assets."""
        return self._specific_variance_asof(date, universe)

    def factor_exposure(
        self,
        weights: WeightVector,
        date: Date,
        universe: Universe,
    ) -> pd.Series:
        """Return portfolio exposures to the independent risk factors."""
        exposure_date = self._latest_exposure_date(date)
        if exposure_date is None:
            return pd.Series(0.0, index=self._risk_factor_names, dtype=float)
        aligned_weights = weights.reindex(universe).fillna(0.0)
        result = {}
        for name in self._risk_factor_names:
            row = self._exposures[name].loc[exposure_date].reindex(universe).fillna(0.0)
            result[name] = float(aligned_weights @ row)
        return pd.Series(result, dtype=float)

    def portfolio_risk(self, weights: WeightVector, date: Date) -> float:
        """Return annualised portfolio volatility from daily covariance."""
        covariance = self.covariance(date, weights.index)
        values = weights.reindex(covariance.index).fillna(0.0).to_numpy(dtype=float)
        variance = max(float(values @ covariance.to_numpy() @ values), 0.0)
        return float(np.sqrt(variance * 252.0))
