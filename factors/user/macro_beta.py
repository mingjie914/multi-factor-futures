"""Point-in-time macro innovation and lagged futures beta factors."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.interfaces import Factor
from factors.user import register_user_factor


@dataclass(frozen=True)
class MacroBetaSpec:
    slug: str
    fields: tuple[str, ...]
    transform: str
    window: int
    description: str


FACTOR_SPECS = (
    MacroBetaSpec(
        "macro_beta_pmi_new_orders_36m", ("pmi_new_orders",), "difference", 36,
        "PMI new-orders innovation times each future's lagged 36-month beta",
    ),
    MacroBetaSpec(
        "macro_beta_pmi_input_prices_36m", ("pmi_input_prices",), "difference", 36,
        "PMI input-price innovation times each future's lagged 36-month beta",
    ),
    MacroBetaSpec(
        "macro_beta_pmi_inventory_36m", ("pmi_raw_inventory",), "difference", 36,
        "PMI raw-inventory innovation times each future's lagged 36-month beta",
    ),
    MacroBetaSpec(
        "macro_beta_ppi_mom_36m", ("ppi_mom",), "difference", 36,
        "PPI monthly-rate innovation times each future's lagged 36-month beta",
    ),
    MacroBetaSpec(
        "macro_beta_ppirm_nonferrous_36m", ("ppirm_nonferrous_yoy",), "difference", 36,
        "Nonferrous input-price innovation times each future's lagged 36-month beta",
    ),
    MacroBetaSpec(
        "macro_beta_ppirm_agri_36m", ("ppirm_agri_yoy",), "difference", 36,
        "Agricultural input-price innovation times each future's lagged 36-month beta",
    ),
    MacroBetaSpec(
        "macro_beta_repo_7d_36m", ("repo_7d",), "difference", 36,
        "Seven-day repo-rate innovation times each future's lagged 36-month beta",
    ),
    MacroBetaSpec(
        "macro_beta_repo_curve_36m", ("repo_1y", "repo_7d"), "spread_difference", 36,
        "One-year less seven-day repo curve innovation times lagged 36-month beta",
    ),
    MacroBetaSpec(
        "macro_beta_shibor_curve_36m", ("shibor_6m", "shibor_3m"), "spread_difference", 36,
        "Six-month less three-month SHIBOR curve innovation times lagged 36-month beta",
    ),
    MacroBetaSpec(
        "macro_beta_taiwan_electronics_36m", ("taiwan_electronics",), "return", 36,
        "Taiwan electronics monthly return times each future's lagged 36-month beta",
    ),
    MacroBetaSpec(
        "macro_beta_social_financing_36m", ("social_financing_yoy",), "difference", 36,
        "Social-financing growth innovation times each future's lagged 36-month beta",
    ),
    MacroBetaSpec(
        "macro_beta_leading_index_36m", ("leading_index",), "return", 36,
        "Leading-indicator monthly return times each future's lagged 36-month beta",
    ),
)


def _empty(dates, universe) -> pd.DataFrame:
    return pd.DataFrame(np.nan, index=dates, columns=universe, dtype=float)


def _monthly_innovation(macro: pd.DataFrame, spec: MacroBetaSpec) -> pd.Series:
    monthly = macro.copy()
    monthly.index = monthly.index.to_period("M")
    monthly = monthly.groupby(level=0).last().sort_index()
    if spec.transform == "spread_difference":
        level = monthly[spec.fields[0]] - monthly[spec.fields[1]]
        return level.diff()
    level = monthly[spec.fields[0]]
    if spec.transform == "difference":
        return level.diff()
    if spec.transform == "return":
        return level.pct_change(fill_method=None)
    raise ValueError(f"unsupported macro transform: {spec.transform}")


def _rolling_beta(
    monthly_returns: pd.DataFrame,
    innovation: pd.Series,
    window: int,
) -> pd.DataFrame:
    """Pairwise rolling beta with the current observation excluded."""
    x = innovation.reindex(monthly_returns.index).astype(float)
    valid = monthly_returns.notna().mul(x.notna(), axis=0)
    y = monthly_returns.where(valid)
    x_by_asset = valid.mul(x, axis=0)
    min_periods = max(24, int(np.ceil(window * 2.0 / 3.0)))

    count = valid.rolling(window, min_periods=1).sum()
    sum_x = x_by_asset.rolling(window, min_periods=1).sum()
    sum_y = y.rolling(window, min_periods=1).sum()
    sum_xy = y.mul(x, axis=0).rolling(window, min_periods=1).sum()
    sum_x2 = x_by_asset.mul(x, axis=0).rolling(window, min_periods=1).sum()

    centered_xy = sum_xy - sum_x * sum_y / count.replace(0.0, np.nan)
    centered_x2 = sum_x2 - sum_x.pow(2) / count.replace(0.0, np.nan)
    beta = centered_xy / centered_x2.where(centered_x2.abs() > 1e-12)
    return beta.where(count >= min_periods).shift(1)


def _map_to_trading_dates(
    monthly_signal: pd.DataFrame,
    dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    # Observation month m becomes usable at the start of m+2. This leaves
    # one complete calendar month for publication and database ingestion.
    available = monthly_signal.copy()
    available.index = (available.index + 2).to_timestamp(how="start")
    combined_index = available.index.union(pd.DatetimeIndex(dates)).sort_values()
    return available.reindex(combined_index).ffill().reindex(dates)


def _compute_spec(spec: MacroBetaSpec, data, dates, universe) -> pd.DataFrame:
    close = data.get("close", dates, universe)
    if close is None or close.empty or not hasattr(data, "get_macro"):
        return _empty(dates, universe)
    close = (
        close.reindex(index=dates, columns=universe)
        .astype(float)
        .shift(1)
        .where(lambda frame: frame > 0)
    )
    monthly_close = close.groupby(close.index.to_period("M")).last()
    monthly_returns = monthly_close.pct_change(fill_method=None)

    macro_start = pd.Timestamp(dates.min()) - pd.DateOffset(
        months=spec.window + 6
    )
    macro = data.get_macro(
        list(spec.fields), start=macro_start, end=pd.Timestamp(dates.max())
    )
    if macro is None or macro.empty or any(
        field not in macro.columns for field in spec.fields
    ):
        return _empty(dates, universe)

    innovation = _monthly_innovation(macro, spec).replace(
        [np.inf, -np.inf], np.nan
    )
    monthly_index = monthly_returns.index.union(innovation.index).sort_values()
    monthly_returns = monthly_returns.reindex(monthly_index)
    innovation = innovation.reindex(monthly_index)
    lagged_beta = _rolling_beta(monthly_returns, innovation, spec.window)
    monthly_signal = lagged_beta.mul(innovation, axis=0)
    return _map_to_trading_dates(
        monthly_signal, pd.DatetimeIndex(dates)
    ).reindex(columns=universe).replace([np.inf, -np.inf], np.nan)


def _make_factor_class(spec: MacroBetaSpec):
    class MacroBetaFactor(Factor):
        name = spec.slug
        category = "macro_sensitivity"
        frequency = "daily"
        description = spec.description
        factor_spec = spec

        def dependencies(self) -> list[str]:
            return ["close"]

        def compute(self, data, dates, universe):
            if len(dates) == 0 or len(universe) == 0:
                return _empty(dates, universe)
            return _compute_spec(self.factor_spec, data, dates, universe)

    class_name = "".join(part.title() for part in spec.slug.split("_"))
    MacroBetaFactor.__name__ = class_name
    MacroBetaFactor.__qualname__ = class_name
    return register_user_factor(spec.slug, category="macro_sensitivity")(
        MacroBetaFactor
    )


for _factor_spec in FACTOR_SPECS:
    globals()[_factor_spec.slug] = _make_factor_class(_factor_spec)


__all__ = ["FACTOR_SPECS", "MacroBetaSpec"] + [
    spec.slug for spec in FACTOR_SPECS
]
