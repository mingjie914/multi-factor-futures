"""Low-complexity economic-family equal-weight alpha baseline."""
from __future__ import annotations

from typing import Dict, Mapping, Optional

import numpy as np
import pandas as pd

from core.interfaces import ReturnModel
from core.registry import register
from core.types import Date, ExpectedReturns, FactorMatrix, ReturnMatrix, Universe, UniverseSchedule
from factors.utils import stack_factors_and_returns
from research.governance import factor_family
from alpha.ols import OLSModel


@register("return_model", "family_equal_weight")
class FamilyEqualWeightModel(ReturnModel):
    """Equal-weight factors within family and equal-weight economic families."""

    def __init__(
        self,
        ic_window: int = 252,
        max_factors_per_family: int = 0,
        family_map: Optional[Mapping[str, str]] = None,
    ):
        self.ic_window = int(ic_window)
        self.max_factors_per_family = int(max_factors_per_family)
        self.family_map = dict(family_map or {})
        self._selected: Dict[str, list[str]] = {}
        self._directions: Dict[str, float] = {}
        self._fitted = False

    def fit(
        self,
        factors: Dict[str, FactorMatrix],
        forward_returns: ReturnMatrix,
        universe: UniverseSchedule = None,
    ) -> "FamilyEqualWeightModel":
        if not factors:
            return self
        merged, names, _, _, _ = stack_factors_and_returns(factors, forward_returns)
        if merged.empty:
            return self
        dates = merged.index.get_level_values(0).unique()
        if self.ic_window > 0:
            merged = merged.loc[dates[-self.ic_window:]]
        X = merged[names].to_numpy(dtype=float)
        y = merged["fwd_ret"].to_numpy(dtype=float)
        ic = OLSModel._compute_ic_vector(X, y, len(names))

        grouped: Dict[str, list[tuple[str, float]]] = {}
        for index, name in enumerate(names):
            value = float(ic[index]) if np.isfinite(ic[index]) else 0.0
            family = factor_family(name, self.family_map)
            grouped.setdefault(family, []).append((name, value))
        self._selected = {}
        self._directions = {}
        for family, entries in grouped.items():
            entries.sort(key=lambda item: (-abs(item[1]), item[0]))
            if self.max_factors_per_family > 0:
                entries = entries[:self.max_factors_per_family]
            self._selected[family] = [name for name, _ in entries]
            for name, value in entries:
                self._directions[name] = 1.0 if value >= 0 else -1.0
        self._fitted = bool(self._selected)
        return self

    def predict(
        self,
        factors: Dict[str, FactorMatrix],
        universe: Universe,
        date: Date,
    ) -> ExpectedReturns:
        if not self._fitted:
            return pd.Series(0.0, index=universe)
        family_forecasts = []
        for names in self._selected.values():
            exposures = []
            for name in names:
                frame = factors.get(name)
                if frame is None or date not in frame.index:
                    continue
                row = frame.loc[date].reindex(universe).astype(float)
                std = float(row.std(ddof=0))
                standardized = (
                    (row - float(row.mean())) / std if std > 1e-12 else row * 0.0
                )
                exposures.append(standardized * self._directions[name])
            if exposures:
                family_forecasts.append(pd.concat(exposures, axis=1).mean(axis=1))
        if not family_forecasts:
            return pd.Series(0.0, index=universe)
        return pd.concat(family_forecasts, axis=1).mean(axis=1).reindex(universe).fillna(0.0)
