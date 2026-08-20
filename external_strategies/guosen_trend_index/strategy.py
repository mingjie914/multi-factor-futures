from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

import numpy as np
import pandas as pd
import yaml

from backtest.research_ledger import build_close_marked_ledger
from data.market_quality import prepare_close_data


@dataclass(frozen=True)
class GuosenTrendIndexSpec:
    universe: tuple[str, ...]
    selection_pct: float
    target_volatility: float
    volatility_window: int
    correlation_window: Optional[int]
    correlation_multiplier_cap: float
    minimum_risk_observations: int
    execution_lag_days: int
    periods_per_year: int
    transaction_cost_rate: float
    annual_management_fee: float
    asset_caps: Mapping[str, float]
    warmup_calendar_days: int = 180
    expected_final_holdings: int = 9
    factor_chunk_size: int = 400
    factor_chunk_overlap: int = 64

    @property
    def selection_count(self) -> int:
        return max(1, min(len(self.universe), round(len(self.universe) * self.selection_pct)))


@dataclass
class ExternalBacktestResult:
    nav: pd.Series
    returns: pd.Series
    gross_returns: pd.Series
    turnover: pd.Series
    costs: pd.Series
    weights: pd.DataFrame
    diagnostics: pd.DataFrame


FactorVariant = tuple[str, int]
FactorSet = dict[str, tuple[FactorVariant, ...]]


def _parse_factor_set(name: str, raw_factors: Mapping) -> FactorSet:
    """Parse factor groups while keeping flat one-parameter snapshots concise."""
    parsed: FactorSet = {}
    for group_name, value in raw_factors.items():
        if isinstance(value, int):
            variants = ((str(group_name), int(value)),)
        elif isinstance(value, list):
            variants = tuple(
                (str(item["name"]), int(item["direction"])) for item in value
            )
        else:
            raise ValueError(f"invalid factor group {group_name!r} in {name!r}")
        if not variants or any(direction not in (-1, 1) for _, direction in variants):
            raise ValueError(f"invalid factor group {group_name!r} in {name!r}")
        parsed[str(group_name)] = variants
    if not parsed:
        raise ValueError(f"empty factor set {name!r}")
    return parsed


def load_snapshot(path: str | Path) -> tuple[GuosenTrendIndexSpec, dict[str, FactorSet], dict]:
    snapshot_path = Path(path)
    raw = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
    spec = GuosenTrendIndexSpec(
        universe=tuple(str(item).upper() for item in raw["universe"]),
        selection_pct=float(raw["selection_pct"]),
        target_volatility=float(raw["target_volatility"]),
        volatility_window=int(raw["volatility_window"]),
        correlation_window=(
            None if raw.get("correlation_window") is None
            else int(raw["correlation_window"])
        ),
        correlation_multiplier_cap=float(raw["correlation_multiplier_cap"]),
        minimum_risk_observations=int(raw["minimum_risk_observations"]),
        execution_lag_days=max(0, int(raw["execution_lag_days"])),
        periods_per_year=int(raw["periods_per_year"]),
        transaction_cost_rate=float(raw["transaction_cost_rate"]),
        annual_management_fee=float(raw["annual_management_fee"]),
        asset_caps={str(k).upper(): float(v) for k, v in raw["asset_caps"].items()},
        warmup_calendar_days=max(0, int(raw.get("warmup_calendar_days", 180))),
        expected_final_holdings=int(raw.get("expected_final_holdings", 9)),
        factor_chunk_size=max(1, int(raw.get("factor_chunk_size", 400))),
        factor_chunk_overlap=max(0, int(raw.get("factor_chunk_overlap", 64))),
    )
    if not 0.0 < spec.selection_pct <= 1.0:
        raise ValueError("selection_pct must be in (0, 1]")
    if len(set(spec.universe)) != len(spec.universe):
        raise ValueError("universe roots must be unique")
    if (
        not np.isfinite(spec.target_volatility)
        or spec.target_volatility <= 0.0
        or spec.volatility_window < 2
        or spec.minimum_risk_observations < 2
        or spec.minimum_risk_observations > spec.volatility_window
        or spec.periods_per_year <= 0
    ):
        raise ValueError("invalid volatility or annualization settings")
    if spec.correlation_window is not None and spec.correlation_window < 2:
        raise ValueError("correlation_window must be at least 2 when enabled")
    if (
        not np.isfinite(spec.correlation_multiplier_cap)
        or spec.correlation_multiplier_cap <= 0.0
        or not np.isfinite(spec.transaction_cost_rate)
        or spec.transaction_cost_rate < 0.0
        or not np.isfinite(spec.annual_management_fee)
        or spec.annual_management_fee < 0.0
    ):
        raise ValueError("invalid correlation or cost settings")
    if spec.factor_chunk_overlap >= spec.factor_chunk_size:
        raise ValueError("factor_chunk_overlap must be smaller than factor_chunk_size")
    if not 1 <= spec.expected_final_holdings <= len(spec.universe):
        raise ValueError("expected_final_holdings must fit inside the universe")
    if set(spec.asset_caps) != set(spec.universe):
        raise ValueError("asset_caps must cover the universe exactly")
    caps = np.asarray(list(spec.asset_caps.values()), dtype=float)
    if not np.isfinite(caps).all() or np.any(caps < 0.0):
        raise ValueError("asset caps must be finite and non-negative")
    if int(np.count_nonzero(caps > 0.0)) < spec.expected_final_holdings:
        raise ValueError("positive asset caps cannot support expected_final_holdings")
    factor_sets = {
        str(name): _parse_factor_set(str(name), factors)
        for name, factors in raw["factor_sets"].items()
    }
    return spec, factor_sets, raw


class GuosenTrendIndexBacktester:
    """Standalone adapter; the production framework never imports this class."""

    def __init__(
        self,
        data_manager,
        factor_engine,
        spec: GuosenTrendIndexSpec,
        audited_nontrading_closes=None,
    ):
        self.data_manager = data_manager
        self.factor_engine = factor_engine
        self.spec = spec
        self.audited_nontrading_closes = audited_nontrading_closes or {}

    def _prepare_close_data(
        self, close: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        prepare = getattr(self.data_manager, "prepare_close_data", None)
        if callable(prepare):
            return prepare(close)
        return prepare_close_data(close, self.audited_nontrading_closes)

    def compute_factor_values(
        self, factor_names: list[str], dates: pd.DatetimeIndex
    ) -> dict[str, pd.DataFrame]:
        dates = pd.DatetimeIndex(dates)
        outputs = {
            name: pd.DataFrame(index=dates, columns=self.spec.universe, dtype=float)
            for name in factor_names
        }
        size = self.spec.factor_chunk_size
        overlap = self.spec.factor_chunk_overlap
        for start in range(0, len(dates), size):
            target = dates[start:start + size]
            request = dates[max(0, start - overlap):start + len(target)]
            computed = self.factor_engine.compute_factors(
                factor_names, request, list(self.spec.universe), parallel=False
            )
            for name in factor_names:
                if name not in computed:
                    raise KeyError(f"factor engine did not return {name!r}")
                outputs[name].loc[target] = computed[name].reindex(
                    index=target, columns=self.spec.universe
                )
        return outputs

    def _correlation_multiplier(self, history: pd.DataFrame) -> float:
        n_assets = history.shape[1]
        if self.spec.correlation_window is None or n_assets < 2:
            return 1.0
        corr = history.tail(self.spec.correlation_window).corr().to_numpy(dtype=float)
        upper = corr[np.triu_indices(n_assets, k=1)]
        upper = upper[np.isfinite(upper)]
        if upper.size == 0:
            return 1.0
        average = float(np.clip(upper.mean(), -0.99 / max(n_assets - 1, 1), 0.99))
        denominator = max(1.0 + (n_assets - 1) * average, 1e-8)
        raw = float(np.sqrt(n_assets / denominator))
        return min(raw, self.spec.correlation_multiplier_cap)

    def _one_factor_weights(
        self,
        signal: pd.Series,
        direction: int,
        risk_history: pd.DataFrame,
    ) -> pd.Series:
        scores = pd.to_numeric(signal, errors="coerce").round(10).dropna()
        if direction == -1:
            scores = -scores
        selected = scores.nlargest(min(self.spec.selection_count, len(scores))).index
        result = pd.Series(0.0, index=self.spec.universe, dtype=float)
        if len(selected) == 0:
            return result
        history = risk_history.reindex(columns=selected).tail(
            self.spec.volatility_window
        )
        counts = history.notna().sum(axis=0)
        volatility = (
            history.std(ddof=1)
            * np.sqrt(self.spec.periods_per_year)
        )
        volatility = volatility.replace([np.inf, -np.inf, 0.0], np.nan)
        valid = volatility.index[
            counts.ge(self.spec.minimum_risk_observations) & volatility.notna()
        ]
        if len(valid) == 0:
            return result
        history = history.reindex(columns=valid)
        multiplier = self._correlation_multiplier(history)
        weights = (
            self.spec.target_volatility
            / len(valid)
            / volatility.reindex(valid)
            * multiplier
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        result.loc[valid] = weights.clip(lower=0.0)
        return result

    def build_factor_portfolios(
        self,
        factor_values: Mapping[str, pd.DataFrame],
        factor_set: Mapping[str, tuple[FactorVariant, ...]],
        close: pd.DataFrame,
    ) -> dict[str, pd.DataFrame]:
        """Build each factor portfolio before cross-factor aggregation."""
        dates = pd.DatetimeIndex(close.index)
        required = list(dict.fromkeys(
            name for variants in factor_set.values() for name, _ in variants
        ))
        missing = sorted(set(required) - set(factor_values))
        if missing:
            raise KeyError(f"factor values are missing configured factors: {missing}")
        normalized_values = {}
        expected_columns = pd.Index(self.spec.universe)
        for name in required:
            frame = factor_values[name]
            if frame.index.has_duplicates or frame.columns.has_duplicates:
                raise ValueError(f"factor {name!r} has duplicate dates or roots")
            if not dates.isin(frame.index).all() or not expected_columns.isin(
                frame.columns
            ).all():
                raise ValueError(f"factor {name!r} does not cover close dates and universe")
            try:
                aligned = frame.reindex(
                    index=dates, columns=expected_columns
                ).astype(float)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"factor {name!r} must be numeric") from exc
            if np.isinf(aligned.to_numpy(dtype=float)).any():
                raise ValueError(f"factor {name!r} contains infinity")
            normalized_values[name] = aligned
        daily_returns, _ = self._prepare_close_data(close)
        portfolios = {
            group_name: pd.DataFrame(
                0.0, index=dates, columns=self.spec.universe, dtype=float
            )
            for group_name in factor_set
        }
        for position, date in enumerate(dates):
            history = daily_returns.iloc[:position + 1]
            if self.spec.execution_lag_days == 0:
                # Framework factors are pre-lagged, so same-index weights are
                # causal only when the risk estimate also excludes return[T].
                history = history.iloc[:-1]
            for group_name, variants in factor_set.items():
                per_parameter = []
                for name, direction in variants:
                    frame = normalized_values[name]
                    one = self._one_factor_weights(frame.loc[date], direction, history)
                    if one.gt(0.0).any():
                        per_parameter.append(one)
                if per_parameter:
                    portfolios[group_name].loc[date] = pd.concat(
                        per_parameter, axis=1
                    ).mean(axis=1)
        return portfolios

    def combine_factor_portfolios(
        self,
        portfolios: Mapping[str, pd.DataFrame],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if not portfolios:
            raise ValueError("at least one factor portfolio is required")
        first = next(iter(portfolios.values()))
        dates = pd.DatetimeIndex(first.index)
        weights = pd.DataFrame(0.0, index=dates, columns=self.spec.universe)
        diagnostics = pd.DataFrame(
            0, index=dates, columns=["active_factors", "selected_assets"], dtype=int
        )
        for date in dates:
            per_factor = [
                frame.loc[date]
                for frame in portfolios.values()
                if frame.loc[date].gt(0.0).any()
            ]
            if not per_factor:
                continue
            combined = pd.concat(per_factor, axis=1).mean(axis=1).fillna(0.0)
            caps = pd.Series(self.spec.asset_caps).reindex(combined.index).fillna(0.0)
            combined = combined.clip(lower=0.0).where(combined.le(caps), caps)
            weights.loc[date] = combined
            diagnostics.loc[date, "active_factors"] = len(per_factor)
            diagnostics.loc[date, "selected_assets"] = int(combined.gt(0.0).sum())
        return weights, diagnostics

    def build_signal_weights(
        self,
        factor_values: Mapping[str, pd.DataFrame],
        factor_set: Mapping[str, tuple[FactorVariant, ...]],
        close: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        portfolios = self.build_factor_portfolios(factor_values, factor_set, close)
        weights, diagnostics = self.combine_factor_portfolios(portfolios)
        self._validate_expected_holdings(weights, close)
        return weights, diagnostics

    def _validate_expected_holdings(
        self,
        weights: pd.DataFrame,
        close: pd.DataFrame,
    ) -> None:
        """Require all risk-eligible roots promised by the frozen snapshot."""
        returns, _ = self._prepare_close_data(close)
        caps = pd.Series(self.spec.asset_caps, dtype=float).reindex(self.spec.universe)
        actual = weights.gt(1e-12).sum(axis=1)
        for position, date in enumerate(weights.index):
            history = returns.iloc[:position + 1]
            if self.spec.execution_lag_days == 0:
                history = history.iloc[:-1]
            window = history.tail(self.spec.volatility_window)
            volatility = window.std(ddof=1).replace(
                [np.inf, -np.inf, 0.0], np.nan
            )
            eligible = (
                window.notna().sum(axis=0).ge(
                    self.spec.minimum_risk_observations
                )
                & volatility.notna()
                & caps.gt(0.0)
            )
            expected = min(
                self.spec.expected_final_holdings,
                int(eligible.sum()),
            )
            if int(actual.loc[date]) != expected:
                raise RuntimeError(
                    f"{date:%Y-%m-%d} final holdings={int(actual.loc[date])}, "
                    f"expected={expected} from risk-eligible universe"
                )

    def project_weights_to_gross(
        self,
        weights: pd.DataFrame,
        target_gross: float,
    ) -> pd.DataFrame:
        """Scale long-only rows to exact gross while preserving caps and ratios."""
        if not np.isfinite(target_gross) or target_gross <= 0.0:
            raise ValueError("target_gross must be positive and finite")
        columns = pd.Index(self.spec.universe)
        caps = pd.Series(self.spec.asset_caps, dtype=float).reindex(columns)
        if caps.isna().any() or caps.lt(0.0).any():
            raise ValueError("asset caps must be non-negative and cover the universe")
        projected = pd.DataFrame(0.0, index=weights.index, columns=columns)
        tolerance = 1e-12
        for date, row in weights.reindex(columns=columns).iterrows():
            base = pd.to_numeric(row, errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            ).fillna(0.0).clip(lower=0.0)
            active = base.index[base.gt(0.0)]
            if len(active) == 0:
                continue
            if float(caps.loc[active].sum()) + tolerance < target_gross:
                raise ValueError(
                    f"cannot reach gross {target_gross:g} on {date:%Y-%m-%d} "
                    "without adding unsupported assets or violating caps"
                )
            remaining = float(target_gross)
            result = pd.Series(0.0, index=columns, dtype=float)
            unfrozen = active
            while len(unfrozen):
                denominator = float(base.loc[unfrozen].sum())
                if denominator <= 0.0:
                    break
                proposed = base.loc[unfrozen] * (remaining / denominator)
                over_cap = proposed.gt(caps.loc[unfrozen] + tolerance)
                if not over_cap.any():
                    result.loc[unfrozen] = proposed
                    remaining = 0.0
                    break
                frozen = unfrozen[over_cap]
                result.loc[frozen] = caps.loc[frozen]
                remaining -= float(caps.loc[frozen].sum())
                unfrozen = unfrozen.difference(frozen, sort=False)
            if abs(remaining) > 1e-10:
                raise ValueError(
                    f"gross projection did not converge on {date:%Y-%m-%d}: "
                    f"residual={remaining:.3e}"
                )
            projected.loc[date] = result
        return projected

    def run_from_weights(
        self,
        signal_weights: pd.DataFrame,
        close: pd.DataFrame,
        diagnostics: Optional[pd.DataFrame] = None,
        base_value: float = 1000.0,
        contract_schedule: Optional[pd.DataFrame] = None,
    ) -> ExternalBacktestResult:
        close = close.reindex(columns=self.spec.universe).sort_index()
        if close.index.has_duplicates:
            raise ValueError("close dates must be unique")
        if signal_weights.index.has_duplicates or signal_weights.columns.has_duplicates:
            raise ValueError("signal weights must have unique dates and roots")
        if (
            set(signal_weights.index) != set(close.index)
            or set(signal_weights.columns) != set(self.spec.universe)
        ):
            raise ValueError("signal weights must exactly cover close dates and universe")
        try:
            signal_weights = signal_weights.reindex(
                index=close.index, columns=self.spec.universe
            ).astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError("signal weights must be numeric") from exc
        if not np.isfinite(signal_weights.to_numpy(dtype=float)).all():
            raise ValueError("signal weights contain NaN or infinity")
        caps = pd.Series(self.spec.asset_caps, dtype=float).reindex(self.spec.universe)
        if signal_weights.lt(-1e-12).any().any():
            raise ValueError("Guosen adapter accepts long-only signal weights")
        if signal_weights.gt(caps + 1e-12, axis="columns").any().any():
            raise ValueError("signal weights exceed configured asset caps")
        if diagnostics is None:
            diagnostics = pd.DataFrame(
                {
                    "active_factors": signal_weights.gt(0.0).any(axis=1).astype(int),
                    "selected_assets": signal_weights.gt(0.0).sum(axis=1),
                },
                index=close.index,
            )
        else:
            required = {"active_factors", "selected_assets"}
            if diagnostics.index.has_duplicates or not required.issubset(
                diagnostics.columns
            ):
                raise ValueError("diagnostics must have unique dates and required columns")
            if set(diagnostics.index) != set(close.index):
                raise ValueError("diagnostics must exactly cover close dates")
            diagnostics = diagnostics.reindex(close.index).astype(int)

        # A row is a close-T decision target.  ``execution_lag_days`` adds
        # optional *extra* decision bars; the ledger always makes that target
        # effective for the following close-to-close return.
        targets = signal_weights.shift(self.spec.execution_lag_days).fillna(0.0)
        asset_returns, close_tradable = self._prepare_close_data(close)
        ledger = build_close_marked_ledger(
            targets,
            asset_returns,
            trade_cost_rate=self.spec.transaction_cost_rate,
            annual_fee=self.spec.annual_management_fee,
            periods_per_year=self.spec.periods_per_year,
            contract_schedule=contract_schedule,
            decision_tradable=close_tradable,
            initial_nav=base_value,
        )
        daily = ledger.daily
        weights = ledger.effective_weights.reindex(columns=self.spec.universe)
        gross_returns = daily["gross_return"].copy()
        net_returns = daily["net_return"].copy()
        turnover = daily["executed_traded_notional"].copy()
        costs = daily["trade_cost"].add(daily["holding_cost"])
        nav = daily["nav_after"].copy()
        nav.name = "index_level"
        net_returns.name = "net_return"
        gross_returns.name = "gross_return"
        turnover.name = "turnover"
        costs.name = "cost"
        diagnostics = diagnostics.shift(
            self.spec.execution_lag_days + 1
        ).fillna(0).astype(int)
        return ExternalBacktestResult(
            nav=nav,
            returns=net_returns,
            gross_returns=gross_returns,
            turnover=turnover,
            costs=costs,
            weights=weights,
            diagnostics=diagnostics,
        )

    def run_from_values(
        self,
        factor_values: Mapping[str, pd.DataFrame],
        factor_set: Mapping[str, tuple[FactorVariant, ...]],
        close: pd.DataFrame,
        base_value: float = 1000.0,
        contract_schedule: Optional[pd.DataFrame] = None,
    ) -> ExternalBacktestResult:
        close = close.reindex(columns=self.spec.universe).sort_index()
        signal_weights, diagnostics = self.build_signal_weights(
            factor_values, factor_set, close
        )
        return self.run_from_weights(
            signal_weights,
            close,
            diagnostics=diagnostics,
            base_value=base_value,
            contract_schedule=contract_schedule,
        )
