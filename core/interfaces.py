"""Abstract interfaces for the multi-factor framework.

All pluggable components are defined here as abstract base classes (ABCs)
to enforce a consistent contract across implementations.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol

import numpy as np  # noqa: F401
import pandas as pd

from core.types import (
    Date,
    DateIndex,
    ExpectedReturns,
    FactorMatrix,
    IndustryMapping,
    NAVSeries,
    PricePanel,
    ReturnMatrix,
    SpecificRisk,
    TickerIndex,
    Universe,
    UniverseSchedule,
    WeightVector,
)


# ---------------------------------------------------------------------------
# Market specification
# ---------------------------------------------------------------------------


class MarketSpec(abc.ABC):
    """Market-specific metadata and conversion rules."""

    market: str = ""

    @abc.abstractmethod
    def trading_calendar(self, start: Date, end: Date) -> DateIndex:
        """Return the trading calendar between *start* and *end*."""
        ...

    @abc.abstractmethod
    def ticker_format(self, ticker: str) -> str:
        """Normalise or validate a ticker string."""
        ...

    @abc.abstractmethod
    def default_universe(self, date: Date) -> Universe:
        """Return the default universe of tradable instruments on *date*."""
        ...

    @abc.abstractmethod
    def industry_classification(
        self, tickers: TickerIndex, date: Date
    ) -> IndustryMapping:
        """Return industry codes for the given *tickers* on *date*."""
        ...

    @abc.abstractmethod
    def risk_free_rate(self, date: Date) -> float:
        """Return the risk-free rate on *date*."""
        ...

    @abc.abstractmethod
    def transaction_cost_rate(
        self, date: Date, ticker: str, turnover_side: str
    ) -> float:
        """Return the transaction cost rate for a trade on *date*.

        Args:
            date: Trading date.
            ticker: Instrument identifier.
            turnover_side: 'buy' or 'sell'.
        """
        ...

    @abc.abstractmethod
    def point_value(self, ticker: str) -> float:
        """Return the notional value per point/index unit for *ticker*."""
        ...


# ---------------------------------------------------------------------------
# Data provider  (abstracted access layer used by factors)
# ---------------------------------------------------------------------------


class DataProvider(abc.ABC):
    """Abstract data access layer consumed by factors and risk models."""

    @abc.abstractmethod
    def get(
        self, field: str, dates: DateIndex, universe: Universe
    ) -> pd.DataFrame:
        """Retrieve a single field of data (dates x tickers)."""
        ...

    @abc.abstractmethod
    def get_industry(
        self, dates: DateIndex, universe: Universe
    ) -> pd.DataFrame:
        """Retrieve industry classification (dates x tickers)."""
        ...

    @abc.abstractmethod
    def get_universe(self, date: Date) -> Universe:
        """Return the universe of tickers available on *date*."""
        ...

    def get_contract_pair(
        self, field: str, dates: DateIndex, universe: Universe
    ) -> Dict[str, pd.DataFrame]:
        """Retrieve near-month and far-month contract prices.

        Used by multi-contract factors (e.g., roll yield). Returns raw
        (unadjusted) prices per contract; rollover gaps are intentional.

        Args:
            field: Price field (close/settle/open etc).
            dates: Date index.
            universe: Root codes (e.g., ['RB', 'IF']).

        Returns:
            {"near": DataFrame(dates x roots), "far": DataFrame(dates x roots)}
            Default implementation returns empty DataFrames (for sources
            that don't support multi-contract queries).
        """
        return {"near": pd.DataFrame(), "far": pd.DataFrame()}

    def get_macro(
        self,
        fields: List[str],
        start: Optional[Date] = None,
        end: Optional[Date] = None,
    ) -> pd.DataFrame:
        """Retrieve macro observations indexed by their observation month.

        Macro publication timing is factor-specific. Implementations return
        the stored observations unchanged; factors must apply an explicit
        publication lag before mapping them to trading dates.
        """
        return pd.DataFrame(columns=list(fields), dtype=float)

    def get_at_frequency(
        self,
        field: str,
        dates: DateIndex,
        universe: Universe,
        frequency: str = "daily",
    ) -> pd.DataFrame:
        """Retrieve a single field at a specific frequency (周期感知接口).

        默认实现: 当 frequency == "daily" 时回退到 get(), 否则抛出
        NotImplementedError. 子类 (如 DataManager) 可重写以路由到
        DataSource.fetch_price_at_frequency.

        Args:
            field: 数据字段 (如 "close", "volume")
            dates: 日期/时间索引
            universe: 标的集合
            frequency: 周期单位 ("daily" / "15min" / "30min" / "hourly")

        Returns:
            DataFrame(dates × tickers)

        Raises:
            NotImplementedError: 当不支持指定频率时
        """
        if frequency == "daily":
            return self.get(field, dates, universe)
        raise NotImplementedError(
            f"{self.__class__.__name__} 不支持 frequency={frequency!r}"
        )


# ---------------------------------------------------------------------------
# Data source  (raw data fetching layer)
# ---------------------------------------------------------------------------


class DataSource(abc.ABC):
    """Raw data source connector (e.g. database, API, file)."""

    market: str = ""

    @abc.abstractmethod
    def fetch_price(
        self,
        tickers: TickerIndex,
        start: Date,
        end: Date,
        fields: List[str],
    ) -> PricePanel:
        """Fetch price data for the requested tickers and date range."""
        ...

    @abc.abstractmethod
    def fetch_fundamental(
        self,
        tickers: TickerIndex,
        start: Date,
        end: Date,
        fields: List[str],
    ) -> dict:
        """Fetch fundamental data for the requested tickers and date range."""
        ...

    @abc.abstractmethod
    def fetch_industry(
        self, tickers: TickerIndex, date: Date
    ) -> IndustryMapping:
        """Fetch industry classification for *tickers* on *date*."""
        ...

    @abc.abstractmethod
    def fetch_index_constituents(
        self, index_code: str, date: Date
    ) -> Universe:
        """Fetch the constituent tickers of an index on *date*."""
        ...

    @abc.abstractmethod
    def fetch_calendar(self, start: Date, end: Date) -> DateIndex:
        """Fetch trading calendar between *start* and *end*."""
        ...

    def fetch_price_at_frequency(
        self,
        tickers: TickerIndex,
        start: Date,
        end: Date,
        fields: List[str],
        frequency: str = "daily",
    ) -> PricePanel:
        """Fetch price data at a specific frequency (周期感知接口).

        默认实现: 当 frequency == "daily" 时回退到 fetch_price (日度数据),
        否则抛出 NotImplementedError. 支持分钟数据的 Parquet 数据源可重写
        该接口.

        Args:
            tickers: 标的列表
            start: 起始日期/时间
            end: 结束日期/时间
            fields: 价格字段列表 (如 ["open","high","low","close","volume"])
            frequency: 周期单位 ("daily" / "15min" / "30min" / "hourly")

        Returns:
            PricePanel: {field: DataFrame(tickers × datetimes)}

        Raises:
            NotImplementedError: 当数据源不支持指定频率时
        """
        if frequency == "daily":
            return self.fetch_price(tickers, start, end, fields)
        raise NotImplementedError(
            f"{self.__class__.__name__} 不支持 frequency={frequency!r}, "
            f"仅支持 daily. 请使用支持该周期的 Parquet 数据源."
        )

    def fetch_contract_schedule(
        self,
        tickers: TickerIndex,
        start: Date,
        end: Date,
    ) -> Optional[pd.DataFrame]:
        """Return concrete contracts effective by root/date when available.

        Sources that expose only an opaque continuous series return ``None``.
        Formal futures ledgers can then distinguish an unavailable contract
        schedule from a genuine no-roll interval.
        """
        return None

    def fetch_macro(
        self,
        fields: List[str],
        start: Optional[Date] = None,
        end: Optional[Date] = None,
    ) -> pd.DataFrame:
        """Fetch macro observations indexed by their observation month."""
        return pd.DataFrame(columns=list(fields), dtype=float)


# ---------------------------------------------------------------------------
# Factor
# ---------------------------------------------------------------------------


class Factor(abc.ABC):
    """A factor that computes exposure values for a universe over time.

    周期架构说明:
        - `frequency` 表示因子的周期单位 ("daily" / "15min" / "30min" / "hourly")
        - 因子内部计算所用的 window/lag 等参数均为"周期数" (bar数), 不是天数
        - 当 frequency="daily" (默认) 时, window=5 表示 5 个交易日
        - 当 frequency="15min" 时, window=5 表示 5 个 15分钟 bar
    """

    name: str = ""
    category: str = ""
    # 周期单位: 与 core.period.PeriodUnit 枚举值对应
    frequency: str = "daily"
    # Formal forward-return horizons, in bars at ``frequency``. Registration
    # freezes a deterministic value for legacy factors that do not declare it.
    validation_horizons: tuple[int, ...] = ()
    horizon_unit: str = "bars"
    description: str = ""

    @abc.abstractmethod
    def dependencies(self) -> List[str]:
        """Return the list of data fields this factor depends on."""
        ...

    @abc.abstractmethod
    def compute(
        self, data: DataProvider, dates: DateIndex, universe: Universe
    ) -> FactorMatrix:
        """Compute factor exposures for the given *dates* and *universe*."""
        ...


# ---------------------------------------------------------------------------
# Processing step
# ---------------------------------------------------------------------------


class ProcessingStep(abc.ABC):
    """A single step in the factor processing pipeline."""

    name: str = ""

    @abc.abstractmethod
    def transform(
        self, factor: FactorMatrix, context: ProcessingContext
    ) -> FactorMatrix:
        """Apply a transformation to the factor matrix."""
        ...


# ---------------------------------------------------------------------------
# Factor test
# ---------------------------------------------------------------------------


class FactorTest(abc.ABC):
    """Test harness for evaluating factor efficacy."""

    name: str = ""

    @abc.abstractmethod
    def run(
        self,
        factor: FactorMatrix,
        forward_returns: ReturnMatrix,
        universe: UniverseSchedule,
        **params,
    ) -> TestResult:
        """Run the test and return a TestResult instance."""
        ...


# ---------------------------------------------------------------------------
# Return model
# ---------------------------------------------------------------------------


class ReturnModel(abc.ABC):
    """Model that predicts expected returns from factor exposures."""

    @abc.abstractmethod
    def fit(
        self,
        factors: Dict[str, FactorMatrix],
        forward_returns: ReturnMatrix,
        universe: UniverseSchedule,
    ) -> ReturnModel:
        """Fit the return model on historical data."""
        ...

    @abc.abstractmethod
    def predict(
        self,
        factors: Dict[str, FactorMatrix],
        universe: Universe,
        date: Date,
    ) -> ExpectedReturns:
        """Predict expected returns for *universe* on *date*."""
        ...


# ---------------------------------------------------------------------------
# Risk model
# ---------------------------------------------------------------------------


class RiskModel(abc.ABC):
    """Model that estimates factor covariance and specific risk."""

    @abc.abstractmethod
    def estimate(
        self,
        data: DataProvider,
        factor_exposures: Dict[str, FactorMatrix],
        forward_returns: ReturnMatrix,
        universe: UniverseSchedule,
    ) -> RiskModel:
        """Estimate risk model parameters from historical data."""
        ...

    @abc.abstractmethod
    def covariance(
        self, date: Date, universe: Universe
    ) -> pd.DataFrame:
        """Return the factor covariance matrix on *date* for *universe*."""
        ...

    @abc.abstractmethod
    def specific_risk(
        self, date: Date, universe: Universe
    ) -> SpecificRisk:
        """Return the specific (idiosyncratic) risk vector on *date*."""
        ...

    @abc.abstractmethod
    def factor_exposure(
        self, weights: WeightVector, date: Date, universe: Universe
    ) -> pd.Series:
        """Return factor exposures of a weighted portfolio."""
        ...

    @abc.abstractmethod
    def portfolio_risk(
        self, weights: WeightVector, date: Date
    ) -> float:
        """Return the total risk (standard deviation) of the portfolio."""
        ...


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------


class Optimizer(abc.ABC):
    """Portfolio optimizer that produces optimal weight vectors."""

    @abc.abstractmethod
    def optimize(
        self,
        expected_returns: ExpectedReturns,
        risk_model: RiskModel,
        current_weights: WeightVector,
        constraints: List[Constraint],
        cost_model: Optional[CostModel],
        date: Date,
        universe: Universe,
    ) -> WeightVector:
        """Produce an optimal weight vector given the inputs."""
        ...


# ---------------------------------------------------------------------------
# Constraint
# ---------------------------------------------------------------------------


class Constraint(abc.ABC):
    """A portfolio optimisation constraint."""

    name: str = ""

    @abc.abstractmethod
    def apply(
        self,
        problem: "cvxpy.Problem",
        variables: Dict,
        context: ConstraintContext,
    ) -> None:
        """Apply the constraint to a cvxpy optimisation problem."""
        ...


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------


class CostModel(abc.ABC):
    """Estimates transaction costs for portfolio transitions."""

    @abc.abstractmethod
    def estimate_cost(
        self,
        target_weights: WeightVector,
        current_weights: WeightVector,
        date: Date,
    ) -> float:
        """Estimate transition cost on the date the target becomes effective."""
        ...

    def estimate_holding_cost(
        self,
        weights: WeightVector,
        date: Date,
    ) -> float:
        """Estimate a daily portfolio-level carry or management fee."""
        return 0.0


# ---------------------------------------------------------------------------
# Context dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ConstraintContext:
    """Context provided to a Constraint when solving an optimisation problem."""

    expected_returns: ExpectedReturns
    current_weights: WeightVector
    risk_model: RiskModel
    industry: IndustryMapping
    date: Date
    universe: Universe
    realized_vol: float = 0.0  # 近期组合已实现年化波动率 (用于 vol targeting)
    current_drawdown: float = 0.0  # 当前组合回撤 (用于 drawdown_control, 如 -0.05 = -5%)


@dataclass
class ProcessingContext:
    """Context provided to a ProcessingStep during factor processing."""

    data: DataProvider
    dates: DateIndex
    universe: Universe
    industry: Optional[pd.DataFrame | pd.Series] = None
    eligibility: Optional[pd.DataFrame] = None


# ---------------------------------------------------------------------------
# Test result
# ---------------------------------------------------------------------------


class TestResult(abc.ABC):
    """Result of a factor test evaluation."""

    @abc.abstractmethod
    def to_dict(self) -> dict:
        """Serialise the test result to a dictionary."""
        ...

    @abc.abstractmethod
    def summary(self) -> str:
        """Return a human-readable summary of the test result."""
        ...


# ---------------------------------------------------------------------------
# Backtest result
# ---------------------------------------------------------------------------


class BacktestResult(Protocol):
    """Protocol for backtest result objects.

    Implementations must provide at least the following attributes and methods.
    """

    nav: NAVSeries
    weights_history: pd.DataFrame

    def summary(self) -> str:
        """Return a human-readable backtest summary."""
        ...

    def plot(self, save_dir: str) -> None:
        """Plot backtest performance charts and optionally save them."""
        ...

    def export_target_weights(self, path: str, as_of=None) -> str:
        """Export a deterministic close-observed target-weight snapshot."""
        ...
