from __future__ import annotations
import logging

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from core.interfaces import DataProvider, DataSource
from core.types import Date, DateIndex, PricePanel, ReturnMatrix, Universe
from data.cache import Cache
from data.market_quality import prepare_close_data

logger = logging.getLogger(__name__)


def _forward_returns_on_valid_bars(
    close: pd.DataFrame, period: int
) -> pd.DataFrame:
    """Compute horizons on each instrument's own observed-bar clock."""
    period = int(period)
    if period < 1:
        raise ValueError("forward-return period must be positive")
    result = pd.DataFrame(np.nan, index=close.index, columns=close.columns, dtype=float)
    for ticker in close.columns:
        series = pd.to_numeric(close[ticker], errors="coerce").dropna()
        if series.empty:
            continue
        forward = series.shift(-period) / series - 1.0
        result.loc[forward.index, ticker] = forward
    return result


def _forward_returns_on_calendar(
    close: pd.DataFrame, period: int
) -> pd.DataFrame:
    """Compute a daily horizon without jumping across missing calendar rows."""
    period = int(period)
    if period < 1:
        raise ValueError("forward-return period must be positive")
    prices = pd.DataFrame(close).apply(pd.to_numeric, errors="coerce")
    return prices.shift(-period).div(prices).sub(1.0)


class DataManager(DataProvider):
    """统一数据访问层. 聚合 DataSource + Cache, 提供因子计算所需的 DataProvider 接口.

    数据获取优先级: 本地缓存 → DataSource
    写入策略: 每次从 DataSource 取到数据后自动写缓存.
    """

    def __init__(
        self,
        source: DataSource,
        market_name: str = "futures",
        cache: Optional[Cache] = None,
        config: Optional[dict] = None,
    ) -> None:
        self._source = source
        self._source_cache_name = str(
            getattr(source, "cache_namespace", source.__class__.__name__)
        )
        self._market = market_name
        self._cache = cache
        self._config = config or {}
        self._prefetch_signature = None
        self._prefetched_data: Dict[str, pd.DataFrame] = {}
        self._contract_pair_memory: Dict[tuple, Dict[str, pd.DataFrame]] = {}
        self._macro_memory: Dict[tuple, pd.DataFrame] = {}

    @staticmethod
    def _request_signature(dates: DateIndex, universe: Universe) -> tuple:
        dates_index = pd.Index(dates)
        first = dates_index[0] if len(dates_index) else None
        last = dates_index[-1] if len(dates_index) else None
        date_hash = hash(
            pd.util.hash_pandas_object(dates_index, index=False)
            .to_numpy(copy=False)
            .tobytes()
        )
        return (
            len(dates_index),
            first,
            last,
            date_hash,
            tuple(str(item) for item in universe),
        )

    def _remember_prefetch(
        self,
        field: str,
        dates: DateIndex,
        universe: Universe,
        frame: pd.DataFrame,
    ) -> pd.DataFrame:
        if self._prefetch_signature == self._request_signature(dates, universe):
            self._prefetched_data[field] = frame
        return frame

    @staticmethod
    def _align_field_frame(
        panel: object,
        field: str,
        dates: DateIndex,
        universe: Universe,
        *,
        source_label: str,
    ) -> pd.DataFrame:
        if not isinstance(panel, dict):
            raise TypeError(f"{source_label} must return a field mapping")
        if field not in panel:
            raise KeyError(f"{source_label} omitted requested field {field!r}")
        frame = panel[field]
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"{source_label} field {field!r} must be a DataFrame")
        result = frame.copy()
        result.index = pd.DatetimeIndex(result.index)
        if result.index.has_duplicates or result.columns.has_duplicates:
            raise ValueError(f"{source_label} field {field!r} has duplicate axes")
        return result.sort_index().reindex(index=dates, columns=universe)

    @classmethod
    def from_config(cls, config) -> "DataManager":
        """从 FrameworkConfig 构造 DataManager (CR-030).

        复用与 PipelineRunner 相同的数据源工厂逻辑, 避免独立入口
        直接传 FrameworkConfig 给 DataManager(source=...) 导致类型不匹配.

        Args:
            config: FrameworkConfig 对象 (含 data/market 等字段)

        Returns:
            DataManager 实例
        """
        from core.registry import create
        from data.cache import Cache

        dc = config.data
        cache_cfg = dc.cache if isinstance(dc.cache, dict) else {}
        source_name = dc.source
        if source_name in {"parquet_futures", "duckdb_futures"} and cache_cfg.get("enabled", True):
            raise ValueError(
                "the generic range cache is not source-fingerprinted and must be "
                "disabled for parquet_futures/duckdb_futures; use the source-owned "
                "caches instead"
            )
        cache = Cache(
            cache_dir=cache_cfg.get("path", "./cache"),
            backend=cache_cfg.get("backend", "parquet"),
        ) if cache_cfg.get("enabled", True) else None

        # Import built-in sources to trigger registry decorators.  External
        # plugins may register their own source before calling this factory.
        if source_name == "parquet_futures":
            from data import parquet_source  # noqa: F401
        elif source_name == "duckdb_futures":
            from data import duckdb_source  # noqa: F401
        elif source_name == "random":
            from data import random_source  # noqa: F401

        def _to_dict(obj):
            if obj is None:
                return {}
            if isinstance(obj, dict):
                return obj
            if hasattr(obj, "model_dump"):
                return obj.model_dump()
            return dict(obj)

        try:
            if source_name == "parquet_futures":
                if dc.parquet is None:
                    raise ValueError("data.parquet is required for parquet_futures")
                source = create(
                    "data_source", source_name,
                    parquet_config=_to_dict(dc.parquet),
                )
            elif source_name == "duckdb_futures":
                if dc.duckdb is None or dc.parquet is None:
                    raise ValueError(
                        "data.duckdb and data.parquet are required for duckdb_futures"
                    )
                source = create(
                    "data_source", source_name,
                    duckdb_config=_to_dict(dc.duckdb),
                    parquet_config=_to_dict(dc.parquet),
                )
            else:
                source = create("data_source", source_name)
        except Exception:
            logger.exception(f"DataManager.from_config: 数据源 '{source_name}' 创建失败")
            raise

        return cls(
            source=source,
            market_name=config.market,
            cache=cache,
            config=_to_dict(dc) if not isinstance(dc, dict) else dc,
        )

    @property
    def source(self) -> DataSource:
        return self._source

    @property
    def cache(self) -> Optional[Cache]:
        return self._cache

    def prepare_close_data(
        self, close: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Apply the configured strict close-gap policy once for all callers."""
        return prepare_close_data(
            close,
            self._config.get("audited_nontrading_closes", {}),
        )

    def get_contract_schedule(
        self, dates: DateIndex, universe: Universe
    ) -> Optional[pd.DataFrame]:
        """Return the source's point-in-time concrete-contract schedule."""
        if dates.empty or len(universe) == 0:
            return None
        fetcher = getattr(self._source, "fetch_contract_schedule", None)
        if not callable(fetcher):
            return None
        schedule = fetcher(list(universe), dates.min(), dates.max())
        if schedule is None:
            return None
        if not isinstance(schedule, pd.DataFrame):
            raise TypeError("contract schedule must be a DataFrame or None")
        result = schedule.copy()
        result.index = pd.DatetimeIndex(result.index)
        if result.index.has_duplicates or result.columns.has_duplicates:
            raise ValueError("contract schedule must have unique dates and roots")
        return result.sort_index().reindex(index=dates, columns=universe)

    def get(
        self, field: str, dates: DateIndex, universe: Universe
    ) -> pd.DataFrame:
        """实现 DataProvider.get(). 返回 dates × tickers DataFrame."""
        if dates.empty or len(universe) == 0:
            return pd.DataFrame(index=dates, columns=universe)

        request_signature = self._request_signature(dates, universe)
        if self._prefetch_signature == request_signature:
            prefetched = self._prefetched_data.get(field)
            if prefetched is not None:
                return prefetched.copy(deep=True)

        start, end = dates.min(), dates.max()

        # 尝试缓存
        cache_enabled = self._config.get("cache", {}).get("enabled", True)
        if self._cache and cache_enabled:
            cached = self._cache.get(
                self._market,
                self._source_cache_name,
                field,
                universe,
                start,
                end,
            )
            if cached is not None:
                frame = cached.reindex(index=dates, columns=universe)
                return self._remember_prefetch(field, dates, universe, frame)

        # 从 DataSource 获取
        try:
            price_panel = self._source.fetch_price(
                universe, start, end, [field],
            )
        except Exception:
            logger.exception(f"fetch_price 失败: {field} [{start}~{end}]")
            raise

        df = self._align_field_frame(
            price_panel,
            field,
            dates,
            universe,
            source_label=f"{self._source.__class__.__name__}.fetch_price",
        )

        # 写缓存
        if self._cache and cache_enabled and not df.empty:
            try:
                self._cache.put(
                    self._market,
                    self._source_cache_name,
                    field,
                    universe,
                    start,
                    end,
                    df,
                )
            except Exception:
                logger.exception(f"Cache put 失败: {field} [{start}~{end}]")

        return self._remember_prefetch(field, dates, universe, df)

    def get_industry(
        self, dates: DateIndex, universe: Universe
    ) -> pd.DataFrame:
        """获取行业分类 (dates × tickers)."""
        industry = self._source.fetch_industry(
            universe, dates.min() if not dates.empty else None,
        )
        if not isinstance(industry, pd.Series):
            raise TypeError("fetch_industry must return a Series")
        # 扩展为日期维度
        result = pd.DataFrame(index=dates, columns=universe, dtype=object)
        if not industry.empty:
            for ticker in universe:
                if ticker in industry.index:
                    result[ticker] = industry[ticker]
        return result

    def get_universe(self, date: Date) -> Universe:
        """获取可投资标的集合."""
        return self._source.fetch_index_constituents("all", date)

    def get_macro(
        self,
        fields: List[str],
        start: Optional[Date] = None,
        end: Optional[Date] = None,
    ) -> pd.DataFrame:
        """获取按观察月份索引的宏观数据，并在进程内复用结果."""
        requested = tuple(dict.fromkeys(str(field) for field in fields))
        if not requested:
            return pd.DataFrame()

        start_ts = pd.Timestamp(start) if start is not None else None
        end_ts = pd.Timestamp(end) if end is not None else None
        memory_key = (requested, start_ts, end_ts)
        cached = self._macro_memory.get(memory_key)
        if cached is not None:
            return cached.copy(deep=True)

        try:
            result = self._source.fetch_macro(
                list(requested), start=start_ts, end=end_ts
            )
        except Exception:
            logger.exception(
                "fetch_macro 失败: fields=%s [%s~%s]",
                ",".join(requested),
                start_ts,
                end_ts,
            )
            raise

        if result is None:
            result = pd.DataFrame(columns=list(requested), dtype=float)
        elif not isinstance(result, pd.DataFrame):
            raise TypeError("fetch_macro must return a DataFrame or None")
        elif result.empty:
            result = pd.DataFrame(columns=list(requested), dtype=float)
        else:
            result = result.copy()
            result.index = pd.DatetimeIndex(result.index)
            result = result.sort_index().reindex(columns=list(requested))
            result = result.apply(pd.to_numeric, errors="coerce")

        self._macro_memory[memory_key] = result
        return result.copy(deep=True)

    def get_listing_dates(self, universe: Universe) -> pd.Series:
        """获取品种上市日期.

        Returns:
            pd.Series: index=品种代码, values=上市日期(datetime)
                      若数据源不支持则返回空 Series
        """
        if hasattr(self._source, "fetch_listing_dates"):
            result = self._source.fetch_listing_dates(list(universe))
            if not isinstance(result, pd.Series):
                raise TypeError("fetch_listing_dates must return a Series")
            return result
        return pd.Series(dtype=object)

    def get_contract_pair(
        self, field: str, dates: DateIndex, universe: Universe
    ) -> Dict[str, pd.DataFrame]:
        """获取近月/远月合约对的价格 (用于展期收益率等多合约因子).

        Args:
            field: 价格字段 (close/settle/open 等).
            dates: 日期索引.
            universe: 品种根代码集合 (如 ['RB', 'IF']).

        Returns:
            {"near": DataFrame(dates × roots), "far": DataFrame(dates × roots)}
            近月/远月原始价格 (未复权, 允许换月跳空).
            数据源不支持时返回两个空 DataFrame.
        """
        if dates.empty or len(universe) == 0:
            return {"near": pd.DataFrame(index=dates, columns=universe),
                    "far": pd.DataFrame(index=dates, columns=universe)}

        signature = self._request_signature(dates, universe)
        memory_key = (field, signature)
        if memory_key in self._contract_pair_memory:
            pair = self._contract_pair_memory[memory_key]
            return {
                "near": pair["near"].copy(deep=True),
                "far": pair["far"].copy(deep=True),
            }

        start, end = dates.min(), dates.max()
        cache_enabled = self._config.get("cache", {}).get("enabled", True)
        if self._cache and cache_enabled:
            cached_near = self._cache.get(
                self._market,
                self._source_cache_name,
                f"contract_pair_{field}_near",
                universe,
                start,
                end,
            )
            cached_far = self._cache.get(
                self._market,
                self._source_cache_name,
                f"contract_pair_{field}_far",
                universe,
                start,
                end,
            )
            if cached_near is not None and cached_far is not None:
                pair = {
                    "near": cached_near.reindex(index=dates, columns=universe),
                    "far": cached_far.reindex(index=dates, columns=universe),
                }
                self._contract_pair_memory[memory_key] = pair
                return {
                    "near": pair["near"].copy(deep=True),
                    "far": pair["far"].copy(deep=True),
                }

        if not hasattr(self._source, "fetch_contract_pair_prices"):
            raise NotImplementedError(
                f"{self._source.__class__.__name__} does not provide contract pairs"
            )

        try:
            pair = self._source.fetch_contract_pair_prices(
                list(universe), start, end, field=field,
            )
        except Exception:
            logger.exception(f"fetch_contract_pair_prices 失败: {field} [{start}~{end}]")
            raise

        if not isinstance(pair, dict) or not {"near", "far"}.issubset(pair):
            raise TypeError(
                "fetch_contract_pair_prices must return near/far DataFrames"
            )

        near = pair["near"]
        far = pair["far"]
        if not isinstance(near, pd.DataFrame) or not isinstance(far, pd.DataFrame):
            raise TypeError("contract-pair near/far values must be DataFrames")
        if (
            near.index.has_duplicates
            or near.columns.has_duplicates
            or far.index.has_duplicates
            or far.columns.has_duplicates
        ):
            raise ValueError("contract-pair panels must have unique axes")

        if not near.empty:
            near = near.reindex(index=dates, columns=universe)
        else:
            near = pd.DataFrame(index=dates, columns=universe)
        if not far.empty:
            far = far.reindex(index=dates, columns=universe)
        else:
            far = pd.DataFrame(index=dates, columns=universe)
        if near.isna().all().all() or far.isna().all().all():
            raise RuntimeError(
                f"contract-pair source returned no usable {field} data"
            )

        pair = {"near": near, "far": far}
        self._contract_pair_memory[memory_key] = pair
        if (
            self._cache
            and cache_enabled
            and not near.empty
            and not far.empty
        ):
            try:
                self._cache.put(
                    self._market,
                    self._source_cache_name,
                    f"contract_pair_{field}_near",
                    universe,
                    start,
                    end,
                    near,
                )
                self._cache.put(
                    self._market,
                    self._source_cache_name,
                    f"contract_pair_{field}_far",
                    universe,
                    start,
                    end,
                    far,
                )
            except Exception:
                logger.exception(
                    "contract-pair cache put failed: %s [%s~%s]",
                    field,
                    start,
                    end,
                )
        return {
            "near": near.copy(deep=True),
            "far": far.copy(deep=True),
        }

    def get_price_panel(
        self,
        dates: DateIndex,
        universe: Universe,
        fields: Optional[List[str]] = None,
    ) -> PricePanel:
        """批量获取价格面板."""
        fields = fields or ["open", "high", "low", "close", "volume", "oi"]
        panel: PricePanel = {}
        for f in fields:
            df = self.get(f, dates, universe)
            if not df.empty:
                panel[f] = df
        return panel

    def get_forward_returns(
        self, dates: DateIndex, universe: Universe, period: int = 1
    ) -> ReturnMatrix:
        """获取严格按日度交易日历对齐的未来 N 期收益."""
        close = self.get("close", dates, universe)
        if close.empty:
            raise RuntimeError("cannot compute forward returns from an empty close panel")
        # Validate unknown post-listing gaps before marking audited closures.
        self.prepare_close_data(close)
        marked = close.apply(pd.to_numeric, errors="coerce").ffill()
        return _forward_returns_on_calendar(marked, period)

    def get_calendar(self, start: Date, end: Date) -> DateIndex:
        """获取交易日历."""
        calendar = pd.DatetimeIndex(self._source.fetch_calendar(start, end))
        if calendar.has_duplicates:
            raise ValueError("trading calendar contains duplicate dates")
        return calendar.sort_values()

    def get_at_frequency(
        self,
        field: str,
        dates: DateIndex,
        universe: Universe,
        frequency: str = "daily",
    ) -> pd.DataFrame:
        """频率感知的数据获取 (周期架构扩展).

        当 frequency == "daily" 时回退到现有 get() (含缓存), 行为完全一致;
        当 frequency 为其他值 (如 "15min") 时, 调用 source.fetch_price_at_frequency
        获取分钟数据, 不走日度缓存.

        Args:
            field: 数据字段 (如 "close", "volume")
            dates: 日期/时间索引
            universe: 标的集合
            frequency: 周期单位 ("daily" / "15min" / "30min" / "hourly")

        Returns:
            DataFrame(dates × tickers)

        Raises:
            NotImplementedError: 当数据源不支持指定频率时
        """
        if frequency == "daily":
            return self.get(field, dates, universe)

        # 非日度频率: 路由到 DataSource.fetch_price_at_frequency
        if dates.empty or len(universe) == 0:
            return pd.DataFrame(index=dates, columns=universe)

        start, end = dates.min(), dates.max()
        try:
            price_panel = self._source.fetch_price_at_frequency(
                universe, start, end, [field], frequency=frequency,
            )
        except Exception:
            logger.exception(
                f"fetch_price_at_frequency 失败: {field} [{start}~{end}] freq={frequency}"
            )
            raise

        return self._align_field_frame(
            price_panel,
            field,
            dates,
            universe,
            source_label=(
                f"{self._source.__class__.__name__}.fetch_price_at_frequency"
            ),
        )

    def prefetch(
        self,
        factors: list,
        dates: DateIndex = None,
        universe: Universe = None,
    ) -> None:
        """Load each dependency once for one exact factor-batch request."""
        all_fields: set = set()
        for factor in factors:
            dependency_loader = getattr(factor, "prefetch_dependencies", None)
            if dependency_loader is None:
                dependency_loader = getattr(factor, "dependencies", None)
            if dependency_loader is not None:
                deps = dependency_loader()
                if deps:
                    all_fields.update(deps)
        if dates is None or universe is None:
            return
        signature = self._request_signature(dates, universe)
        if self._prefetch_signature != signature:
            self._prefetched_data = {}
        self._prefetch_signature = signature
        for field in sorted(all_fields):
            if field not in self._prefetched_data:
                self.get(field, dates, universe)

    def clear_prefetch(self) -> None:
        self._prefetch_signature = None
        self._prefetched_data.clear()


class FrequencyDataProvider(DataProvider):
    """Provide non-daily research data on its real bar index."""

    def __init__(
        self,
        manager: DataManager,
        frequency: str,
        start,
        end,
        universe: Universe,
    ) -> None:
        if str(frequency).lower() == "daily":
            raise ValueError("FrequencyDataProvider is only for non-daily bars")
        self._manager = manager
        self.frequency = str(frequency).lower()
        self._start = pd.Timestamp(start)
        self._end = pd.Timestamp(end)
        self._universe = pd.Index(universe)
        self._panels: Dict[str, pd.DataFrame] = {}
        self._loaded_fields: set[str] = set()

    def _load(self, fields: List[str]) -> None:
        missing = [
            str(field) for field in dict.fromkeys(fields)
            if str(field) not in self._loaded_fields
        ]
        if not missing:
            return
        panel = self._manager.source.fetch_price_at_frequency(
            self._universe,
            self._start,
            self._end,
            missing,
            frequency=self.frequency,
        )
        if not isinstance(panel, dict):
            raise TypeError("fetch_price_at_frequency must return a field mapping")
        for field in missing:
            if field not in panel:
                raise KeyError(
                    f"fetch_price_at_frequency omitted requested field {field!r}"
                )
            frame = panel[field]
            if not isinstance(frame, pd.DataFrame):
                raise TypeError(f"intraday field {field!r} must be a DataFrame")
            result = frame.copy()
            result.index = pd.DatetimeIndex(result.index)
            if result.index.has_duplicates or result.columns.has_duplicates:
                raise ValueError(f"intraday field {field!r} has duplicate axes")
            self._panels[field] = result.sort_index().reindex(
                columns=self._universe
            )
        self._loaded_fields.update(missing)

    def get_calendar(self) -> pd.DatetimeIndex:
        self._load(["close"])
        close = self._panels.get("close")
        if close is None or close.empty:
            raise RuntimeError(f"{self.frequency} close calendar is empty")
        return pd.DatetimeIndex(close.index.unique()).sort_values()

    def get(self, field: str, dates: DateIndex, universe: Universe) -> pd.DataFrame:
        self._load([field])
        frame = self._panels.get(field)
        if frame is None:
            raise KeyError(f"intraday field {field!r} was not loaded")
        return frame.reindex(index=dates, columns=universe)

    def get_at_frequency(
        self,
        field: str,
        dates: DateIndex,
        universe: Universe,
        frequency: str = "daily",
    ) -> pd.DataFrame:
        if str(frequency).lower() == self.frequency:
            return self.get(field, dates, universe)
        return self._manager.get_at_frequency(field, dates, universe, frequency)

    def get_forward_returns(
        self, dates: DateIndex, universe: Universe, period: int = 1
    ) -> ReturnMatrix:
        close = self.get("close", dates, universe)
        return _forward_returns_on_valid_bars(close, period)

    def get_industry(
        self, dates: DateIndex, universe: Universe
    ) -> pd.DataFrame:
        return self._manager.get_industry(dates, universe)

    def get_universe(self, date: Date) -> Universe:
        return self._manager.get_universe(date)

    def get_macro(
        self,
        fields: List[str],
        start: Optional[Date] = None,
        end: Optional[Date] = None,
    ) -> pd.DataFrame:
        return self._manager.get_macro(fields, start=start, end=end)

    def get_contract_pair(
        self, field: str, dates: DateIndex, universe: Universe
    ) -> Dict[str, pd.DataFrame]:
        raise NotImplementedError(
            f"contract-pair data is not defined on {self.frequency} bars"
        )

    def prefetch(
        self,
        factors: list,
        dates: DateIndex = None,
        universe: Universe = None,
    ) -> None:
        fields = []
        for factor in factors:
            dependency_loader = getattr(
                factor, "prefetch_dependencies", getattr(factor, "dependencies", None)
            )
            dependencies = dependency_loader() if dependency_loader else []
            fields.extend(dependencies or [])
        self._load(fields)

    def clear_prefetch(self) -> None:
        return None
