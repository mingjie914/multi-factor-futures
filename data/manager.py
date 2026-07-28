from __future__ import annotations
import logging

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from core.interfaces import DataProvider, DataSource
from core.types import *
from data.cache import Cache

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


class DataManager(DataProvider):
    """统一数据访问层. 聚合 DataSource + Cache, 提供因子计算所需的 DataProvider 接口.

    数据获取优先级: 本地缓存 → DataSource → 回退
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
        self._market = market_name
        self._cache = cache or Cache()
        self._config = config or {}
        cache_config = self._config.get("cache", {})
        self._cache_only = bool(cache_config.get("only", False))
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
        cache = Cache(
            cache_dir=cache_cfg.get("path", "./cache"),
            backend=cache_cfg.get("backend", "parquet"),
        ) if cache_cfg.get("enabled", True) else None

        source_name = dc.source

        # 确保数据源模块被 import (触发 @register 装饰器)
        if source_name in ("akshare_futures", "akshare"):
            try:
                from data import akshare_futures_source  # noqa: F401
            except ImportError:
                pass
        elif source_name == "mysql_futures":
            try:
                from data import mysql_source  # noqa: F401
            except ImportError:
                pass
        elif source_name == "ddb_futures":
            try:
                from data import ddb_source  # noqa: F401
            except ImportError:
                pass
        elif source_name == "parquet_futures":
            try:
                from data import parquet_source  # noqa: F401
            except ImportError:
                pass

        def _to_dict(obj):
            if obj is None:
                return {}
            if isinstance(obj, dict):
                return obj
            if hasattr(obj, "model_dump"):
                return obj.model_dump()
            return dict(obj)

        try:
            if source_name == "mysql_futures" and dc.mysql:
                source = create(
                    "data_source", source_name,
                    mysql_config=_to_dict(dc.mysql),
                )
            elif source_name == "ddb_futures" and dc.ddb:
                source = create(
                    "data_source", source_name,
                    ddb_config=_to_dict(dc.ddb),
                )
            elif source_name == "parquet_futures" and dc.parquet:
                source = create(
                    "data_source", source_name,
                    parquet_config=_to_dict(dc.parquet),
                    mysql_config=_to_dict(dc.mysql) if dc.mysql else None,
                )
            else:
                source = create("data_source", source_name)
        except Exception as e:
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
                self._source.__class__.__name__,
                field,
                universe,
                start,
                end,
            )
            if cached is not None:
                frame = cached.reindex(index=dates, columns=universe)
                return self._remember_prefetch(field, dates, universe, frame)

        # 从 DataSource 获取
        if self._cache_only:
            logger.warning(
                "cache-only miss: %s [%s~%s] universe=%s",
                field, start, end, len(universe),
            )
            frame = pd.DataFrame(index=dates, columns=universe, dtype=float)
            return self._remember_prefetch(field, dates, universe, frame)
        try:
            price_panel = self._source.fetch_price(
                universe, start, end, [field],
            )
        except Exception:
            logger.exception(f"fetch_price 失败: {field} [{start}~{end}]")
            price_panel = {}

        if field in price_panel:
            df = price_panel[field]
        else:
            df = pd.DataFrame(
                index=pd.DatetimeIndex([]), columns=universe,
            )

        # 确保 index 是 DatetimeIndex, 列对齐
        if not df.empty:
            df = df.reindex(index=dates, columns=universe)

        # 写缓存
        if self._cache and cache_enabled and not df.empty:
            try:
                self._cache.put(
                    self._market,
                    self._source.__class__.__name__,
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
        if self._cache_only:
            return pd.DataFrame(index=dates, columns=universe, dtype=object)
        industry = self._source.fetch_industry(
            universe, dates.min() if not dates.empty else None,
        )
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

        if self._cache_only:
            result = pd.DataFrame(columns=list(requested), dtype=float)
            self._macro_memory[memory_key] = result
            return result.copy(deep=True)

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
            result = pd.DataFrame(columns=list(requested), dtype=float)

        if result is None or result.empty:
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
        if self._cache_only:
            return pd.Series(dtype=object)
        if hasattr(self._source, "fetch_listing_dates"):
            try:
                return self._source.fetch_listing_dates(list(universe))
            except Exception:
                return pd.Series(dtype=object)
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
                self._source.__class__.__name__,
                f"contract_pair_{field}_near",
                universe,
                start,
                end,
            )
            cached_far = self._cache.get(
                self._market,
                self._source.__class__.__name__,
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

        if self._cache_only:
            empty = pd.DataFrame(index=dates, columns=universe, dtype=float)
            pair = {"near": empty.copy(), "far": empty.copy()}
            self._contract_pair_memory[memory_key] = pair
            return pair

        if not hasattr(self._source, "fetch_contract_pair_prices"):
            return {"near": pd.DataFrame(index=dates, columns=universe),
                    "far": pd.DataFrame(index=dates, columns=universe)}

        try:
            pair = self._source.fetch_contract_pair_prices(
                list(universe), start, end, field=field,
            )
        except Exception:
            logger.exception(f"fetch_contract_pair_prices 失败: {field} [{start}~{end}]")
            return {"near": pd.DataFrame(index=dates, columns=universe),
                    "far": pd.DataFrame(index=dates, columns=universe)}

        near = pair.get("near", pd.DataFrame())
        far = pair.get("far", pd.DataFrame())

        if not near.empty:
            near = near.reindex(index=dates, columns=universe)
        else:
            near = pd.DataFrame(index=dates, columns=universe)
        if not far.empty:
            far = far.reindex(index=dates, columns=universe)
        else:
            far = pd.DataFrame(index=dates, columns=universe)

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
                    self._source.__class__.__name__,
                    f"contract_pair_{field}_near",
                    universe,
                    start,
                    end,
                    near,
                )
                self._cache.put(
                    self._market,
                    self._source.__class__.__name__,
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
        """获取未来 N 期收益."""
        close = self.get("close", dates, universe)
        if close.empty:
            return pd.DataFrame(index=dates, columns=universe)
        return _forward_returns_on_valid_bars(close, period)

    def get_calendar(self, start: Date, end: Date) -> DateIndex:
        """获取交易日历."""
        if self._cache_only:
            return pd.date_range(start, end, freq="B")
        try:
            return self._source.fetch_calendar(start, end)
        except Exception:
            return pd.date_range(start, end, freq="B")

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
        except NotImplementedError:
            raise
        except Exception:
            logger.exception(
                f"fetch_price_at_frequency 失败: {field} [{start}~{end}] freq={frequency}"
            )
            price_panel = {}

        if field in price_panel:
            df = price_panel[field]
        else:
            df = pd.DataFrame(index=pd.DatetimeIndex([]), columns=universe)

        if not df.empty:
            df = df.reindex(index=dates, columns=universe)

        return df

    def prefetch(
        self,
        factors: list,
        dates: DateIndex = None,
        universe: Universe = None,
    ) -> None:
        """Load each dependency once for one exact factor-batch request."""
        all_fields: set = set()
        for factor in factors:
            if hasattr(factor, 'dependencies'):
                deps = factor.dependencies()
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
        for field in missing:
            frame = panel.get(field)
            if frame is not None and not frame.empty:
                result = frame.copy()
                result.index = pd.DatetimeIndex(result.index)
                self._panels[field] = result.sort_index().reindex(
                    columns=self._universe
                )
        self._loaded_fields.update(missing)

    def get_calendar(self) -> pd.DatetimeIndex:
        self._load(["close"])
        close = self._panels.get("close")
        if close is None or close.empty:
            return pd.DatetimeIndex([])
        return pd.DatetimeIndex(close.index.unique()).sort_values()

    def get(self, field: str, dates: DateIndex, universe: Universe) -> pd.DataFrame:
        self._load([field])
        frame = self._panels.get(field)
        if frame is None:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
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
        empty = pd.DataFrame(np.nan, index=dates, columns=universe)
        return {"near": empty.copy(), "far": empty.copy()}

    def prefetch(
        self,
        factors: list,
        dates: DateIndex = None,
        universe: Universe = None,
    ) -> None:
        fields = []
        for factor in factors:
            dependencies = factor.dependencies() if hasattr(factor, "dependencies") else []
            fields.extend(dependencies or [])
        self._load(fields)

    def clear_prefetch(self) -> None:
        return None
