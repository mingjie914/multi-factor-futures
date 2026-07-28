from __future__ import annotations

from typing import Dict, List

import numpy as np  # noqa: F401
import pandas as pd

# ⚠️ akshare 是延迟导入的 (在每个方法内部 import)
# 这样即使没有装 akshare, 框架其他部分也能正常导入.
_Akshare = None

def _get_ak() -> "ak":
    """延迟获取 akshare 模块."""
    global _Akshare
    if _Akshare is None:
        import akshare as _Akshare
    return _Akshare

from core.interfaces import DataSource
from core.registry import register
from core.types import *


@register("data_source", "akshare_futures")
class AkshareFuturesSource(DataSource):
    """akshare 期货数据源. 免费，有缺失，适合 MVP.

    提供:
    - 新浪主力连续合约（日线）
    - 基础日线 (OHLCV)
    - 交易日历
    """
    market = "futures"
    formal_research_safe = False

    def __init__(self, **kwargs) -> None:
        # 缓存已获取的 DataFrame 避免重复 API 调用
        self._cache: dict = {}

    def fetch_price(
        self,
        tickers: TickerIndex,
        start: Date,
        end: Date,
        fields: List[str],
    ) -> PricePanel:
        """拉取期货日线行情.

        AKShare ``futures_main_sina`` 使用 ``RB0``/``IF0`` 形式的主力
        连续代码，并返回中文字段。这里显式完成代码和字段适配；该免费源
        不提供可审计的逐次换月台账，因此不应用于正式准入研究。
        """
        result: PricePanel = {}
        for ticker in tickers:
            root = str(ticker).split(".")[0].upper()
            sym = f"{root}0"
            # 尝试获取主力连续
            try:
                ak = _get_ak()
                df = ak.futures_main_sina(symbol=sym)
            except Exception:
                continue
            if df is None or df.empty:
                continue
            df = df.rename(columns={
                'date': 'date', 'open': 'open', 'high': 'high',
                'low': 'low', 'close': 'close', 'volume': 'volume',
                'hold': 'oi', 'close_0': 'settle',
                '日期': 'date', '开盘价': 'open', '最高价': 'high',
                '最低价': 'low', '收盘价': 'close', '成交量': 'volume',
                '持仓量': 'oi', '动态结算价': 'settle',
            })
            if "date" not in df.columns:
                continue
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            mask = (df['date'] >= pd.Timestamp(start)) & (
                df['date'] <= pd.Timestamp(end))
            df = df[mask]
            if df.empty:
                continue
            for fld in fields:
                if fld in df.columns:
                    series = df[['date', fld]].set_index('date')[fld]
                    series.name = ticker
                    if fld not in result:
                        result[fld] = series.to_frame()
                    else:
                        result[fld] = result[fld].join(series, how='outer')
        return result

    def fetch_fundamental(
        self,
        tickers: TickerIndex,
        start: Date,
        end: Date,
        fields: List[str],
    ) -> dict:
        """akshare 期货无基本面数据."""
        return {}

    def fetch_industry(
        self,
        tickers: TickerIndex,
        date: Date,
    ) -> IndustryMapping:
        return pd.Series(dtype=object)

    def fetch_index_constituents(
        self,
        index_code: str,
        date: Date,
    ) -> Universe:
        return pd.Index([])

    def fetch_calendar(self, start: Date, end: Date) -> DateIndex:
        try:
            ak = _get_ak()
            df = ak.tool_trade_date_hist_sina()
        except Exception:
            return pd.DatetimeIndex([])
        df = df[df['trade_date'].between(
            pd.Timestamp(start), pd.Timestamp(end))]
        return pd.DatetimeIndex(df['trade_date'].sort_values())
