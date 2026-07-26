"""随机数据源 — 用于测试框架流程, 不需要任何外部数据.
生成随机期货 OHLCV + OI 数据, 让因子计算和回测产生有效结果.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List
from core.types import *
from core.interfaces import DataSource
from core.registry import register

_TICKERS = ['RB.SHFE', 'CU.SHFE', 'AU.SHFE', 'AG.SHFE', 'SC.INE',
            'TA.CZCE', 'MA.CZCE', 'I.DCE', 'J.DCE', 'JM.DCE',
            'NI.SHFE', 'ZN.SHFE', 'AL.SHFE', 'SR.CZCE', 'CF.CZCE']


@register("data_source", "random")
class RandomDataSource(DataSource):
    """随机数据源: 生成模拟期货日线数据.
    无需网络, 无需任何第三方包, 纯 numpy 生成.
    """
    market = "futures"

    def __init__(self, seed: int = 42, n_assets: int = 10, **kwargs):
        self._rng = np.random.RandomState(seed)
        self._n_assets = min(n_assets, len(_TICKERS))
        self._tickers = _TICKERS[:self._n_assets]

    def _generate_panel(self, tickers, start, end,
                        fields: List[str]) -> PricePanel:
        dates = pd.date_range(start, end, freq="B")
        if len(dates) == 0:
            return {}

        result = {}
        for field in fields:
            data = {}
            for t in tickers:
                # 随机游走生成价格序列
                n = len(dates)
                ret = self._rng.randn(n) * 0.02  # 日收益 ~2% 波动
                price = 100 * np.exp(np.cumsum(ret))
                if field == "close":
                    series = pd.Series(price, index=dates, name=t)
                elif field == "open":
                    series = pd.Series(price * (1 + self._rng.randn(n) * 0.005), index=dates, name=t)
                elif field in ("high", "low"):
                    offset = self._rng.rand(n) * price * 0.02
                    series = pd.Series(price + offset if field == "high" else price - offset, index=dates, name=t)
                elif field == "volume":
                    series = pd.Series(self._rng.randint(10000, 500000, n), index=dates, name=t)
                elif field == "oi":
                    series = pd.Series(
                        np.maximum(1000, 10000 + np.cumsum(self._rng.randint(-500, 500, n))),
                        index=dates, name=t
                    )
                elif field in ("settle", "pre_settle"):
                    series = pd.Series(price, index=dates, name=t)
                else:
                    series = pd.Series(self._rng.randn(n), index=dates, name=t)
                data[t] = series

            panel = pd.DataFrame(data)
            panel.index.name = "date"
            if not panel.empty and all(col in panel.columns for col in tickers):
                result[field] = panel[tickers]
            else:
                result[field] = pd.DataFrame(index=dates, columns=tickers)

        return result

    def fetch_price(self, tickers: TickerIndex, start: Date, end: Date,
                    fields: List[str]) -> PricePanel:
        return self._generate_panel(tickers, start, end, fields)

    def fetch_fundamental(self, tickers: TickerIndex, start: Date, end: Date,
                          fields: List[str]) -> dict:
        return {}

    def fetch_industry(self, tickers: TickerIndex, date: Date) -> IndustryMapping:
        return pd.Series({t: t[:2] for t in tickers})

    def fetch_index_constituents(self, index_code: str, date: Date) -> Universe:
        return pd.Index(self._tickers)

    def fetch_calendar(self, start: Date, end: Date) -> DateIndex:
        return pd.bdate_range(start, end)
