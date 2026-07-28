"""DolphinDB 数据源封装.

支持:
1. 标准 DataSource 接口 (日度 OHLCV, 可替代 MySQLSource)
2. fetch_minute_bars(): 分钟级 K 线数据获取
3. fetch_intraday_features(): 分钟数据聚合为日度日内特征

设计要点:
- 分钟数据在 DDB 中按合约存储, 本模块负责查询+聚合
- 聚合后的日度特征 (vwap, intraday_ret, overnight_gap 等) 通过
  fetch_price() 标准接口暴露, 无需修改因子框架
- 连接管理: 每次查询建立连接, 查完关闭 (由 data.dolphindb_client 管理)

配置 (config/default.yaml):
    ddb:
      host: '${MF_DDB_HOST}'
      port: 8961
      user: '${MF_DDB_USER}'
      password: '${MF_DDB_PASSWORD}'
      minute_db: 'dfs://kline_db'        # 分钟K线库
      minute_table: 'kline_futures_1min'  # 期货分钟K线表名
      eod_db: 'dfs://wind_db'             # 日度行情库
      eod_table: 'CCommodityFuturesEODPrices'
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from core.interfaces import DataSource
from core.registry import register
from core.types import *

logger = logging.getLogger(__name__)

# 日内特征字段列表 (聚合后暴露给因子框架)
INTRADAY_FIELDS = [
    "vwap",              # 成交量加权均价
    "intraday_return",   # 日内收益 (close - open) / open
    "overnight_gap",     # 隔夜跳空 (open - prev_close) / prev_close
    "intraday_volatility",  # 日内波动率 (high-low)/open
    "close_to_vwap",     # 收盘相对VWAP偏离 (close-vwap)/vwap
    "volume_concentration",  # 成交量集中度 (peak_vol / avg_vol)
    "amihud_illiquidity",   # Amihud非流动性 |ret| / amount
    "tail_momentum",     # 尾盘动量 (close - last_30min_close) / close
]


@register("data_source", "ddb_futures")
class DDBSource(DataSource):
    """DolphinDB 期货数据源.

    支持日度行情 + 分钟K线 + 日内特征聚合.
    可替代 MySQLSource 作为主数据源, 也可作为补充数据源提供日内特征.

    优先级:
    - 日度 OHLCV: 优先 DDB (实时性更好), 回退 MySQL
    - 日内特征: 仅 DDB (MySQL 无分钟数据)
    """

    market = "futures"

    def __init__(self, ddb_config: Optional[dict] = None, **kwargs) -> None:
        self._config = ddb_config or {}
        self._host = self._config.get("host", "")
        self._port = self._config.get("port", 8961)
        self._user = self._config.get("user", "")
        self._password = self._config.get("password", "")
        self._minute_db = self._config.get("minute_db", "dfs://kline_db")
        self._minute_table = self._config.get("minute_table", "kline_futures_1min")
        self._eod_db = self._config.get("eod_db", "dfs://wind_db")
        self._eod_table = self._config.get("eod_table", "CCommodityFuturesEODPrices")
        self._data_version = str(self._config.get("data_version", "v1"))
        self._dominant_lag_days = max(int(self._config.get("dominant_lag_days", 1)), 1)

        # 缓存: 日内特征 (避免重复查询)
        self._intraday_cache: Dict[tuple, Dict[str, pd.DataFrame]] = {}

    def _get_connection(self):
        """获取 DDB 连接 (用完需 close)."""
        from data import dolphindb_client as _gdb

        # Use credentials supplied by local.yaml or MF_DDB_* overrides.
        _gdb.user = self._user
        _gdb.password = self._password
        conn = _gdb.get_Conn(self._host, self._port)
        if conn is None:
            raise ConnectionError(f"DDB 连接失败: {self._host}:{self._port}")
        return conn

    def _query(self, script: str) -> pd.DataFrame:
        """执行 DDB 查询, 自动管理连接."""
        from data import dolphindb_client as _gdb

        _gdb.user = self._user
        _gdb.password = self._password
        return _gdb.fetchData(script, self._host, self._port)

    # ------------------------------------------------------------------
    # DataSource 接口实现
    # ------------------------------------------------------------------

    def fetch_price(
        self,
        tickers: TickerIndex,
        start: Date,
        end: Date,
        fields: List[str],
    ) -> PricePanel:
        """从 DDB 获取日度行情.

        对标准 OHLCV 字段, 查询日度行情表.
        对日内特征字段 (vwap 等), 查询分钟K线表并聚合.
        """
        standard_fields = {"open", "high", "low", "close", "volume",
                          "amount", "oi", "settle", "pre_settle"}
        intraday_fields = set(fields) - standard_fields

        result: PricePanel = {}

        # 标准日度字段
        std_to_fetch = [f for f in fields if f in standard_fields]
        if std_to_fetch:
            panel = self._fetch_eod_price(tickers, start, end, std_to_fetch)
            result.update(panel)

        # 日内特征字段 (从分钟数据聚合)
        if intraday_fields:
            intraday_panel = self._fetch_intraday_features(
                tickers, start, end, list(intraday_fields)
            )
            result.update(intraday_panel)

        return result

    def _fetch_eod_price(
        self,
        tickers: TickerIndex,
        start: Date,
        end: Date,
        fields: List[str],
    ) -> PricePanel:
        """从 DDB 日度行情表获取标准 OHLCV.

        CR-015: 始终附带查询 volume, 按每个交易日仅使用当日可见的成交量
        选择主力合约, 不再对整个查询区间选择非空记录最多的单一合约.
        """
        requested_start = pd.Timestamp(start)
        requested_end = pd.Timestamp(end)
        query_start = requested_start - pd.Timedelta(days=14)
        start_str = query_start.strftime("%Y.%m.%d")
        end_str = pd.Timestamp(end).strftime("%Y.%m.%d")

        # DDB 字段映射 (Wind 格式 → 标准字段)
        field_map = {
            "open": "S_DQ_OPEN",
            "high": "S_DQ_HIGH",
            "low": "S_DQ_LOW",
            "close": "S_DQ_CLOSE",
            "settle": "S_DQ_SETTLE",
            "pre_settle": "S_DQ_PRESETTLE",
            "volume": "S_DQ_VOLUME",
            "amount": "S_DQ_AMOUNT",
            "oi": "S_DQ_OI",
        }
        # CR-015: 始终查询 volume 用于按日选择主力合约
        query_fields = list(dict.fromkeys(fields + ["volume", "close"]))
        ddb_cols = [field_map.get(f, f) for f in query_fields]
        col_select = ", ".join(ddb_cols)

        # 构建品种过滤条件 (Wind 代码格式: RB.SHF)
        ticker_filter = " or ".join(
            f"S_INFO_WINDCODE like '{t}.%'" for t in tickers
        )

        script = f"""
        t = loadTable("{self._eod_db}", "{self._eod_table}")
        select TRADE_DT, S_INFO_WINDCODE, {col_select} from t
        where ({ticker_filter})
        and TRADE_DT >= {start_str} and TRADE_DT <= {end_str}
        order by S_INFO_WINDCODE, TRADE_DT
        """
        df = self._query(script)
        if df.empty:
            return {}

        # 转换为 PricePanel
        df["TRADE_DT"] = pd.to_datetime(df["TRADE_DT"].astype(str))

        # CR-015: 先用 volume 构建每日主力合约映射 (避免使用全查询期信息)
        vol_col = field_map["volume"]
        dominant_map = None
        if vol_col in df.columns:
            vol_pivot = df.pivot_table(
                index="TRADE_DT", columns="S_INFO_WINDCODE",
                values=vol_col, aggfunc="first",
            )
            dominant_map = self._build_daily_dominant_contract(vol_pivot)
        if not dominant_map:
            logger.error("DDB 日度数据缺少可用成交量，拒绝使用全样本合约回退")
            return {}
        close_pivot = None
        roll_scales = None
        close_col = field_map["close"]
        if dominant_map is not None and close_col in df.columns:
            close_pivot = df.pivot_table(
                index="TRADE_DT", columns="S_INFO_WINDCODE",
                values=close_col, aggfunc="first",
            )
            roll_scales = self._build_roll_scales(close_pivot, dominant_map)

        result: PricePanel = {}
        for field in fields:
            ddb_col = field_map.get(field, field)
            if ddb_col in df.columns:
                pivot = df.pivot_table(
                    index="TRADE_DT", columns="S_INFO_WINDCODE",
                    values=ddb_col, aggfunc="first",
                )
                # CR-015: 按日选择主力合约 (不再对全查询期选单一合约)
                pivot = self._aggregate_to_root_by_daily_volume(
                    pivot,
                    dominant_map,
                    roll_scales=roll_scales,
                    adjust_prices=field in {
                        "open", "high", "low", "close", "settle", "pre_settle"
                    },
                )
                result[field] = pivot.loc[
                    (pivot.index >= requested_start) & (pivot.index <= requested_end)
                ]
        return result

    def _build_daily_dominant_contract(
        self, vol_pivot: pd.DataFrame
    ) -> Dict[str, "pd.Series"]:
        """CR-015: 构建每日主力合约映射.

        对每个品种根代码, 在每个交易日选择当日成交量最大的合约作为主力.
        这避免了旧实现对整个查询区间选择非空记录最多的单一合约 (使用未来信息).

        Args:
            vol_pivot: 成交量 pivot 表 (日期×合约)

        Returns:
            {root: pd.Series(index=date, values=contract_code)} 每日主力合约
        """
        if vol_pivot.empty:
            return {}

        # 合约代码 → 品种根代码
        root_map: Dict[str, list] = {}
        for col in vol_pivot.columns:
            root = str(col).split(".")[0].rstrip("0123456789")
            if root:
                root_map.setdefault(root, []).append(col)

        dominant_map: Dict[str, "pd.Series"] = {}
        for root, cols in root_map.items():
            if len(cols) == 1:
                # 只有一个合约, 每天都用它
                raw_dominant = pd.Series(cols[0], index=vol_pivot.index)
            else:
                sub = vol_pivot[cols]
                # 将 NaN 填为 -1, 不会选到没有数据的合约
                sub_filled = sub.fillna(-1)
                # 每行最大值的列名 = 当日成交量最大的合约
                dominant = sub_filled.idxmax(axis=1)
                # 全 NaN 的日期标记为 None
                all_nan = sub.isna().all(axis=1)
                dominant = dominant.where(~all_nan)
                raw_dominant = dominant
            # A contract selected with day-t volume can only become effective
            # after that close. Shift by trading rows, not calendar days.
            dominant_map[root] = raw_dominant.shift(self._dominant_lag_days)

        return dominant_map

    def _aggregate_to_root_by_daily_volume(
        self,
        df: pd.DataFrame,
        dominant_map: Dict[str, "pd.Series"],
        roll_scales: Optional[Dict[str, "pd.Series"]] = None,
        adjust_prices: bool = False,
    ) -> pd.DataFrame:
        """CR-015: 按每日主力合约映射将合约级数据聚合为品种根代码级.

        对每个交易日, 仅使用当日主力合约的值,
        不再对整个查询区间选非空最多的单一合约.
        """
        if df.empty:
            return df

        root_map: Dict[str, list] = {}
        for col in df.columns:
            root = str(col).split(".")[0].rstrip("0123456789")
            if root:
                root_map.setdefault(root, []).append(col)

        result: Dict[str, pd.Series] = {}
        for root, cols in root_map.items():
            if root not in dominant_map:
                result[root] = pd.Series(np.nan, index=df.index, dtype=float)
                continue

            dominant = dominant_map[root]
            sub = df[cols]
            # 向量化: 构建每行的选择索引
            col_to_idx = {c: i for i, c in enumerate(sub.columns)}
            select_idx = dominant.map(col_to_idx)  # (n_dates,) 或 NaN
            vals = sub.values  # (n_dates, n_contracts)
            n = len(sub)
            row_idx = np.arange(n)
            # 处理 NaN 索引 (dominant 为 None 的日期)
            valid_mask = select_idx.notna().values
            result_vals = np.full(n, np.nan, dtype=np.float64)
            if valid_mask.any():
                result_vals[valid_mask] = vals[
                    row_idx[valid_mask],
                    select_idx.values[valid_mask].astype(int),
                ]
            if adjust_prices and roll_scales and root in roll_scales:
                scale = roll_scales[root].reindex(df.index).to_numpy(dtype=float)
                result_vals = result_vals * scale
            result[root] = pd.Series(result_vals, index=df.index)

        return pd.DataFrame(result)

    def _build_roll_scales(
        self,
        close_pivot: pd.DataFrame,
        dominant_map: Dict[str, "pd.Series"],
    ) -> Dict[str, "pd.Series"]:
        """Build ratio scales from the latest common close, failing closed."""
        from data.continuous_contract import RolloverAdjustmentError

        scales: Dict[str, pd.Series] = {}
        for root, dominant in dominant_map.items():
            columns = [
                column for column in close_pivot.columns
                if str(column).split(".")[0].rstrip("0123456789") == root
            ]
            if not columns:
                continue
            prices = close_pivot[columns]
            schedule = dominant.reindex(prices.index)
            scale = pd.Series(np.nan, index=prices.index, dtype=float)
            previous_contract = None
            previous_scale = 1.0
            previous_date = None
            for date, contract in schedule.items():
                if pd.isna(contract) or contract not in prices.columns:
                    previous_date = date
                    continue
                if previous_contract is None:
                    current_scale = 1.0
                elif contract == previous_contract:
                    current_scale = previous_scale
                else:
                    cutoff = pd.Timestamp(previous_date or date)
                    old_values = pd.to_numeric(
                        prices.loc[:cutoff, previous_contract], errors="coerce"
                    )
                    new_values = pd.to_numeric(
                        prices.loc[:cutoff, contract], errors="coerce"
                    )
                    common = old_values.dropna().index.intersection(
                        new_values.dropna().index
                    )
                    if not len(common):
                        raise RolloverAdjustmentError(
                            f"no common close at or before {cutoff.date()} for "
                            f"{previous_contract}->{contract}"
                        )
                    overlap_date = common.max()
                    old_value = float(old_values.loc[overlap_date])
                    new_value = float(new_values.loc[overlap_date])
                    if not np.isfinite(old_value) or not np.isfinite(new_value) or new_value <= 0:
                        raise RolloverAdjustmentError(
                            f"invalid common close for {previous_contract}->{contract}"
                        )
                    current_scale = previous_scale * old_value / new_value
                scale.at[date] = current_scale
                previous_contract = contract
                previous_scale = current_scale
                previous_date = date
            scales[root] = scale
        return scales

    def _aggregate_to_root(self, df: pd.DataFrame) -> pd.DataFrame:
        """将合约级数据聚合为品种根代码级 (RB2401.SHF → RB).

        CR-015: 此方法为回退逻辑 (无 volume 数据时使用),
        保留旧的全查询期选最优合约行为.
        """
        if df.empty:
            return df
        root_map = {}
        for col in df.columns:
            root = str(col).split(".")[0].rstrip("0123456789")
            if root:
                root_map.setdefault(root, []).append(col)

        result = {}
        for root, cols in root_map.items():
            if len(cols) == 1:
                result[root] = df[cols[0]]
            else:
                # 取非NaN最多的合约
                best = max(cols, key=lambda c: df[c].notna().sum())
                result[root] = df[best]
        return pd.DataFrame(result)

    # ------------------------------------------------------------------
    # 分钟K线 + 日内特征
    # ------------------------------------------------------------------

    def fetch_minute_bars(
        self,
        tickers: List[str],
        start: Date,
        end: Date,
        frequency: str = "1min",
        **kwargs,
    ) -> pd.DataFrame:
        """获取分钟级 K 线数据.

        Args:
            tickers: 品种根代码列表 (如 ['RB', 'CU'])
            start/end: 日期范围
            frequency: source frequency; currently the table stores one-minute bars

        Returns:
            DataFrame: MultiIndex (datetime, ticker) × columns [open, high, low, close, volume, amount]
        """
        if "freq" in kwargs:
            frequency = kwargs.pop("freq")
        if kwargs:
            raise TypeError(f"unexpected fetch_minute_bars options: {sorted(kwargs)}")
        requested_start = pd.Timestamp(start).normalize()
        requested_end = pd.Timestamp(end).normalize()
        query_start = requested_start - pd.Timedelta(days=14)
        start_str = query_start.strftime("%Y.%m.%d")
        end_str = pd.Timestamp(end).strftime("%Y.%m.%d")

        ticker_filter = " or ".join(
            f"InstrumentID like '{t}%'" for t in tickers
        )

        script = f"""
        t = loadTable("{self._minute_db}", "{self._minute_table}")
        select TradeDate, InstrumentID, Time, OpenPrice, HighPrice,
               LowPrice, ClosePrice, Volume, Turnover
        from t
        where ({ticker_filter})
        and TradeDate >= {start_str} and TradeDate <= {end_str}
        order by InstrumentID, TradeDate, Time
        """
        df = self._query(script)
        if df.empty:
            logger.warning(f"DDB 分钟数据为空: {tickers[:3]}... [{start}~{end}]")
            return pd.DataFrame()

        # 合并日期+时间为 datetime
        if "Time" in df.columns:
            df["datetime"] = pd.to_datetime(
                df["TradeDate"].astype(str) + " " + df["Time"].astype(str)
            )
        else:
            df["datetime"] = pd.to_datetime(df["TradeDate"].astype(str))

        df["TradeDate"] = pd.to_datetime(df["TradeDate"].astype(str)).dt.normalize()
        # Preserve the concrete contract until T-1 dominant selection is done.
        df["root"] = df["InstrumentID"].astype(str).str.extract(
            r"^([A-Za-z]+)", expand=False
        ).str.upper()

        # 重命名列
        rename = {
            "OpenPrice": "open", "HighPrice": "high", "LowPrice": "low",
            "ClosePrice": "close", "Volume": "volume", "Turnover": "amount",
        }
        df = df.rename(columns=rename)

        daily_volume = (
            df.groupby(["root", "TradeDate", "InstrumentID"], sort=False)["volume"]
            .sum(min_count=1)
            .unstack("InstrumentID")
        )
        schedule_rows = []
        dominant_map = {}
        for root, root_volume in daily_volume.groupby(level="root", sort=False):
            root_volume = root_volume.droplevel("root")
            raw = root_volume.fillna(-1.0).idxmax(axis=1)
            raw = raw.where(~root_volume.isna().all(axis=1))
            effective = raw.shift(self._dominant_lag_days)
            dominant_map[root] = effective
            schedule_rows.append(pd.DataFrame({
                "root": root,
                "TradeDate": effective.index,
                "selected_contract": effective.to_numpy(),
            }))
        if not schedule_rows:
            return pd.DataFrame()
        daily_close = (
            df.sort_values(["TradeDate", "Time"])
            .groupby(["TradeDate", "InstrumentID"], sort=True)["close"]
            .last()
            .unstack("InstrumentID")
        )
        roll_scales = self._build_roll_scales(daily_close, dominant_map)
        schedule = pd.concat(schedule_rows, ignore_index=True)
        df = df.merge(schedule, on=["root", "TradeDate"], how="left", validate="many_to_one")
        df = df[df["InstrumentID"] == df["selected_contract"]]
        scale_rows = []
        for root, scale in roll_scales.items():
            scale_rows.append(pd.DataFrame({
                "root": root,
                "TradeDate": scale.index,
                "roll_scale": scale.to_numpy(),
            }))
        if scale_rows:
            scales = pd.concat(scale_rows, ignore_index=True)
            df = df.merge(
                scales, on=["root", "TradeDate"], how="left", validate="many_to_one"
            )
            if df["roll_scale"].isna().any():
                raise ValueError("selected minute contract has no roll scale")
            for field in ("open", "high", "low", "close"):
                df[field] = pd.to_numeric(df[field], errors="coerce") * df["roll_scale"]
        df = df[
            (df["TradeDate"] >= requested_start) & (df["TradeDate"] <= requested_end)
        ]
        if df.empty:
            return pd.DataFrame()

        df = df.set_index(["datetime", "root"])
        result = df[["open", "high", "low", "close", "volume", "amount"]].sort_index()
        if result.index.duplicated().any():
            raise ValueError("selected minute contract still has duplicate datetime/root rows")
        return result

    def fetch_price_at_frequency(
        self,
        tickers: List[str],
        start: Date,
        end: Date,
        fields: List[str],
        frequency: str = "15min",
    ) -> PricePanel:
        """获取指定频率的 OHLCV 面板 (分钟级).

        从 1min K线数据重采样为目标频率 (15min/30min/60min),
        返回与日度 fetch_price 相同格式的 PricePanel.

        Args:
            tickers: 品种根代码列表 (如 ['RB', 'CU'])
            start/end: 日期范围
            fields: 字段列表 (如 ['open', 'high', 'low', 'close', 'volume'])
            frequency: 目标频率 ('15min' / '30min' / '60min' / 'hourly')

        Returns:
            {field: DataFrame(index=分钟时间戳, columns=tickers)}
            DDB 不可用时返回空字典
        """
        # 获取 1min 原始数据
        minute_df = self.fetch_minute_bars(tickers, start, end, frequency="1min")
        if minute_df.empty:
            return {}

        # 重置索引: (datetime, root) → 列
        minute_df = minute_df.reset_index()

        # 按频率重采样 (对每个 root 分组)
        # 重采样规则: open=first, high=max, low=min, close=last, volume=sum, amount=sum
        agg_rules = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "amount": "sum",
        }

        frequency_alias = {"hourly": "60min", "60m": "60min"}.get(
            frequency, frequency
        )
        all_resampled = (
            minute_df.sort_values(["root", "datetime"])
            .groupby(
                ["root", pd.Grouper(key="datetime", freq=frequency_alias)],
                sort=True,
                observed=True,
            )
            .agg(agg_rules)
        )
        if all_resampled.empty:
            return {}

        # 转换为 PricePanel: {field: DataFrame(index=datetime, columns=tickers)}
        result: PricePanel = {}
        for field in fields:
            if field in all_resampled.columns:
                pivot = all_resampled[field].unstack("root")
                # 过滤掉全 NaN 的行
                pivot = pivot.dropna(how="all")
                result[field] = pivot

        return result

    def _fetch_intraday_features(
        self,
        tickers: TickerIndex,
        start: Date,
        end: Date,
        fields: List[str],
    ) -> PricePanel:
        """从分钟K线聚合为日度日内特征.

        聚合规则:
        - vwap = sum(close * volume) / sum(volume)  (按日聚合)
        - intraday_return = (last_close - first_open) / first_open
        - overnight_gap = (first_open - prev_day_close) / prev_day_close
        - intraday_volatility = (max_high - min_low) / first_open
        - close_to_vwap = (last_close - vwap) / vwap
        - volume_concentration = max_minute_volume / avg_minute_volume
        - amihud_illiquidity = mean(|minute_ret|) / (amount / volume)
        - tail_momentum = (last_close - 14:30_close) / last_close
        """
        cache_key = (
            self._data_version,
            f"tminus{self._dominant_lag_days}_dominant",
            str(pd.Timestamp(start)),
            str(pd.Timestamp(end)),
            tuple(sorted(map(str, tickers))),
            tuple(sorted(map(str, fields))),
        )
        if cache_key in self._intraday_cache:
            cached = self._intraday_cache[cache_key]
            return {f: cached[f] for f in fields if f in cached}

        # 获取分钟数据
        minute_df = self.fetch_minute_bars(
            list(tickers), start, end, frequency="1min"
        )
        if minute_df.empty:
            return {f: pd.DataFrame() for f in fields}
        features = self._compute_intraday_features_from_bars(minute_df, fields)

        # 缓存
        self._intraday_cache[cache_key] = features

        return {f: features.get(f, pd.DataFrame()) for f in fields}

    @staticmethod
    def _compute_intraday_features_from_bars(
        minute_bars: pd.DataFrame, fields: List[str]
    ) -> Dict[str, pd.DataFrame]:
        """Vectorised daily feature aggregation from one selected contract."""
        bars = minute_bars.reset_index().sort_values(["datetime", "root"])
        bars["date"] = bars["datetime"].dt.normalize()
        bars["weighted_close"] = bars["close"] * bars["volume"].clip(lower=0)
        group_keys = ["date", "root"]
        grouped = bars.groupby(group_keys, sort=True, observed=True)
        daily = grouped.agg(
            first_open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            last_close=("close", "last"),
            volume_sum=("volume", "sum"),
            volume_max=("volume", "max"),
            volume_mean=("volume", "mean"),
            amount_sum=("amount", "sum"),
            weighted_close_sum=("weighted_close", "sum"),
            bar_count=("close", "size"),
            last_time=("datetime", "max"),
        )

        safe_open = daily["first_open"].where(daily["first_open"] > 0)
        safe_volume = daily["volume_sum"].where(daily["volume_sum"] > 0)
        vwap = daily["weighted_close_sum"] / safe_volume
        series: Dict[str, pd.Series] = {
            "vwap": vwap,
            "intraday_return": (daily["last_close"] - daily["first_open"]) / safe_open,
            "intraday_volatility": (daily["high"] - daily["low"]) / safe_open,
            "close_to_vwap": (daily["last_close"] - vwap) / vwap.where(vwap > 0),
            "volume_concentration": daily["volume_max"]
            / daily["volume_mean"].where(daily["volume_mean"] > 0),
        }

        daily_wide_close = daily["last_close"].unstack("root")
        daily_wide_open = daily["first_open"].unstack("root")
        previous_close = daily_wide_close.shift(1)
        overnight = (daily_wide_open - previous_close) / previous_close.where(previous_close > 0)

        bars["abs_minute_return"] = bars.groupby(
            group_keys, sort=False, observed=True
        )["close"].pct_change(fill_method=None).abs()
        absolute_return_sum = bars.groupby(
            group_keys, sort=True, observed=True
        )["abs_minute_return"].sum(min_count=1)
        series["amihud_illiquidity"] = absolute_return_sum / daily["amount_sum"].where(
            daily["amount_sum"] > 0
        )

        targets = daily[["last_time", "last_close"]].reset_index()
        targets["target_time"] = targets["last_time"] - pd.Timedelta(minutes=30)
        references = pd.merge_asof(
            targets.sort_values("target_time"),
            bars[["date", "root", "datetime", "close"]].sort_values("datetime"),
            left_on="target_time",
            right_on="datetime",
            by=["date", "root"],
            direction="backward",
        ).set_index(group_keys)
        series["tail_momentum"] = (
            references["last_close"] - references["close"]
        ) / references["last_close"].where(references["last_close"] > 0)

        result: Dict[str, pd.DataFrame] = {}
        for field in fields:
            if field == "overnight_gap":
                frame = overnight
            elif field in series:
                frame = series[field].unstack("root")
            else:
                continue
            frame.index = pd.DatetimeIndex(frame.index)
            frame.index.name = "date"
            result[field] = frame.sort_index()
        return result

    # ------------------------------------------------------------------
    # 其他 DataSource 方法
    # ------------------------------------------------------------------

    def fetch_fundamental(self, tickers, start, end, fields) -> dict:
        return {}

    def fetch_industry(self, tickers, date) -> IndustryMapping:
        return pd.Series(dtype=object)

    def fetch_index_constituents(self, index_code, date) -> Universe:
        return pd.Index([])

    def fetch_calendar(self, start, end) -> DateIndex:
        """从 DDB 获取交易日历 (查日度行情表的 distinct date)."""
        start_str = pd.Timestamp(start).strftime("%Y.%m.%d")
        end_str = pd.Timestamp(end).strftime("%Y.%m.%d")
        script = f"""
        t = loadTable("{self._eod_db}", "{self._eod_table}")
        select distinct TRADE_DT from t
        where TRADE_DT >= {start_str} and TRADE_DT <= {end_str}
        order by TRADE_DT
        """
        df = self._query(script)
        if df.empty:
            return pd.DatetimeIndex([])
        days = pd.to_datetime(df["TRADE_DT"].astype(str))
        return pd.DatetimeIndex(days)
