from __future__ import annotations

import logging
import os
import re
import sqlite3
import time
import urllib.parse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

from core.interfaces import DataSource
from core.registry import register
from core.types import *


logger = logging.getLogger(__name__)


@register("data_source", "mysql_futures")
class MySQLSource(DataSource):
    """连接阿里云 RDS MySQL, 读取 Wind 格式的期货数据.

    配置要求:
    - mysql.host / port / user / password / database
    - mysql.tables: 表名和字段映射

    Wind 格式字段映射见 config/default.yaml 的 mysql.tables.

    ⚠️ 价格表最新至当日，持仓表滞后约 3 个月 (最新 2026-04-21).
    """
    market = "futures"

    # 股指期货品种代码 (行情在 cindexfutureseodprices)
    INDEX_FUTURES_ROOTS = {"IF", "IH", "IC", "IM"}
    # 国债期货品种代码 (行情在 cbondfutureseodprices)
    BOND_FUTURES_ROOTS = {"T", "TF", "TS", "TL"}
    _INTRADAY_CURVE_FIELDS = {
        "curve_total_oi",
        "curve_top2_oi",
        "curve_total_volume",
        "curve_contract_count",
        "curve_oi_breadth",
        "curve_oi_concentration",
        "curve_oi_hhi",
    }
    _INTRADAY_PRICE_FIELDS = {"open", "high", "low", "close"}

    def __init__(self, mysql_config: Optional[dict] = None, **kwargs) -> None:
        self._config = mysql_config or {}
        self._engine = None
        self._active_endpoint_index: Optional[int] = None
        self._engines: Dict[int, object] = {}
        self._circuit_open_until = 0.0
        self._failure_cooldown = max(
            float(self._config.get("failure_cooldown", 30.0)), 0.0
        )
        self._dominant_lag_days = max(
            int(self._config.get("dominant_lag_days", 1)), 1
        )
        self._schedule_buffer_days = max(
            int(self._config.get("schedule_buffer_days", 20)), 7
        )
        self._endpoints = self._normalise_endpoints(self._config)
        self._table_map: Dict = (
            self._config.get("tables", {}) if self._config else {}
        )
        self._contract_pair_cache: Dict[tuple, Dict[str, pd.DataFrame]] = {}
        self._local_ths_5minute_db = Path(
            os.environ.get(
                "MF_THS_5MINUTE_DB",
                str(
                    self._config.get(
                        "local_ths_5minute_db",
                        r"E:\程明杰公司内容\期货行情数据\本地表\ths_data\ths_data_5minute.db",
                    )
                ),
            )
        )

    @staticmethod
    def _normalise_endpoints(config: dict) -> List[dict]:
        common = {
            "port": config.get("port", 3306),
            "user": config.get("user", ""),
            "password": config.get("password", ""),
            "database": config.get("database", ""),
            "charset": config.get("charset", "utf8mb4"),
        }
        configured = list(config.get("endpoints") or [])
        fallbacks = list(config.get("fallbacks") or [])
        if configured:
            raw_endpoints = configured + fallbacks
        else:
            raw_endpoints = []
            if config.get("host"):
                raw_endpoints.append({
                    "name": config.get("name", "primary"),
                    "host": config.get("host", ""),
                    **common,
                })
            raw_endpoints.extend(fallbacks)
        endpoints = []
        seen = set()
        for index, endpoint in enumerate(raw_endpoints):
            merged = {**common, **dict(endpoint)}
            if not merged.get("host"):
                continue
            key = (
                merged["host"], int(merged.get("port", 3306)),
                merged.get("database", ""), merged.get("user", ""),
            )
            if key in seen:
                continue
            seen.add(key)
            merged["name"] = merged.get("name") or f"endpoint_{index + 1}"
            endpoints.append(merged)
        return endpoints

    def _create_endpoint_engine(self, endpoint_index: int):
        endpoint = self._endpoints[endpoint_index]
        password = urllib.parse.quote_plus(str(endpoint.get("password", "")))
        query = urllib.parse.urlencode({
            "charset": endpoint.get("charset", "utf8mb4"),
            "connect_timeout": int(self._config.get("connect_timeout", 5)),
        })
        url = (
            f"mysql+pymysql://{endpoint.get('user', '')}:{password}@"
            f"{endpoint['host']}:{int(endpoint.get('port', 3306))}/"
            f"{endpoint.get('database', '')}?{query}"
        )
        return create_engine(
            url,
            pool_size=int(self._config.get("pool_size", 5)),
            pool_pre_ping=True,
        )

    def _engine_for(self, endpoint_index: int):
        if endpoint_index not in self._engines:
            self._engines[endpoint_index] = self._create_endpoint_engine(endpoint_index)
        return self._engines[endpoint_index]

    def _ordered_endpoint_indices(self) -> List[int]:
        if self._active_endpoint_index is None:
            return list(range(len(self._endpoints)))
        active = self._active_endpoint_index
        return [active] + [i for i in range(len(self._endpoints)) if i != active]

    def _activate_available_endpoint(self):
        self._raise_if_circuit_open()
        if not self._endpoints:
            raise ValueError("no MySQL endpoint is configured")
        failures = []
        for index in self._ordered_endpoint_indices():
            endpoint = self._endpoints[index]
            engine = self._engine_for(index)
            try:
                with engine.connect() as connection:
                    connection.execute(text("SELECT 1"))
                self._active_endpoint_index = index
                self._engine = engine
                self._circuit_open_until = 0.0
                logger.info(
                    "MySQL endpoint active: %s (%s:%s/%s)",
                    endpoint["name"], endpoint["host"], endpoint.get("port", 3306),
                    endpoint.get("database", ""),
                )
                return engine
            except Exception as exc:
                failures.append(f"{endpoint['name']}:{type(exc).__name__}")
                try:
                    engine.dispose()
                except Exception:
                    pass
                self._engines.pop(index, None)
        self._open_circuit()
        raise ConnectionError(
            "all configured MySQL endpoints failed (" + "; ".join(failures) + ")"
        )

    @property
    def engine(self):
        if self._engine is None:
            return self._activate_available_endpoint()
        return self._engine

    @property
    def active_endpoint_name(self) -> Optional[str]:
        if self._active_endpoint_index is None:
            return None
        return str(self._endpoints[self._active_endpoint_index]["name"])

    def _read_sql(self, sql: str) -> pd.DataFrame:
        """Execute read-only SQL with ordered endpoint failover and one retry each."""
        if self._should_read_local_ths_5minute(sql):
            try:
                return self._read_local_ths_5minute(sql)
            except Exception:
                logger.warning(
                    "Local ths_data_5minute mirror query failed; falling back to RDS",
                    exc_info=True,
                )
        self._raise_if_circuit_open()
        if not self._endpoints:
            raise ValueError("no MySQL endpoint is configured")
        failures = []
        for index in self._ordered_endpoint_indices():
            endpoint = self._endpoints[index]
            try:
                engine = self._engine_for(index)
                with engine.connect() as connection:
                    frame = pd.read_sql(text(sql), connection)
                self._active_endpoint_index = index
                self._engine = engine
                self._circuit_open_until = 0.0
                return frame
            except Exception as exc:
                failures.append(f"{endpoint['name']}:{type(exc).__name__}")
                logger.warning(
                    "MySQL query failed on endpoint %s; trying next endpoint",
                    endpoint["name"],
                )
                engine = self._engines.pop(index, None)
                if engine is not None:
                    try:
                        engine.dispose()
                    except Exception:
                        pass
                if self._active_endpoint_index == index:
                    self._active_endpoint_index = None
                    self._engine = None
        self._open_circuit()
        raise ConnectionError("MySQL query failed on all endpoints (" + "; ".join(failures) + ")")

    def _should_read_local_ths_5minute(self, sql: str) -> bool:
        if not self._local_ths_5minute_db.is_file():
            return False
        table_config = self._get_table_config("intraday_5m") or {}
        table_name = str(table_config.get("table_name", "ths_data_5minute"))
        lowered = sql.lower()
        return "select" in lowered and table_name.lower() in lowered

    @staticmethod
    def _sqlite_regexp(pattern: str, value: object) -> int:
        if value is None:
            return 0
        return 1 if re.search(pattern, str(value)) else 0

    def _read_local_ths_5minute(self, sql: str) -> pd.DataFrame:
        read_only_uri = (
            f"file:{self._local_ths_5minute_db.as_posix()}?mode=ro"
        )
        with sqlite3.connect(read_only_uri, uri=True) as connection:
            connection.create_function("REGEXP", 2, self._sqlite_regexp)
            return pd.read_sql_query(
                self._optimise_local_ths_5minute_sql(sql),
                connection,
            )

    def _optimise_local_ths_5minute_sql(self, sql: str) -> str:
        table_config = self._get_table_config("intraday_5m") or {}
        ticker_column = str(
            (table_config.get("columns") or {}).get("ticker", "code")
        )
        quoted_ticker = re.escape(self._quote_identifier(ticker_column))
        regexp_clause = re.compile(
            rf"{quoted_ticker}\s+REGEXP\s+'\^\((?P<roots>[A-Z|]+)\)\[0-9\]\+\$'",
            re.IGNORECASE,
        )

        def _replace(match: re.Match) -> str:
            roots = [root for root in match.group("roots").split("|") if root]
            if not roots:
                return match.group(0)
            clauses = [
                f"{self._quote_identifier(ticker_column)} GLOB '{root}[0-9]*'"
                for root in roots
            ]
            return "(" + " OR ".join(clauses) + ")"

        return regexp_clause.sub(_replace, sql)

    def _open_circuit(self) -> None:
        self._circuit_open_until = time.monotonic() + self._failure_cooldown

    def _raise_if_circuit_open(self) -> None:
        remaining = self._circuit_open_until - time.monotonic()
        if remaining > 0:
            raise ConnectionError(
                f"MySQL endpoint circuit open; retry after {remaining:.1f}s"
            )

    def _get_table_config(self, table_key: str) -> Optional[dict]:
        return self._table_map.get(table_key)

    @staticmethod
    def _quote_identifier(identifier: str) -> str:
        """Quote a configured MySQL identifier, including schema prefixes."""
        if not identifier or "\x00" in identifier:
            raise ValueError("invalid MySQL identifier")
        return ".".join(
            f"`{part.replace('`', '``')}`" for part in identifier.split(".")
        )

    @staticmethod
    def _normalise_intraday_frequency(frequency: str) -> tuple[str, Optional[str]]:
        aliases = {
            "5m": "5min",
            "15m": "15min",
            "30m": "30min",
            "60m": "hourly",
            "60min": "hourly",
            "1h": "hourly",
        }
        value = aliases.get(str(frequency).lower(), str(frequency).lower())
        routes = {
            "5min": ("5min", None),
            "15min": ("5min", "15min"),
            "30min": ("5min", "30min"),
            "hourly": ("5min", "60min"),
        }
        if value not in routes:
            raise NotImplementedError(
                f"MySQLSource does not support frequency={frequency!r}; "
                "available intraday frequencies are 5min, 15min, 30min, hourly"
            )
        return routes[value]

    def _assign_trading_dates(self, timestamps: pd.Series) -> pd.Series:
        """Map night bars to the next exchange trading day."""
        values = pd.to_datetime(timestamps, errors="coerce")
        if values.empty:
            return pd.Series(dtype="datetime64[ns]", index=timestamps.index)
        natural_dates = values.dt.normalize()
        targets = natural_dates.where(
            values.dt.hour.lt(18), natural_dates + pd.Timedelta(days=1)
        )
        calendar = self.fetch_calendar(
            targets.min() - pd.Timedelta(days=7),
            targets.max() + pd.Timedelta(days=7),
        )
        calendar = pd.DatetimeIndex(calendar).normalize().unique().sort_values()
        if len(calendar) == 0:
            return targets.map(lambda value: value + pd.offsets.BDay(0))
        locations = calendar.searchsorted(targets.to_numpy(), side="left")
        assigned = np.full(len(targets), np.datetime64("NaT"), dtype="datetime64[ns]")
        valid = locations < len(calendar)
        assigned[valid] = calendar.to_numpy()[locations[valid]]
        return pd.Series(assigned, index=timestamps.index)

    def _fetch_intraday_contract_rows(
        self,
        tickers,
        start,
        end,
        fields: List[str],
    ) -> pd.DataFrame:
        cfg = self._get_table_config("intraday_5m")
        if not cfg:
            raise ValueError("mysql.tables.intraday_5m is not configured")
        columns = dict(cfg.get("columns") or {})
        required = {"datetime", "ticker", "close", "volume", "oi"}
        missing = sorted(required - set(columns))
        if missing:
            raise ValueError(
                "intraday_5m configuration is missing columns: "
                + ", ".join(missing)
            )

        roots = tuple(dict.fromkeys(str(value).strip().upper() for value in tickers))
        if not roots:
            return pd.DataFrame()
        invalid = [root for root in roots if not re.fullmatch(r"[A-Z]+", root)]
        if invalid:
            raise ValueError("invalid futures roots: " + ", ".join(invalid))

        logical_fields = tuple(dict.fromkeys(
            ["close", "volume", "oi"]
            + [field for field in fields if field in columns]
        ))
        selections = [
            f"{self._quote_identifier(columns['datetime'])} AS `trade_datetime`",
            f"{self._quote_identifier(columns['ticker'])} AS `symbol`",
        ]
        selections.extend(
            f"{self._quote_identifier(columns[field])} AS "
            f"{self._quote_identifier(field)}"
            for field in logical_fields
        )
        buffer_start = (
            pd.Timestamp(start).normalize()
            - pd.Timedelta(days=self._schedule_buffer_days)
        )
        exclusive_end = pd.Timestamp(end).normalize() + pd.Timedelta(days=1)
        root_pattern = "^(" + "|".join(roots) + ")[0-9]+$"
        datetime_sql = self._quote_identifier(columns["datetime"])
        ticker_sql = self._quote_identifier(columns["ticker"])
        sql = (
            f"SELECT {', '.join(selections)} "
            f"FROM {self._quote_identifier(str(cfg['table_name']))} "
            f"WHERE {datetime_sql} >= '{buffer_start:%Y-%m-%d %H:%M:%S}' "
            f"AND {datetime_sql} < '{exclusive_end:%Y-%m-%d %H:%M:%S}' "
            f"AND {ticker_sql} REGEXP '{root_pattern}' "
            f"ORDER BY {datetime_sql}, {ticker_sql}"
        )
        frame = self._read_sql(sql)
        if frame.empty:
            return frame
        frame["trade_datetime"] = pd.to_datetime(
            frame["trade_datetime"], errors="coerce"
        )
        frame["symbol"] = frame["symbol"].astype(str).str.strip().str.upper()
        frame["root"] = frame["symbol"].str.extract(
            r"^([A-Z]+)(?=[0-9]+$)", expand=False
        )
        frame = frame.loc[frame["root"].isin(roots)].copy()
        for field in logical_fields:
            frame[field] = pd.to_numeric(frame[field], errors="coerce")
        frame["trade_date"] = self._assign_trading_dates(
            frame["trade_datetime"]
        )
        if str(cfg.get("timestamp_convention", "start")).lower() == "end":
            bar_minutes = max(int(cfg.get("bar_minutes", 5)), 1)
            frame["trade_datetime"] = (
                frame["trade_datetime"] - pd.Timedelta(minutes=bar_minutes)
            )
        frame = frame.dropna(subset=["trade_datetime", "trade_date", "root"])
        return (
            frame.sort_values(["trade_datetime", "root", "symbol"])
            .drop_duplicates(["trade_datetime", "root", "symbol"], keep="last")
            .reset_index(drop=True)
        )

    def _build_intraday_continuous_plan(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame()
        daily_last = (
            frame.sort_values("trade_datetime")
            .groupby(["trade_date", "root", "symbol"], sort=True, as_index=False)
            .tail(1)
        )
        dominant = (
            daily_last.sort_values(
                ["trade_date", "root", "oi", "volume", "symbol"],
                ascending=[True, True, False, False, True],
                na_position="last",
            )
            .drop_duplicates(["trade_date", "root"], keep="first")
            .sort_values(["root", "trade_date"])
        )
        dominant["contract"] = dominant.groupby("root")["symbol"].shift(
            self._dominant_lag_days
        )
        close_lookup = daily_last.set_index(
            ["trade_date", "root", "symbol"]
        )["close"]
        plans = []
        for root, group in dominant.groupby("root", sort=False):
            adjustment = 1.0
            previous_contract = None
            previous_date = None
            for row in group.itertuples(index=False):
                contract = row.contract
                if pd.isna(contract):
                    previous_date = row.trade_date
                    continue
                contract = str(contract)
                if (
                    previous_contract is not None
                    and contract != previous_contract
                    and previous_date is not None
                ):
                    old_close = close_lookup.get(
                        (previous_date, root, previous_contract), np.nan
                    )
                    new_close = close_lookup.get(
                        (previous_date, root, contract), np.nan
                    )
                    if (
                        np.isfinite(old_close)
                        and np.isfinite(new_close)
                        and float(new_close) > 0.0
                    ):
                        adjustment *= float(old_close) / float(new_close)
                plans.append({
                    "trade_date": row.trade_date,
                    "root": root,
                    "contract": contract,
                    "adjustment": adjustment,
                })
                previous_contract = contract
                previous_date = row.trade_date
        return pd.DataFrame(plans)

    @classmethod
    def _intraday_curve_rows(cls, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame()
        work = frame.copy()
        work["oi"] = work["oi"].clip(lower=0.0)
        work["volume"] = work["volume"].clip(lower=0.0)
        work = work.sort_values(["root", "symbol", "trade_datetime"])
        previous = work.groupby(["root", "symbol"])["oi"].shift(1)
        work["oi_increased"] = work["oi"].gt(previous).where(previous.notna())
        work["oi_sq"] = work["oi"].pow(2)
        keys = ["trade_datetime", "root"]
        aggregate = work.groupby(keys, sort=True).agg(
            curve_total_oi=("oi", "sum"),
            curve_total_volume=("volume", "sum"),
            curve_contract_count=("oi", lambda values: int(values.gt(0).sum())),
            curve_oi_breadth=("oi_increased", "mean"),
            oi_sq_sum=("oi_sq", "sum"),
        )
        top2 = (
            work.sort_values(keys + ["oi"], ascending=[True, True, False])
            .groupby(keys, sort=True)
            .head(2)
            .groupby(keys, sort=True)["oi"]
            .sum()
            .rename("curve_top2_oi")
        )
        aggregate = aggregate.join(top2, how="left")
        denominator = aggregate["curve_total_oi"].replace(0.0, np.nan)
        aggregate["curve_oi_concentration"] = (
            aggregate["curve_top2_oi"] / denominator
        )
        aggregate["curve_oi_hhi"] = aggregate.pop("oi_sq_sum") / denominator.pow(2)
        return aggregate.reset_index()

    @staticmethod
    def _resample_intraday_rows(frame: pd.DataFrame, rule: str) -> pd.DataFrame:
        if frame.empty:
            return frame
        work = frame.copy()
        work["trade_datetime"] = pd.to_datetime(
            work["trade_datetime"]
        ).dt.floor(rule)
        aggregations = {
            field: method
            for field, method in (
                ("open", "first"),
                ("high", "max"),
                ("low", "min"),
                ("close", "last"),
                ("volume", "sum"),
                ("amount", "sum"),
                ("oi", "last"),
                ("curve_total_oi", "last"),
                ("curve_top2_oi", "last"),
                ("curve_total_volume", "sum"),
                ("curve_contract_count", "last"),
                ("curve_oi_breadth", "mean"),
                ("curve_oi_concentration", "last"),
                ("curve_oi_hhi", "last"),
            )
            if field in work.columns
        }
        return (
            work.groupby(["trade_datetime", "root"], sort=True, as_index=False)
            .agg(aggregations)
        )

    def fetch_price_at_frequency(
        self,
        tickers: TickerIndex,
        start: Date,
        end: Date,
        fields: List[str],
        frequency: str = "daily",
    ) -> PricePanel:
        if str(frequency).lower() == "daily":
            return self.fetch_price(tickers, start, end, fields)
        _, resample_rule = self._normalise_intraday_frequency(frequency)
        requested = tuple(dict.fromkeys(str(field) for field in fields))
        raw_fields = [
            field for field in requested
            if field in {"open", "high", "low", "close", "volume", "amount", "oi"}
        ]
        needs_curve = any(field in self._INTRADAY_CURVE_FIELDS for field in requested)
        load_fields = list(dict.fromkeys(raw_fields + ["close", "volume", "oi"]))
        raw = self._fetch_intraday_contract_rows(
            tickers, start, end, load_fields
        )
        if raw.empty:
            return {}

        plan = self._build_intraday_continuous_plan(raw)
        selected = raw.merge(
            plan, on=["trade_date", "root"], how="inner", validate="many_to_one"
        )
        selected = selected.loc[selected["symbol"].eq(selected["contract"])].copy()
        for field in self._INTRADAY_PRICE_FIELDS.intersection(raw_fields):
            selected[field] = selected[field] * selected["adjustment"]
        selected = selected[
            ["trade_datetime", "root"]
            + [field for field in raw_fields if field in selected]
        ]

        combined = selected
        if needs_curve:
            curve = self._intraday_curve_rows(raw)
            combined = curve if combined.empty else combined.merge(
                curve, on=["trade_datetime", "root"], how="outer"
            )
        start_date = pd.Timestamp(start).normalize()
        end_date = pd.Timestamp(end).normalize()
        valid_dates = raw[
            ["trade_datetime", "trade_date"]
        ].drop_duplicates("trade_datetime")
        valid_timestamps = valid_dates.loc[
            valid_dates["trade_date"].between(start_date, end_date),
            "trade_datetime",
        ]
        combined = combined.loc[
            combined["trade_datetime"].isin(valid_timestamps)
        ]
        if resample_rule:
            combined = self._resample_intraday_rows(combined, resample_rule)

        result: PricePanel = {}
        for field in requested:
            if field == "oi_change" and "oi" in combined:
                panel = combined.pivot(
                    index="trade_datetime", columns="root", values="oi"
                ).sort_index().diff()
            elif field in combined:
                panel = combined.pivot(
                    index="trade_datetime", columns="root", values=field
                ).sort_index()
            else:
                continue
            panel.index = pd.DatetimeIndex(panel.index)
            result[field] = panel.reindex(columns=list(tickers))
        return result

    def fetch_macro(
        self,
        fields: List[str],
        start: Optional[Date] = None,
        end: Optional[Date] = None,
    ) -> pd.DataFrame:
        """Read configured macro fields without proxying unavailable columns."""
        requested = list(dict.fromkeys(str(field) for field in fields))
        if not requested:
            return pd.DataFrame()

        cfg = self._get_table_config("macrodata")
        if not cfg:
            return pd.DataFrame(columns=requested, dtype=float)
        column_map = cfg.get("columns", {})
        missing = [field for field in requested if field not in column_map]
        if missing:
            raise ValueError(
                "unconfigured macro fields: " + ", ".join(sorted(missing))
            )
        if "date" not in column_map:
            raise ValueError("macrodata configuration requires a date column")

        table_sql = self._quote_identifier(str(cfg["table_name"]))
        date_sql = self._quote_identifier(str(column_map["date"]))
        parsed_date = f"STR_TO_DATE(CAST({date_sql} AS CHAR), '%Y/%c/%e')"
        selections = [f"{parsed_date} AS `observation_date`"]
        selections.extend(
            f"{self._quote_identifier(str(column_map[field]))} AS "
            f"{self._quote_identifier(field)}"
            for field in requested
        )

        where_clauses = [f"{parsed_date} IS NOT NULL"]
        if start is not None:
            start_text = pd.Timestamp(start).strftime("%Y-%m-%d")
            where_clauses.append(f"{parsed_date} >= '{start_text}'")
        if end is not None:
            end_text = pd.Timestamp(end).strftime("%Y-%m-%d")
            where_clauses.append(f"{parsed_date} <= '{end_text}'")
        sql = (
            f"SELECT {', '.join(selections)} FROM {table_sql} "
            f"WHERE {' AND '.join(where_clauses)} ORDER BY `observation_date`"
        )
        frame = self._read_sql(sql)
        if frame.empty:
            return pd.DataFrame(columns=requested, dtype=float)

        frame["observation_date"] = pd.to_datetime(
            frame["observation_date"], errors="coerce"
        )
        frame = frame.dropna(subset=["observation_date"])
        frame = frame.drop_duplicates("observation_date", keep="last")
        frame = frame.set_index("observation_date").sort_index()
        return frame.reindex(columns=requested).apply(pd.to_numeric, errors="coerce")

    def _build_query(
        self,
        table_cfg: dict,
        fields: list,
        tickers,
        start,
        end,
        limit: Optional[int] = None,
    ) -> str:
        """构建 SQL 查询语句."""
        tbl = table_cfg["table_name"]
        col_map = table_cfg.get("columns", {})
        # 只获取需要的列
        requested_cols: set = set(col_map.get(f, f) for f in fields)
        if "date" in col_map:
            requested_cols.add(col_map["date"])
        if "ticker" in col_map:
            requested_cols.add(col_map["ticker"])

        select_cols = ", ".join(requested_cols) if requested_cols else "*"

        where_clauses: list = []
        if tickers is not None and "ticker" in col_map:
            ticker_col = col_map["ticker"]
            # Wind 代码格式: 'RB2401.SHF' (具体合约) 或 'RB' (品种根代码)
            # 对短代码用 LIKE, 长代码用 IN
            like_parts = []
            in_parts = []
            for t in tickers:
                t_str = str(t).strip()
                # 去掉交易所后缀再判断
                short = t_str.split(".")[0]
                if len(short) <= 4 and not any(c.isdigit() for c in short):
                    like_parts.append(f"{ticker_col} LIKE '{short}%'")
                else:
                    in_parts.append(f"'{t_str}'")
            if in_parts:
                where_clauses.append(f"{ticker_col} IN ({', '.join(in_parts)})")
            if like_parts:
                where_clauses.append(f"({' OR '.join(like_parts)})")
        if start is not None and "date" in col_map:
            date_col = col_map["date"]
            start_str = pd.Timestamp(start).strftime("%Y%m%d")
            where_clauses.append(f"{date_col} >= '{start_str}'")
        if end is not None and "date" in col_map:
            date_col = col_map["date"]
            end_str = pd.Timestamp(end).strftime("%Y%m%d")
            where_clauses.append(f"{date_col} <= '{end_str}'")

        where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"
        sql = (
            f"SELECT {select_cols} FROM {tbl} WHERE {where_sql} "
            f"ORDER BY {col_map.get('ticker', 'ticker')}, "
            f"{col_map.get('date', 'date')}"
        )
        if limit:
            sql += f" LIMIT {limit}"
        return sql

    def _get_price_table_for_root(self, root: str) -> str:
        """根据品种根代码返回对应行情表名."""
        if root in self.INDEX_FUTURES_ROOTS:
            return "cindexfutureseodprices"
        elif root in self.BOND_FUTURES_ROOTS:
            return "cbondfutureseodprices"
        return "ccommodityfutureseodprices"

    def _get_table_config_for(self, table_name: str) -> dict:
        """获取行情表配置 (三张表字段相同, 复用 commodity_eod 的列映射)."""
        cfg = self._get_table_config("commodity_eod")
        if cfg:
            return {**cfg, "table_name": table_name}
        return {"table_name": table_name, "columns": {}}

    def fetch_price(
        self,
        tickers: TickerIndex,
        start: Date,
        end: Date,
        fields: List[str],
    ) -> PricePanel:
        """从期货行情表取数据, 使用主力合约映射构建连续合约.

        自动路由: 商品期货→ccommodityfutureseodprices,
                  股指期货→cindexfutureseodprices,
                  国债期货→cbondfutureseodprices.

        不支持的字段 (如 DDB 聚合字段 overnight_gap/vwap 等) 自动返回空
        DataFrame, 让日内因子优雅降级为 NaN.
        """
        # 过滤出 MySQL 表支持的字段, 不支持的字段返回空 DataFrame
        base_cfg = self._get_table_config("commodity_eod")
        col_map = (base_cfg or {}).get("columns", {})
        supported = [f for f in fields if f in col_map]
        unsupported = [f for f in fields if f not in col_map]

        result: PricePanel = {}
        # 不支持的字段 (DDB 聚合特征等) 返回空, 因子自动降级为 NaN
        for f in unsupported:
            result[f] = pd.DataFrame()

        if not supported:
            return result

        # 尝试获取主力合约映射
        mapping = self._fetch_main_contract_mapping(tickers, start, end)
        if mapping is not None and not mapping.empty:
            cont = self._fetch_continuous_price(tickers, start, end, supported, mapping)
            if cont:
                result.update(cont)
                return result

        raise RuntimeError(
            "main-contract mapping is unavailable or continuous construction "
            "failed; refusing the full-window single-contract legacy fallback"
        )

    def _fetch_main_contract_mapping(
        self,
        root_codes: TickerIndex,
        start: Date,
        end: Date,
    ) -> pd.DataFrame:
        """从 cfuturescontractmapping 查询主力合约映射.

        Returns:
            DataFrame: 列 [root, main_contract, start_date, end_date]
            root 为 2 字母品种代码 (如 RB), main_contract 为具体合约 (如 RB2401.SHF).
        """
        cfg = self._get_table_config("contract_mapping")
        if not cfg:
            return pd.DataFrame()

        col_map = cfg.get("columns", {})
        tbl = cfg["table_name"]
        ticker_col = col_map.get("ticker", "S_INFO_WINDCODE")
        mapped_col = col_map.get("mapped_to", "FS_MAPPING_WINDCODE")
        start_col = col_map.get("start_date", "STARTDATE")
        end_col = col_map.get("end_date", "ENDDATE")

        # 只匹配品种根代码+交易所后缀 (如 RB.SHF), 排除 RB01M.SHF 等其他映射类型
        like_parts = [f"{ticker_col} LIKE '{t}.%'" for t in root_codes]
        where_ticker = f"({' OR '.join(like_parts)})"

        start_str = pd.Timestamp(start).strftime("%Y%m%d")
        end_str = pd.Timestamp(end).strftime("%Y%m%d")

        sql = (
            f"SELECT {ticker_col} AS root, {mapped_col} AS main_contract, "
            f"{start_col} AS start_date, {end_col} AS end_date "
            f"FROM {tbl} WHERE {where_ticker} "
            f"AND {end_col} >= '{start_str}' AND {start_col} <= '{end_str}' "
            f"ORDER BY {ticker_col}, {start_col}"
        )

        try:
            df = self._read_sql(sql)
        except Exception:
            return pd.DataFrame()

        if df.empty:
            return df

        # 提取品种根代码: "RB.SHF" → "RB", "L.DCE" → "L", "IF.CFE" → "IF"
        # 不截断为 2 字符, 保留原始长度 (单字母品种如 L/P/J 也正确)
        df["root"] = df["root"].str.split(".").str[0]
        df["start_date"] = pd.to_datetime(df["start_date"].astype(str))
        df["end_date"] = pd.to_datetime(df["end_date"].astype(str))

        return df

    def _fetch_continuous_price(
        self,
        root_codes: TickerIndex,
        start: Date,
        end: Date,
        fields: List[str],
        mapping: pd.DataFrame,
    ) -> PricePanel:
        """使用主力合约映射构建连续合约价格序列 (比例后复权).

        自动路由: 根据品种代码查询对应的行情表
        (商品/股指/国债期货分别在不同表中).
        """
        from data.continuous_contract import build_continuous_series

        base_cfg = self._get_table_config("commodity_eod")
        if not base_cfg:
            return {}

        col_map = base_cfg.get("columns", {})
        date_col = col_map.get("date", "TRADE_DT")
        ticker_col = col_map.get("ticker", "S_INFO_WINDCODE")

        # 按品种分组, 确定每个品种对应的行情表
        root_to_table: Dict[str, str] = {}
        table_to_roots: Dict[str, list] = {}
        for root in root_codes:
            tbl = self._get_price_table_for_root(str(root))
            root_to_table[root] = tbl
            table_to_roots.setdefault(tbl, []).append(root)

        # 按表分组查询, 合并结果
        contract_data: Dict[str, pd.DataFrame] = {}
        for table_name, roots in table_to_roots.items():
            # 获取该表对应品种的主力合约
            table_mapping = mapping[mapping["root"].isin(roots)]
            if table_mapping.empty:
                continue
            table_contracts = table_mapping["main_contract"].unique().tolist()

            # 使用该表的配置 (复用列映射, 替换表名)
            tbl_cfg = self._get_table_config_for(table_name)
            sql = self._build_query(
                tbl_cfg, fields + ["date", "ticker"], table_contracts, start, end,
            )
            try:
                df = self._read_sql(sql)
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(
                    f"查询 {table_name} 失败 ({roots[:3]}...): {e}"
                )
                continue

            if df.empty:
                continue

            df[date_col] = pd.to_datetime(df[date_col].astype(str))

            # 按合约分组, 重命名 Wind 列为标准字段名
            for contract in table_contracts:
                mask = df[ticker_col] == contract
                if mask.any():
                    cdf = df[mask].set_index(date_col).sort_index()
                    rename = {}
                    for field in fields:
                        wind_col = col_map.get(field, field)
                        if wind_col in cdf.columns and wind_col != field:
                            rename[wind_col] = field
                    if rename:
                        cdf = cdf.rename(columns=rename)
                    contract_data[contract] = cdf

        # 为每个品种构建连续合约
        root_series: Dict[str, pd.DataFrame] = {}
        for root in root_codes:
            root_schedule = mapping[mapping["root"] == root]
            if root_schedule.empty:
                continue
            continuous = build_continuous_series(contract_data, root_schedule, fields)
            if not continuous.empty:
                root_series[root] = continuous

        if not root_series:
            return {}

        # 转换为 PricePanel 格式: {field: DataFrame(dates × roots)}
        price_panel: PricePanel = {}
        for field in fields:
            col_data = {}
            for root, series in root_series.items():
                if field in series.columns:
                    col_data[root] = series[field]
            if col_data:
                panel = pd.DataFrame(col_data)
                panel.index.name = "date"
                panel.columns.name = None
                price_panel[field] = panel
            else:
                price_panel[field] = pd.DataFrame()

        return price_panel

    def fetch_fundamental(
        self,
        tickers: TickerIndex,
        start: Date,
        end: Date,
        fields: List[str],
    ) -> dict:
        """期货无基本面，返回空 dict."""
        return {}

    def fetch_industry(
        self,
        tickers: TickerIndex,
        date: Date,
    ) -> IndustryMapping:
        """从品种描述表获取品种→板块映射."""
        cfg = self._get_table_config("contract_description")
        if not cfg:
            return pd.Series(dtype=object)
        col_map = cfg.get("columns", {})
        tbl = cfg["table_name"]
        ticker_col = col_map.get("ticker", "S_INFO_WINDCODE")
        name_col = col_map.get("name", "S_INFO_NAME")

        sql = (
            f"SELECT DISTINCT {ticker_col} AS ticker, "
            f"{name_col} AS name FROM {tbl}"
        )
        try:
            df = self._read_sql(sql)
        except Exception:
            return pd.Series(dtype=object)
        if df.empty:
            return pd.Series(dtype=object)
        # 取品种前缀 (前 2 位)
        df["root"] = df["ticker"].str[:2]
        return df.set_index("ticker")["root"]

    def fetch_index_constituents(
        self,
        index_code: str,
        date: Date,
    ) -> Universe:
        """从主力映射表获取所有活跃品种的根代码 (RB, CU, AU, IF, T...).

        不再限制返回数量, 自动包含商品+股指+国债期货.
        """
        cfg = self._get_table_config("contract_mapping")
        if not cfg:
            return pd.Index([])
        col_map = cfg.get("columns", {})
        tbl = cfg["table_name"]
        ticker_col = col_map.get("ticker", "S_INFO_WINDCODE")

        # 从映射表获取所有品种根代码 (格式: RB.SHF, L.DCE, IF.CFE, T.CFE)
        sql = f"SELECT DISTINCT {ticker_col} AS root_code FROM {tbl}"
        try:
            df = self._read_sql(sql)
        except Exception:
            return pd.Index([])
        if df.empty:
            return pd.Index([])

        # 提取品种根代码: "RB.SHF" → "RB", "L.DCE" → "L", "IF.CFE" → "IF"
        roots = []
        for code in df["root_code"].unique():
            if not isinstance(code, str) or "." not in code:
                continue
            root = code.split(".")[0]
            # 只保留 1-2 位纯字母代码
            if 1 <= len(root) <= 2 and root.isalpha():
                roots.append(root)

        # 去重并排序 (不再截断前 20 个)
        roots = sorted(set(roots))
        return pd.Index(roots)

    def fetch_listing_dates(self, roots: list) -> pd.Series:
        """从主力合约映射表获取品种上市日期 (取该品种最早主力合约的起始日).

        使用 cfuturescontractmapping 表的 MIN(STARTDATE) 作为品种上市日期,
        而非 cfuturesdescription 表 (后者可能包含已退市的老合约导致日期偏早).

        Args:
            roots: 品种根代码列表, 如 ['RB', 'IF', 'T']

        Returns:
            pd.Series: index=root_code, values=list_date (datetime)
                      未找到的品种返回 NaT
        """
        cfg = self._get_table_config("contract_mapping")
        if not cfg:
            return pd.Series(dtype=object)
        col_map = cfg.get("columns", {})
        tbl = cfg["table_name"]
        ticker_col = col_map.get("ticker", "S_INFO_WINDCODE")
        start_col = col_map.get("start_date", "STARTDATE")

        # 查询每个品种根代码的最早主力合约起始日
        # 映射表中 ticker 格式为 "RB.SHF", "IF.CFE", "T.CFE" 等
        root_list = list(roots)
        like_parts = [f"{ticker_col} LIKE '{r}.%'" for r in root_list]
        where_ticker = f"({' OR '.join(like_parts)})"

        sql = (
            f"SELECT {ticker_col} AS root, MIN({start_col}) AS list_date "
            f"FROM {tbl} WHERE {where_ticker} "
            f"GROUP BY {ticker_col}"
        )
        try:
            df = self._read_sql(sql)
        except Exception:
            return pd.Series(dtype=object)
        if df.empty:
            return pd.Series(dtype=object)

        # 提取品种根代码: "RB.SHF" → "RB", "IF.CFE" → "IF"
        df["root"] = df["root"].str.split(".").str[0]
        df["list_date"] = pd.to_datetime(df["list_date"].astype(str), errors="coerce")

        result = df.set_index("root")["list_date"]
        return result.reindex(roots)

    def fetch_contract_pair_prices(
        self,
        root_codes: list,
        start: Date,
        end: Date,
        field: str = "close",
    ) -> Dict[str, pd.DataFrame]:
        """查询每个品种每个交易日的近月和远月合约价格.

        近月/远月确定方式 (避免未来函数):
        - 查询日期范围内每个品种的所有活跃合约
        - 对每个交易日, 按合约 Wind 代码字符串排序 (等同于到期日排序)
        - 取该日有数据的最近到期合约作为近月, 次近作为远月
        - 退市日当天若仍有数据则计入, 次日自动切换到下一合约

        注意: 不使用 cfuturescontractmapping 主力合约表, 因为:
        1. 主力合约表只给一个合约, 无法区分近月/远月
        2. 主力合约切换规则可能基于成交量, 而成交量数据可能有未来信息

        Args:
            root_codes: 品种根代码列表, 如 ['RB', 'IF']
            start/end: 日期范围
            field: 价格字段 (close/settle/open 等)

        Returns:
            {"near": DataFrame(dates × roots), "far": DataFrame(dates × roots)}
            近月/远月合约的原始价格 (未复权, 允许换月跳空)
        """
        cache_key = (
            tuple(str(root) for root in root_codes),
            pd.Timestamp(start),
            pd.Timestamp(end),
            str(field),
        )
        cached = self._contract_pair_cache.get(cache_key)
        if cached is not None:
            return {
                "near": cached["near"].copy(deep=True),
                "far": cached["far"].copy(deep=True),
            }

        base_cfg = self._get_table_config("commodity_eod")
        if not base_cfg:
            return {"near": pd.DataFrame(), "far": pd.DataFrame()}

        col_map = base_cfg.get("columns", {})
        date_col = col_map.get("date", "TRADE_DT")
        ticker_col = col_map.get("ticker", "S_INFO_WINDCODE")
        wind_field = col_map.get(field, field)

        # 按品种根代码路由到行情表
        root_to_table: Dict[str, str] = {}
        table_to_roots: Dict[str, list] = {}
        for root in root_codes:
            tbl = self._get_price_table_for_root(str(root))
            root_to_table[root] = tbl
            table_to_roots.setdefault(tbl, []).append(root)

        near_panels: Dict[str, pd.Series] = {}
        far_panels: Dict[str, pd.Series] = {}

        start_str = pd.Timestamp(start).strftime("%Y%m%d")
        end_str = pd.Timestamp(end).strftime("%Y%m%d")

        for table_name, roots in table_to_roots.items():
            tbl_cfg = self._get_table_config_for(table_name)

            # 查询所有匹配品种的合约 (用 LIKE 'RB%' 匹配所有 RB 系列合约)
            like_parts = [f"{ticker_col} LIKE '{r}%'" for r in roots]
            where_ticker = f"({' OR '.join(like_parts)})"

            sql = (
                f"SELECT {date_col}, {ticker_col}, {wind_field} "
                f"FROM {table_name} "
                f"WHERE {where_ticker} "
                f"AND {date_col} >= '{start_str}' "
                f"AND {date_col} <= '{end_str}' "
                f"ORDER BY {ticker_col}, {date_col}"
            )

            try:
                df = self._read_sql(sql)
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(
                    f"fetch_contract_pair_prices 查询 {table_name} 失败: {e}"
                )
                continue

            if df.empty:
                continue

            df[date_col] = pd.to_datetime(df[date_col].astype(str))
            # Extract the root once for the whole query. The previous code ran
            # a Python lambda over the full table once per requested root.
            df["_root"] = df[ticker_col].astype(str).str.extract(
                r"^([A-Za-z]+)(?=\d)", expand=False
            )

            # 按品种根代码分组处理
            for root in roots:
                root_df = df[df["_root"] == str(root)]

                if root_df.empty:
                    continue

                # 删除价格为 NaN 的行
                root_df = root_df.dropna(subset=[wind_field])

                # 向量化: pivot 成 日期×合约 矩阵, 列按合约代码排序
                pivot = root_df.pivot_table(
                    index=date_col, columns=ticker_col, values=wind_field,
                    aggfunc="first",
                )
                # 按合约代码排序 (Wind 代码排序 ≈ 到期日排序)
                pivot = pivot.reindex(sorted(pivot.columns), axis=1)

                if pivot.empty:
                    continue

                # 逐行取第一个非 NaN 作为近月, 第二个非 NaN 作为远月
                # 向量化: 用 argsort 按 NaN 位置取前两个
                vals = pivot.values  # (n_dates, n_contracts)
                # 构造 mask: True 表示有值
                mask = ~np.isnan(vals)
                # 每行第一个非 NaN 的列索引 (近月)
                # np.argmax 返回第一个 True 的位置
                near_idx = np.argmax(mask, axis=1)
                near_valid = mask.any(axis=1)
                # 远月: 把近月位置置为 False 后再取第一个
                mask_after_near = mask.copy()
                rows_with_near = np.where(near_valid)[0]
                mask_after_near[rows_with_near, near_idx[rows_with_near]] = False
                far_idx = np.argmax(mask_after_near, axis=1)
                far_valid = mask_after_near.any(axis=1)

                # 提取近月/远月价格
                near_vals = np.where(near_valid, vals[np.arange(len(vals)), near_idx], np.nan)
                far_vals = np.where(far_valid, vals[np.arange(len(vals)), far_idx], np.nan)

                near_panels[root] = pd.Series(near_vals, index=pivot.index, name=root)
                far_panels[root] = pd.Series(far_vals, index=pivot.index, name=root)

        near_df = pd.DataFrame(near_panels) if near_panels else pd.DataFrame()
        far_df = pd.DataFrame(far_panels) if far_panels else pd.DataFrame()

        if not near_df.empty:
            near_df.index.name = "date"
        if not far_df.empty:
            far_df.index.name = "date"

        result = {"near": near_df, "far": far_df}
        self._contract_pair_cache[cache_key] = result
        return {
            "near": near_df.copy(deep=True),
            "far": far_df.copy(deep=True),
        }

    def fetch_calendar(self, start: Date, end: Date) -> DateIndex:
        """从交易日历表读取."""
        cfg = self._get_table_config("calendar")
        if not cfg:
            return pd.DatetimeIndex([])
        col_map = cfg.get("columns", {})
        tbl = cfg["table_name"]
        day_col = col_map.get("day", "TRADE_DAYS")

        sql = f"SELECT DISTINCT {day_col} AS day FROM {tbl} ORDER BY day"
        try:
            df = self._read_sql(sql)
        except Exception:
            return pd.DatetimeIndex([])
        if df.empty:
            return pd.DatetimeIndex([])
        days = pd.to_datetime(df["day"].astype(str))
        filtered = days[
            (days >= pd.Timestamp(start)) & (days <= pd.Timestamp(end))
        ]
        return pd.DatetimeIndex(filtered)
