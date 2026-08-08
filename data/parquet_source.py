"""Local partitioned-Parquet futures market-data source.

The source builds a point-in-time continuous series from concrete contracts.
The contract used on trading day ``t`` is the vendor dominant contract observed
on the previous trading day. Price fields are forward ratio-adjusted at a roll
using prices that were already known at the end of ``t-1``.
"""
from __future__ import annotations

from collections import OrderedDict
from datetime import date
import hashlib
import json
import logging
import os
from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from core.interfaces import DataSource
from core.registry import register
from core.sectors import sector_for


logger = logging.getLogger(__name__)

_SYMBOL_RE = re.compile(r"^([A-Za-z]+)(\d+)$")
_SYNTHETIC_SUFFIXES = {"8888", "9998", "9999"}
_RAW_FIELD_MAP = {
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume",
    "amount": "amount",
    "oi": "position",
}
_CURVE_FIELDS = (
    "curve_total_oi",
    "curve_top2_oi",
    "curve_total_volume",
    "curve_contract_count",
    "curve_oi_breadth",
    "curve_oi_concentration",
    "curve_oi_hhi",
)
_PRICE_FIELDS = {"open", "high", "low", "close"}
_DEFAULT_EAGER_FIELDS = tuple(_RAW_FIELD_MAP)
_CURVE_CACHE_SCHEMA_VERSION = 1
_SELECTED_CACHE_SCHEMA_VERSION = 1
_FREQUENCY_ROUTE = {
    "daily": ("daily", None),
    "1min": ("1min", None),
    "5min": ("1min", "5min"),
    "15min": ("15min", None),
    "30min": ("15min", "30min"),
    "hourly": ("15min", "60min"),
}


def _normalise_frequency(value: str) -> str:
    aliases = {
        "1m": "1min",
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "60m": "hourly",
        "60min": "hourly",
        "1h": "hourly",
    }
    frequency = aliases.get(str(value).lower(), str(value).lower())
    if frequency not in _FREQUENCY_ROUTE:
        raise ValueError(
            f"unsupported parquet frequency {value!r}; "
            f"expected one of {sorted(_FREQUENCY_ROUTE)}"
        )
    return frequency


def _root_and_suffix(symbol: str) -> tuple[str, str]:
    match = _SYMBOL_RE.fullmatch(str(symbol).strip())
    if not match:
        return "", ""
    return match.group(1).upper(), match.group(2)


@register("data_source", "parquet_futures")
class ParquetFuturesSource(DataSource):
    """Read futures bars from Hive-style ``year_month=YYYY-MM`` partitions."""

    market = "futures"

    def __init__(
        self,
        parquet_config: Optional[dict] = None,
        mysql_config: Optional[dict] = None,
    ) -> None:
        config = dict(parquet_config or {})
        root_text = str(config.get("root_path", "")).strip()
        if not root_text:
            raise ValueError("parquet.root_path is required")
        self.root_path = Path(root_text).expanduser().resolve()
        if not self.root_path.is_dir():
            raise FileNotFoundError(self.root_path)

        datasets = dict(config.get("datasets") or {})
        self.datasets = {
            "daily": datasets.get("daily", "futureshistoryprices1d"),
            "1min": datasets.get("1min", "futureshistoryprices1m"),
            "15min": datasets.get("15min", "futureshistoryprices15m"),
        }
        # 日历缓存 (fetch_calendar 每次全量读 1d + 正则, 静态数据按 (start,end) 缓存)
        self._CALENDAR_CACHE: dict = {}
        for name, relative in self.datasets.items():
            path = self.root_path / str(relative)
            if not path.is_dir():
                raise FileNotFoundError(f"parquet dataset {name!r} not found: {path}")

        self.dominant_lag_days = max(int(config.get("dominant_lag_days", 1)), 1)
        self.schedule_buffer_days = max(
            int(config.get("schedule_buffer_days", 45)), 10
        )
        self.eager_fields = bool(config.get("eager_fields", True))
        self.panel_cache_entries = max(int(config.get("panel_cache_entries", 1)), 0)
        self.curve_cache_enabled = bool(config.get("curve_cache_enabled", False))
        curve_cache_text = str(
            config.get("curve_cache_path", "./cache/curve_aggregates")
        ).strip()
        self.curve_cache_path = Path(curve_cache_text).expanduser().resolve()
        if self.curve_cache_enabled:
            self.curve_cache_path.mkdir(parents=True, exist_ok=True)
        self.selected_cache_enabled = bool(
            config.get("selected_cache_enabled", False)
        )
        selected_cache_text = str(
            config.get("selected_cache_path", "./cache/selected_contracts")
        ).strip()
        self.selected_cache_path = Path(
            selected_cache_text
        ).expanduser().resolve()
        if self.selected_cache_enabled:
            self.selected_cache_path.mkdir(parents=True, exist_ok=True)
        self._schema_cache: Dict[Path, set[str]] = {}
        self._plan_cache: OrderedDict[tuple, pd.DataFrame] = OrderedDict()
        self._panel_cache: OrderedDict[tuple, Dict[str, pd.DataFrame]] = OrderedDict()

        self._macro_source = None
        if mysql_config:
            try:
                from data.mysql_source import MySQLSource

                self._macro_source = MySQLSource(mysql_config)
            except Exception:
                logger.warning(
                    "MySQL macro delegate is unavailable for parquet source",
                    exc_info=True,
                )

    @staticmethod
    def _normalise_tickers(tickers: Iterable[str]) -> tuple[str, ...]:
        return tuple(dict.fromkeys(str(item).upper() for item in tickers))

    def _dataset_path(self, native_frequency: str) -> Path:
        return self.root_path / self.datasets[native_frequency]

    def _month_files(
        self, native_frequency: str, start, end
    ) -> list[Path]:
        dataset = self._dataset_path(native_frequency)
        periods = pd.period_range(
            pd.Timestamp(start).to_period("M"),
            pd.Timestamp(end).to_period("M"),
            freq="M",
        )
        files: list[Path] = []
        for period in periods:
            directory = dataset / f"year_month={period}"
            if directory.is_dir():
                files.extend(sorted(directory.glob("*.parquet")))
        return files

    def _curve_cache_source_fingerprint(
        self,
        native_frequency: str,
        month_start: pd.Timestamp,
        month_end: pd.Timestamp,
    ) -> str:
        files = self._month_files(
            native_frequency,
            month_start - pd.Timedelta(days=7),
            month_end,
        )
        files.extend(
            self._month_files(
                "daily",
                month_start - pd.Timedelta(days=15),
                month_end + pd.Timedelta(days=7),
            )
        )
        records = []
        for path in sorted(set(files), key=lambda item: str(item)):
            stat = path.stat()
            try:
                relative = path.relative_to(self.root_path).as_posix()
            except ValueError:
                relative = str(path)
            records.append((relative, int(stat.st_size), int(stat.st_mtime_ns)))
        payload = json.dumps(records, ensure_ascii=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _files_fingerprint(root_path: Path, files: Iterable[Path]) -> str:
        records = []
        for path in sorted(set(files), key=lambda item: str(item)):
            stat = path.stat()
            try:
                relative = path.relative_to(root_path).as_posix()
            except ValueError:
                relative = str(path)
            records.append((relative, int(stat.st_size), int(stat.st_mtime_ns)))
        payload = json.dumps(records, ensure_ascii=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _selected_cache_source_fingerprint(
        self,
        native_frequency: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> str:
        files = self._month_files(
            native_frequency, start.normalize() - pd.Timedelta(days=7), end
        )
        files.extend(
            self._month_files(
                "daily",
                start.normalize() - pd.Timedelta(days=self.schedule_buffer_days),
                end.normalize() + pd.Timedelta(days=7),
            )
        )
        return self._files_fingerprint(self.root_path, files)

    def _selected_cache_files(
        self,
        frequency: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        tickers: tuple[str, ...],
        fields: tuple[str, ...],
    ) -> tuple[Path, Path]:
        payload = json.dumps(
            {
                "frequency": frequency,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "tickers": list(tickers),
                "fields": list(fields),
            },
            ensure_ascii=True,
            sort_keys=True,
        )
        request_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
        stem = f"selected_v{_SELECTED_CACHE_SCHEMA_VERSION}_{frequency}_{request_hash}"
        return (
            self.selected_cache_path / f"{stem}.parquet",
            self.selected_cache_path / f"{stem}.json",
        )

    def _read_selected_cache(
        self,
        frequency: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        tickers: tuple[str, ...],
        fields: tuple[str, ...],
        source_fingerprint: str,
    ) -> Optional[pd.DataFrame]:
        if not self.selected_cache_enabled:
            return None
        data_path, metadata_path = self._selected_cache_files(
            frequency, start, end, tickers, fields
        )
        if not data_path.is_file() or not metadata_path.is_file():
            return None
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            expected = {
                "schema_version": _SELECTED_CACHE_SCHEMA_VERSION,
                "frequency": frequency,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "tickers": list(tickers),
                "fields": list(fields),
                "source_fingerprint": source_fingerprint,
            }
            if any(metadata.get(key) != value for key, value in expected.items()):
                return None
            frame = pd.read_parquet(data_path)
            required = {"trade_datetime", "root", *fields}
            if not required.issubset(frame.columns):
                return None
            frame["trade_datetime"] = pd.to_datetime(frame["trade_datetime"])
            if not set(frame["root"].dropna().astype(str)).issubset(tickers):
                return None
            return frame
        except Exception:
            logger.warning("selected-contract cache read failed: %s", data_path)
            return None

    def _write_selected_cache(
        self,
        frame: pd.DataFrame,
        frequency: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        tickers: tuple[str, ...],
        fields: tuple[str, ...],
        source_fingerprint: str,
    ) -> None:
        if not self.selected_cache_enabled or frame.empty:
            return
        data_path, metadata_path = self._selected_cache_files(
            frequency, start, end, tickers, fields
        )
        suffix = f".{os.getpid()}.tmp"
        data_temp = data_path.with_name(data_path.name + suffix)
        metadata_temp = metadata_path.with_name(metadata_path.name + suffix)
        metadata = {
            "schema_version": _SELECTED_CACHE_SCHEMA_VERSION,
            "frequency": frequency,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "tickers": list(tickers),
            "fields": list(fields),
            "source_fingerprint": source_fingerprint,
        }
        try:
            frame.to_parquet(data_temp, index=False)
            metadata_temp.write_text(
                json.dumps(metadata, ensure_ascii=True, sort_keys=True),
                encoding="utf-8",
            )
            data_temp.replace(data_path)
            metadata_temp.replace(metadata_path)
        except Exception:
            logger.warning("selected-contract cache write failed: %s", data_path)
        finally:
            data_temp.unlink(missing_ok=True)
            metadata_temp.unlink(missing_ok=True)

    def _curve_cache_files(
        self,
        frequency: str,
        period: pd.Period,
        tickers: tuple[str, ...],
    ) -> tuple[Path, Path]:
        roots_text = "\x1f".join(tickers)
        roots_hash = hashlib.sha256(roots_text.encode("utf-8")).hexdigest()[:16]
        stem = f"curve_v{_CURVE_CACHE_SCHEMA_VERSION}_{frequency}_{period}_{roots_hash}"
        return (
            self.curve_cache_path / f"{stem}.parquet",
            self.curve_cache_path / f"{stem}.json",
        )

    def _read_curve_month_cache(
        self,
        frequency: str,
        period: pd.Period,
        tickers: tuple[str, ...],
        source_fingerprint: str,
    ) -> Optional[pd.DataFrame]:
        if not self.curve_cache_enabled:
            return None
        data_path, metadata_path = self._curve_cache_files(
            frequency, period, tickers
        )
        if not data_path.is_file() or not metadata_path.is_file():
            return None
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            expected = {
                "schema_version": _CURVE_CACHE_SCHEMA_VERSION,
                "frequency": frequency,
                "period": str(period),
                "tickers": list(tickers),
                "source_fingerprint": source_fingerprint,
            }
            if any(metadata.get(key) != value for key, value in expected.items()):
                return None
            frame = pd.read_parquet(data_path)
            required = {"trade_datetime", "root", *_CURVE_FIELDS}
            if not required.issubset(frame.columns):
                return None
            frame["trade_datetime"] = pd.to_datetime(frame["trade_datetime"])
            if not set(frame["root"].dropna().astype(str)).issubset(tickers):
                return None
            return frame
        except Exception:
            logger.warning("curve aggregate cache read failed: %s", data_path)
            return None

    def _write_curve_month_cache(
        self,
        frame: pd.DataFrame,
        frequency: str,
        period: pd.Period,
        tickers: tuple[str, ...],
        source_fingerprint: str,
    ) -> None:
        if not self.curve_cache_enabled or frame.empty:
            return
        data_path, metadata_path = self._curve_cache_files(
            frequency, period, tickers
        )
        suffix = f".{os.getpid()}.tmp"
        data_temp = data_path.with_name(data_path.name + suffix)
        metadata_temp = metadata_path.with_name(metadata_path.name + suffix)
        metadata = {
            "schema_version": _CURVE_CACHE_SCHEMA_VERSION,
            "frequency": frequency,
            "period": str(period),
            "tickers": list(tickers),
            "source_fingerprint": source_fingerprint,
        }
        try:
            frame.to_parquet(data_temp, index=False)
            metadata_temp.write_text(
                json.dumps(metadata, ensure_ascii=True, sort_keys=True),
                encoding="utf-8",
            )
            data_temp.replace(data_path)
            metadata_temp.replace(metadata_path)
        except Exception:
            logger.warning("curve aggregate cache write failed: %s", data_path)
        finally:
            data_temp.unlink(missing_ok=True)
            metadata_temp.unlink(missing_ok=True)

    def _available_columns(self, path: Path) -> set[str]:
        cached = self._schema_cache.get(path)
        if cached is not None:
            return cached
        import pyarrow.parquet as pq

        columns = set(pq.ParquetFile(path).schema_arrow.names)
        self._schema_cache[path] = columns
        return columns

    def _read_partitions(
        self,
        native_frequency: str,
        start,
        end,
        columns: Iterable[str],
    ) -> pd.DataFrame:
        files = self._month_files(native_frequency, start, end)
        if not files:
            return pd.DataFrame(columns=list(columns))
        available = self._available_columns(files[0])
        selected = [column for column in dict.fromkeys(columns) if column in available]
        if not selected:
            return pd.DataFrame()
        filters = [
            ("trade_date", ">=", pd.Timestamp(start).date()),
            ("trade_date", "<=", pd.Timestamp(end).date()),
        ]
        # 逐文件读取再 concat: 避免 pyarrow 多文件 schema merge 时
        # dictionary 编码 vs plain string 的 year_month 列类型冲突 (ArrowTypeError)
        frames = []
        for f in files:
            try:
                df = pd.read_parquet(f, columns=selected, filters=filters)
                if not df.empty:
                    frames.append(df)
            except Exception:
                continue
        if not frames:
            return pd.DataFrame(columns=selected)
        return pd.concat(frames, ignore_index=True)

    @staticmethod
    def _annotate_symbols(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty or "symbol" not in frame:
            return frame
        result = frame.copy()
        result["symbol"] = result["symbol"].astype(str).str.strip().str.upper()
        parsed = result["symbol"].str.extract(_SYMBOL_RE)
        result["root"] = parsed[0].str.upper()
        result["suffix"] = parsed[1]
        result["trade_date"] = pd.to_datetime(result["trade_date"]).dt.normalize()
        return result.dropna(subset=["root", "suffix", "trade_date"])

    def _assign_intraday_trade_dates(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Replace vendor natural dates with exchange trading dates."""
        if frame.empty:
            return frame
        result = frame.copy()
        timestamps = pd.to_datetime(result["trade_datetime"], errors="coerce")
        natural_dates = timestamps.dt.normalize()
        targets = natural_dates.where(
            timestamps.dt.hour.lt(18), natural_dates + pd.Timedelta(days=1)
        )
        calendar = self.fetch_calendar(
            targets.min() - pd.Timedelta(days=7),
            targets.max() + pd.Timedelta(days=7),
        )
        calendar = pd.DatetimeIndex(calendar).normalize().unique().sort_values()
        if len(calendar) == 0:
            result["trade_date"] = targets.map(
                lambda value: value + pd.offsets.BDay(0)
            )
            return result
        locations = calendar.searchsorted(targets.to_numpy(), side="left")
        assigned = np.full(
            len(targets), np.datetime64("NaT"), dtype="datetime64[ns]"
        )
        valid = locations < len(calendar)
        assigned[valid] = calendar.to_numpy()[locations[valid]]
        result["trade_date"] = assigned
        return result

    @staticmethod
    def _deduplicate(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
        if frame.empty:
            return frame
        sort_columns = list(keys)
        if "sequence" in frame.columns:
            sort_columns.append("sequence")
        return (
            frame.sort_values(sort_columns)
            .drop_duplicates(keys, keep="last")
            .reset_index(drop=True)
        )

    def _infer_vendor_main(
        self, frame: pd.DataFrame, tickers: tuple[str, ...]
    ) -> pd.DataFrame:
        frame = self._annotate_symbols(frame)
        frame = frame.loc[frame["root"].isin(tickers)]
        concrete = frame.loc[~frame["suffix"].isin(_SYNTHETIC_SUFFIXES)].copy()
        concrete = self._deduplicate(concrete, ["trade_date", "root", "symbol"])
        synthetic = frame.loc[frame["suffix"].eq("9999")].copy()
        synthetic = self._deduplicate(synthetic, ["trade_date", "root"])

        fallback = (
            concrete.sort_values(
                ["trade_date", "root", "position", "volume"],
                ascending=[True, True, False, False],
                na_position="last",
            )
            .drop_duplicates(["trade_date", "root"], keep="first")
            [["trade_date", "root", "symbol"]]
        )
        if synthetic.empty or concrete.empty:
            return fallback

        target_columns = [
            "trade_date", "root", "close", "volume", "position"
        ]
        targets = synthetic[target_columns].rename(
            columns={
                "close": "target_close",
                "volume": "target_volume",
                "position": "target_position",
            }
        )
        candidates = concrete.merge(targets, on=["trade_date", "root"], how="inner")
        close_scale = candidates["target_close"].abs().clip(lower=1.0)
        candidates["match_score"] = (
            (candidates["close"] - candidates["target_close"]).abs() / close_scale
            + (
                np.log1p(candidates["volume"].clip(lower=0))
                - np.log1p(candidates["target_volume"].clip(lower=0))
            ).abs()
            + (
                np.log1p(candidates["position"].clip(lower=0))
                - np.log1p(candidates["target_position"].clip(lower=0))
            ).abs()
        )
        matched = (
            candidates.sort_values(
                ["trade_date", "root", "match_score", "position"],
                ascending=[True, True, True, False],
                na_position="last",
            )
            .drop_duplicates(["trade_date", "root"], keep="first")
            [["trade_date", "root", "symbol"]]
        )
        keys = pd.MultiIndex.from_frame(matched[["trade_date", "root"]])
        fallback_keys = pd.MultiIndex.from_frame(fallback[["trade_date", "root"]])
        missing = fallback.loc[~fallback_keys.isin(keys)]
        return pd.concat([matched, missing], ignore_index=True).sort_values(
            ["root", "trade_date"]
        )

    def _continuous_plan(
        self, tickers: tuple[str, ...], start, end
    ) -> pd.DataFrame:
        start_ts = pd.Timestamp(start).normalize()
        end_ts = pd.Timestamp(end).normalize()
        key = (tickers, start_ts, end_ts, self.dominant_lag_days)
        cached = self._plan_cache.get(key)
        if cached is not None:
            self._plan_cache.move_to_end(key)
            return cached.copy(deep=False)

        buffer_start = start_ts - pd.Timedelta(days=self.schedule_buffer_days)
        daily = self._read_partitions(
            "daily",
            buffer_start,
            end_ts,
            [
                "symbol", "trade_date", "close", "volume", "position",
                "sequence",
            ],
        )
        daily = self._annotate_symbols(daily)
        daily = daily.loc[daily["root"].isin(tickers)]
        mapping = self._infer_vendor_main(daily, tickers)
        mapping = mapping.sort_values(["root", "trade_date"])
        mapping["contract"] = mapping.groupby("root")["symbol"].shift(
            self.dominant_lag_days
        )
        mapping["contract"] = mapping["contract"].fillna(mapping["symbol"])

        concrete = daily.loc[~daily["suffix"].isin(_SYNTHETIC_SUFFIXES)].copy()
        concrete = self._deduplicate(concrete, ["trade_date", "root", "symbol"])
        available = pd.MultiIndex.from_frame(
            concrete.loc[concrete["close"].notna(), ["trade_date", "root", "symbol"]]
        )
        selected_keys = pd.MultiIndex.from_frame(
            mapping[["trade_date", "root", "contract"]].rename(
                columns={"contract": "symbol"}
            )
        )
        current_keys = pd.MultiIndex.from_frame(
            mapping[["trade_date", "root", "symbol"]]
        )
        selected_missing = ~selected_keys.isin(available)
        current_available = current_keys.isin(available)
        mapping.loc[selected_missing & current_available, "contract"] = mapping.loc[
            selected_missing & current_available, "symbol"
        ]
        unresolved = selected_missing & ~current_available
        if unresolved.any():
            sample = mapping.loc[
                unresolved, ["trade_date", "root", "contract", "symbol"]
            ].head(5).to_dict("records")
            raise ValueError(
                "continuous-contract plan has no available selected or current "
                f"contract close: {sample}"
            )
        plans = []
        for root, group in mapping.groupby("root", sort=False):
            adjustment = 1.0
            previous_contract = None
            previous_date = None
            for row in group.itertuples(index=False):
                contract = str(row.contract)
                if (
                    previous_contract is not None
                    and contract != previous_contract
                    and previous_date is not None
                ):
                    root_prices = concrete.loc[
                        concrete["root"].eq(root)
                        & concrete["symbol"].isin([previous_contract, contract])
                        # 共同报价限换月日当天及之前 (防用换月后未来价调整, 与 continuous_contract.py 一致)
                        & concrete["trade_date"].ge(previous_date - pd.Timedelta(days=2))
                        & concrete["trade_date"].le(previous_date),
                        ["trade_date", "symbol", "close"],
                    ].pivot_table(
                        index="trade_date", columns="symbol", values="close",
                        aggfunc="last",
                    )
                    if previous_contract not in root_prices or contract not in root_prices:
                        common = pd.DataFrame()
                    else:
                        # 取换月日(±2天)起的共同 close: 容忍主力切换日判定偏差
                        # (FU 等低流动性品种换月日早于新合约活跃, 原 le(previous_date)
                        # 会找不到共同 close). 正常品种仍取换月日附近共同日, 数值不变.
                        common = root_prices[
                            [previous_contract, contract]
                        ].dropna(how="any")
                    if common.empty:
                        # 换月对无共同交易日 (数据稀疏品种罕见情形):
                        # 跳过该换月点的比例调整 (adjustment 不变), 用旧合约价格续接.
                        # warning 级别让数据质量事件可见 (fail-open 但可观测).
                        import logging
                        logging.getLogger("multi_factor").warning(
                            "skip rollover adjustment (no common close) %s->%s at %s",
                            previous_contract, contract, previous_date.date(),
                        )
                        continue
                    old_close = float(common.iloc[-1][previous_contract])
                    new_close = float(common.iloc[-1][contract])
                    if not np.isfinite(new_close) or new_close <= 0.0:
                        raise ValueError(
                            f"invalid rollover close for {previous_contract}->{contract}"
                        )
                    adjustment *= old_close / new_close
                plans.append(
                    {
                        "trade_date": row.trade_date,
                        "root": root,
                        "contract": contract,
                        "adjustment": adjustment,
                    }
                )
                previous_contract = contract
                previous_date = row.trade_date

        plan = pd.DataFrame(plans)
        if not plan.empty:
            plan = plan.loc[
                plan["trade_date"].between(start_ts, end_ts)
            ].reset_index(drop=True)
        self._plan_cache[key] = plan
        while len(self._plan_cache) > 4:
            self._plan_cache.popitem(last=False)
        return plan.copy(deep=False)

    @staticmethod
    def _resample_long(frame: pd.DataFrame, rule: str) -> pd.DataFrame:
        if frame.empty:
            return frame
        work = frame.copy()
        work["trade_datetime"] = pd.to_datetime(
            work["trade_datetime"]
        ).dt.floor(rule)
        aggregations = {}
        for field, method in (
            ("open", "first"),
            ("high", "max"),
            ("low", "min"),
            ("close", "last"),
            ("volume", "sum"),
            ("amount", "sum"),
            ("oi", "last"),
        ):
            if field in work.columns:
                aggregations[field] = method
        return (
            work.groupby(["trade_datetime", "root"], sort=True, as_index=False)
            .agg(aggregations)
        )

    def _selected_long(
        self,
        tickers: tuple[str, ...],
        start,
        end,
        frequency: str,
        fields: tuple[str, ...],
    ) -> pd.DataFrame:
        native_frequency, resample_rule = _FREQUENCY_ROUTE[frequency]
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        source_fingerprint = self._selected_cache_source_fingerprint(
            native_frequency, start_ts, end_ts
        )
        cached = self._read_selected_cache(
            frequency,
            start_ts,
            end_ts,
            tickers,
            fields,
            source_fingerprint,
        )
        if cached is not None:
            return cached
        plan = self._continuous_plan(tickers, start, end)
        if plan.empty:
            return pd.DataFrame()

        raw_fields = [_RAW_FIELD_MAP[field] for field in fields if field in _RAW_FIELD_MAP]
        columns = ["symbol", "trade_datetime", "trade_date", "sequence"] + raw_fields
        frames = []
        for period in pd.period_range(
            pd.Timestamp(start).to_period("M"),
            pd.Timestamp(end).to_period("M"),
            freq="M",
        ):
            month_start = max(pd.Timestamp(start), period.start_time)
            month_end = min(pd.Timestamp(end), period.end_time)
            read_start = (
                month_start
                if native_frequency == "daily"
                else month_start.normalize() - pd.Timedelta(days=7)
            )
            raw = self._read_partitions(
                native_frequency, read_start, month_end, columns
            )
            if raw.empty:
                continue
            raw = self._annotate_symbols(raw)
            if native_frequency != "daily":
                raw = self._assign_intraday_trade_dates(raw)
            raw = raw.loc[
                raw["root"].isin(tickers)
                & ~raw["suffix"].isin(_SYNTHETIC_SUFFIXES)
            ]
            month_plan = plan.loc[
                plan["trade_date"].between(
                    pd.Timestamp(month_start).normalize(),
                    pd.Timestamp(month_end).normalize(),
                )
            ]
            selected = raw.merge(
                month_plan,
                on=["trade_date", "root"],
                how="inner",
                validate="many_to_one",
            )
            selected = selected.loc[selected["symbol"].eq(selected["contract"])]
            if selected.empty:
                continue
            selected["trade_datetime"] = pd.to_datetime(selected["trade_datetime"])
            selected = self._deduplicate(
                selected, ["trade_datetime", "root"]
            )
            for field in _PRICE_FIELDS.intersection(fields):
                selected[field] = selected[field].astype(float) * selected["adjustment"]
            if "oi" in fields and "position" in selected:
                selected["oi"] = selected["position"].astype(float)
            keep = ["trade_datetime", "root"] + [
                field for field in fields if field in selected.columns
            ]
            frames.append(selected[keep])

        if not frames:
            return pd.DataFrame()
        result = pd.concat(frames, ignore_index=True)
        result = self._deduplicate(result, ["trade_datetime", "root"])
        if resample_rule:
            result = self._resample_long(result, resample_rule)
        result = result.sort_values(["trade_datetime", "root"]).reset_index(drop=True)
        self._write_selected_cache(
            result,
            frequency,
            start_ts,
            end_ts,
            tickers,
            fields,
            source_fingerprint,
        )
        return result

    def _daily_curve_long(
        self,
        tickers: tuple[str, ...],
        start,
        end,
    ) -> pd.DataFrame:
        """Aggregate concrete-contract OI without using synthetic contracts."""
        buffer_start = pd.Timestamp(start).normalize() - pd.Timedelta(days=10)
        raw = self._read_partitions(
            "daily",
            buffer_start,
            pd.Timestamp(end).normalize(),
            [
                "symbol", "trade_datetime", "trade_date", "position",
                "volume", "sequence",
            ],
        )
        raw = self._annotate_symbols(raw)
        if raw.empty:
            return pd.DataFrame()
        concrete = raw.loc[
            raw["root"].isin(tickers)
            & ~raw["suffix"].isin(_SYNTHETIC_SUFFIXES)
        ].copy()
        concrete = self._deduplicate(
            concrete, ["trade_date", "root", "symbol"]
        )
        if concrete.empty:
            return pd.DataFrame()
        concrete["position"] = pd.to_numeric(
            concrete["position"], errors="coerce"
        ).clip(lower=0.0)
        concrete["volume"] = pd.to_numeric(
            concrete["volume"], errors="coerce"
        ).clip(lower=0.0)
        concrete = concrete.sort_values(["root", "symbol", "trade_date"])
        previous = concrete.groupby(["root", "symbol"])["position"].shift(1)
        concrete["oi_increased"] = (
            concrete["position"].gt(previous).where(previous.notna())
        )
        concrete["position_sq"] = concrete["position"].pow(2)

        keys = ["trade_date", "root"]
        aggregate = concrete.groupby(keys, sort=True).agg(
            curve_total_oi=("position", "sum"),
            curve_total_volume=("volume", "sum"),
            curve_contract_count=("position", lambda value: int(value.gt(0).sum())),
            curve_oi_breadth=("oi_increased", "mean"),
            position_sq_sum=("position_sq", "sum"),
        )
        top2 = (
            concrete.sort_values(
                keys + ["position"],
                ascending=[True, True, False],
                na_position="last",
            )
            .groupby(keys, sort=True)
            .head(2)
            .groupby(keys, sort=True)["position"]
            .sum()
            .rename("curve_top2_oi")
        )
        aggregate = aggregate.join(top2, how="left")
        denominator = aggregate["curve_total_oi"].replace(0.0, np.nan)
        aggregate["curve_oi_concentration"] = (
            aggregate["curve_top2_oi"] / denominator
        )
        aggregate["curve_oi_hhi"] = (
            aggregate.pop("position_sq_sum") / denominator.pow(2)
        )
        result = aggregate.reset_index().rename(
            columns={"trade_date": "trade_datetime"}
        )
        result["trade_datetime"] = pd.to_datetime(result["trade_datetime"])
        start_ts = pd.Timestamp(start).normalize()
        end_ts = pd.Timestamp(end).normalize()
        return result.loc[
            result["trade_datetime"].between(start_ts, end_ts)
        ].sort_values(["trade_datetime", "root"]).reset_index(drop=True)

    @staticmethod
    def _aggregate_intraday_states(
        raw: pd.DataFrame,
        daily: pd.DataFrame,
    ) -> pd.DataFrame:
        frames = []
        daily_by_root = {
            root: group for root, group in daily.groupby("root", sort=False)
        }
        for (root, trade_date), group in raw.groupby(
            ["root", "trade_date"], sort=True
        ):
            day = pd.Timestamp(trade_date).normalize()
            root_daily = daily_by_root.get(root, pd.DataFrame())
            baseline = pd.Series(dtype=float)
            if not root_daily.empty:
                previous_dates = root_daily.loc[
                    root_daily["trade_date"].lt(day), "trade_date"
                ]
                if not previous_dates.empty:
                    baseline_date = previous_dates.max()
                    baseline_rows = root_daily.loc[
                        root_daily["trade_date"].eq(baseline_date)
                    ]
                    baseline = baseline_rows.set_index("symbol")["position"]

            observed = group.pivot_table(
                index="trade_datetime",
                columns="symbol",
                values="position",
                aggfunc="last",
            ).sort_index()
            if observed.empty:
                continue
            contracts = observed.columns.union(baseline.index)
            observed = observed.reindex(columns=contracts)
            initial = baseline.reindex(contracts).fillna(0.0)
            states = observed.copy()
            states.iloc[0] = states.iloc[0].fillna(initial)
            states = states.ffill().fillna(0.0).clip(lower=0.0)

            state_values = states.to_numpy(dtype=float, copy=False)
            total_oi = state_values.sum(axis=1)
            if state_values.shape[1] <= 2:
                top2_oi = total_oi.copy()
            else:
                top2_oi = np.partition(
                    state_values, state_values.shape[1] - 2, axis=1
                )[:, -2:].sum(axis=1)
            contract_count = (state_values > 0.0).sum(axis=1)
            denominator = np.where(total_oi > 0.0, total_oi, np.nan)
            concentration = top2_oi / denominator
            hhi = np.square(state_values).sum(axis=1) / np.square(denominator)

            previous_states = states.shift(1)
            previous_states.iloc[0] = initial
            comparable = previous_states.notna()
            increased = states.gt(previous_states) & comparable
            breadth_denominator = comparable.sum(axis=1).replace(0, np.nan)
            breadth = increased.sum(axis=1) / breadth_denominator

            volume = group.pivot_table(
                index="trade_datetime",
                columns="symbol",
                values="volume",
                aggfunc="sum",
            ).reindex(index=states.index).fillna(0.0).sum(axis=1)
            frames.append(pd.DataFrame({
                "trade_datetime": states.index,
                "root": root,
                "curve_total_oi": total_oi,
                "curve_top2_oi": top2_oi,
                "curve_total_volume": volume.to_numpy(dtype=float),
                "curve_contract_count": contract_count,
                "curve_oi_breadth": breadth.to_numpy(dtype=float),
                "curve_oi_concentration": concentration,
                "curve_oi_hhi": hhi,
            }))
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    @staticmethod
    def _resample_curve_long(frame: pd.DataFrame, rule: str) -> pd.DataFrame:
        if frame.empty:
            return frame
        work = frame.copy()
        work["trade_datetime"] = pd.to_datetime(
            work["trade_datetime"]
        ).dt.floor(rule)
        return (
            work.groupby(["trade_datetime", "root"], sort=True, as_index=False)
            .agg({
                "curve_total_oi": "last",
                "curve_top2_oi": "last",
                "curve_total_volume": "sum",
                "curve_contract_count": "last",
                "curve_oi_breadth": "mean",
                "curve_oi_concentration": "last",
                "curve_oi_hhi": "last",
            })
        )

    def _intraday_curve_long(
        self,
        tickers: tuple[str, ...],
        start,
        end,
        frequency: str,
    ) -> pd.DataFrame:
        native_frequency, resample_rule = _FREQUENCY_ROUTE[frequency]
        frames = []
        for period in pd.period_range(
            pd.Timestamp(start).to_period("M"),
            pd.Timestamp(end).to_period("M"),
            freq="M",
        ):
            month_start = period.start_time
            month_end = period.end_time
            source_fingerprint = self._curve_cache_source_fingerprint(
                native_frequency, month_start, month_end
            )
            cached = self._read_curve_month_cache(
                frequency, period, tickers, source_fingerprint
            )
            if cached is not None:
                frames.append(cached)
                continue
            raw = self._read_partitions(
                native_frequency,
                month_start.normalize() - pd.Timedelta(days=7),
                month_end,
                [
                    "symbol", "trade_datetime", "trade_date", "position",
                    "volume",
                ],
            )
            raw = self._annotate_symbols(raw)
            raw = self._assign_intraday_trade_dates(raw)
            raw = raw.loc[
                raw["root"].isin(tickers)
                & ~raw["suffix"].isin(_SYNTHETIC_SUFFIXES)
            ].copy()
            raw = raw.loc[
                raw["trade_date"].between(
                    month_start.normalize(), month_end.normalize()
                )
            ]
            raw = self._deduplicate(
                raw, ["trade_datetime", "root", "symbol"]
            )
            if raw.empty:
                continue
            raw["position"] = pd.to_numeric(
                raw["position"], errors="coerce"
            ).clip(lower=0.0)
            raw["volume"] = pd.to_numeric(
                raw["volume"], errors="coerce"
            ).clip(lower=0.0)

            daily = self._read_partitions(
                "daily",
                month_start.normalize() - pd.Timedelta(days=15),
                month_end.normalize(),
                ["symbol", "trade_date", "position", "sequence"],
            )
            daily = self._annotate_symbols(daily)
            daily = daily.loc[
                daily["root"].isin(tickers)
                & ~daily["suffix"].isin(_SYNTHETIC_SUFFIXES)
            ].copy()
            daily = self._deduplicate(
                daily, ["trade_date", "root", "symbol"]
            )
            daily["position"] = pd.to_numeric(
                daily["position"], errors="coerce"
            ).clip(lower=0.0)
            aggregate = self._aggregate_intraday_states(raw, daily)
            if not aggregate.empty:
                if resample_rule:
                    aggregate = self._resample_curve_long(
                        aggregate, resample_rule
                    )
                self._write_curve_month_cache(
                    aggregate,
                    frequency,
                    period,
                    tickers,
                    source_fingerprint,
                )
                frames.append(aggregate)
        if not frames:
            return pd.DataFrame()
        result = pd.concat(frames, ignore_index=True)
        result = self._deduplicate(result, ["trade_datetime", "root"])
        dated = self._assign_intraday_trade_dates(
            result[["trade_datetime"]].copy()
        )
        start_date = pd.Timestamp(start).normalize()
        end_date = pd.Timestamp(end).normalize()
        result = result.loc[
            dated["trade_date"].between(start_date, end_date).to_numpy()
        ]
        return result.sort_values(["trade_datetime", "root"]).reset_index(drop=True)

    def _build_panels(
        self,
        tickers: tuple[str, ...],
        start,
        end,
        frequency: str,
        fields: tuple[str, ...],
    ) -> Dict[str, pd.DataFrame]:
        available_fields = tuple(field for field in fields if field in _RAW_FIELD_MAP)
        curve_fields = tuple(field for field in fields if field in _CURVE_FIELDS)
        panels: Dict[str, pd.DataFrame] = {}
        if available_fields:
            long = self._selected_long(
                tickers, start, end, frequency, available_fields
            )
            for field in available_fields:
                if field not in long:
                    continue
                panel = long.pivot(
                    index="trade_datetime", columns="root", values=field
                )
                panel.index = pd.DatetimeIndex(panel.index)
                panels[field] = panel.sort_index().reindex(columns=tickers)
        if curve_fields:
            if frequency == "daily":
                curve_long = self._daily_curve_long(tickers, start, end)
            else:
                curve_long = self._intraday_curve_long(
                    tickers, start, end, frequency
                )
            for field in curve_fields:
                if field not in curve_long:
                    continue
                panel = curve_long.pivot(
                    index="trade_datetime", columns="root", values=field
                )
                panel.index = pd.DatetimeIndex(panel.index)
                panels[field] = panel.sort_index().reindex(columns=tickers)
        if "oi_change" in fields and "oi" in panels:
            panels["oi_change"] = panels["oi"].diff()
        return panels

    def fetch_price(
        self,
        tickers,
        start,
        end,
        fields: List[str],
    ) -> Dict[str, pd.DataFrame]:
        return self.fetch_price_at_frequency(
            tickers, start, end, fields, frequency="daily"
        )

    def fetch_price_at_frequency(
        self,
        tickers,
        start,
        end,
        fields: List[str],
        frequency: str = "daily",
    ) -> Dict[str, pd.DataFrame]:
        frequency = _normalise_frequency(frequency)
        roots = self._normalise_tickers(tickers)
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        if not roots or start_ts > end_ts:
            return {}

        requested = tuple(dict.fromkeys(str(field) for field in fields))
        load_fields = requested
        if self.eager_fields:
            load_fields = tuple(dict.fromkeys(_DEFAULT_EAGER_FIELDS + requested))
        if any(field in _CURVE_FIELDS for field in requested):
            load_fields = tuple(dict.fromkeys(load_fields + _CURVE_FIELDS))
        if "oi_change" in requested and "oi" not in load_fields:
            load_fields += ("oi",)
        cache_key = (roots, start_ts, end_ts, frequency)
        cached = self._panel_cache.get(cache_key)
        supported_fields = set(_RAW_FIELD_MAP) | set(_CURVE_FIELDS) | {"oi_change"}
        if cached is None:
            cached = self._build_panels(
                roots, start_ts, end_ts, frequency, load_fields
            )
        else:
            missing_fields = tuple(
                field for field in load_fields
                if field in supported_fields and field not in cached
            )
            if "oi_change" in missing_fields and "oi" in cached:
                cached["oi_change"] = cached["oi"].diff()
                missing_fields = tuple(
                    field for field in missing_fields if field != "oi_change"
                )
            if missing_fields:
                cached.update(
                    self._build_panels(
                        roots, start_ts, end_ts, frequency, missing_fields
                    )
                )
        if self.panel_cache_entries:
            self._panel_cache[cache_key] = cached
            self._panel_cache.move_to_end(cache_key)
            while len(self._panel_cache) > self.panel_cache_entries:
                self._panel_cache.popitem(last=False)
        return {
            field: cached[field]
            for field in requested
            if field in cached
        }

    def fetch_fundamental(self, tickers, start, end, fields: List[str]) -> dict:
        return {}

    def fetch_industry(self, tickers, date) -> pd.Series:
        roots = self._normalise_tickers(tickers)
        return pd.Series({root: sector_for(root) for root in roots}, dtype=object)

    def fetch_index_constituents(self, index_code: str, date) -> pd.Index:
        timestamp = pd.Timestamp(date)
        frame = self._read_partitions(
            "daily", timestamp - pd.Timedelta(days=45), timestamp,
            ["symbol", "trade_date"],
        )
        frame = self._annotate_symbols(frame)
        roots = sorted(frame.loc[frame["suffix"].eq("9999"), "root"].unique())
        return pd.Index(roots)

    def fetch_calendar(self, start, end) -> pd.DatetimeIndex:
        # 日历是静态数据: 按 (start,end) 缓存, 避免每次 erc_w/回测重复读全量 1d
        # 并重复 _annotate_symbols 正则 (6000万次 str.extract 热点)
        key = (pd.Timestamp(start), pd.Timestamp(end))
        if key in self._CALENDAR_CACHE:
            return self._CALENDAR_CACHE[key]
        frame = self._read_partitions(
            "daily", start, end, ["symbol", "trade_date"]
        )
        frame = self._annotate_symbols(frame)
        if frame.empty:
            res = pd.DatetimeIndex([])
        else:
            dates = frame.loc[frame["suffix"].eq("9999"), "trade_date"].dropna().unique()
            res = pd.DatetimeIndex(sorted(dates))
        # LRU: 最多缓存 32 个日历 (回测窗口固定, 命中率高)
        if len(self._CALENDAR_CACHE) >= 32:
            self._CALENDAR_CACHE.pop(next(iter(self._CALENDAR_CACHE)))
        self._CALENDAR_CACHE[key] = res
        return res

    def fetch_macro(
        self,
        fields: List[str],
        start=None,
        end=None,
    ) -> pd.DataFrame:
        if self._macro_source is None:
            return pd.DataFrame(columns=list(fields), dtype=float)
        return self._macro_source.fetch_macro(fields, start=start, end=end)


__all__ = ["ParquetFuturesSource"]
