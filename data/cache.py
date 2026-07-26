from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Optional, Union

import pandas as pd


class Cache:
    """本地 parquet 缓存. 基于 (market, source, field, ticker_hash, date_range) 作 key.

    Args:
        cache_dir: 缓存目录路径.
        backend: 存储后端, 当前仅支持 'parquet'.
    """

    def __init__(self, cache_dir: str = "./cache", backend: str = "parquet") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.backend = backend  # 'parquet' | 'feather'
        self._parquet_metadata = {}

    def _inspect_parquet(self, path: Path):
        """Return columns and index bounds without loading the data columns."""
        stat = path.stat()
        cache_key = (str(path), stat.st_mtime_ns, stat.st_size)
        cached = self._parquet_metadata.get(cache_key)
        if cached is not None:
            return cached

        try:
            import pyarrow.parquet as pq

            parquet = pq.ParquetFile(path)
            metadata = parquet.metadata
            pandas_metadata = json.loads(
                metadata.metadata[b"pandas"].decode("utf-8")
            )
            index_fields = [
                item for item in pandas_metadata.get("index_columns", [])
                if isinstance(item, str)
            ]
            if not index_fields:
                raise ValueError("parquet cache has no materialized index")
            index_field = index_fields[0]
            columns = {
                str(item.get("name"))
                for item in pandas_metadata.get("columns", [])
                if item.get("field_name") not in index_fields
                and item.get("name") is not None
            }

            minimum = None
            maximum = None
            for row_group_index in range(metadata.num_row_groups):
                row_group = metadata.row_group(row_group_index)
                for column_index in range(row_group.num_columns):
                    column = row_group.column(column_index)
                    if column.path_in_schema != index_field:
                        continue
                    stats = column.statistics
                    if stats is None or not stats.has_min_max:
                        raise ValueError("parquet index statistics are unavailable")
                    group_min = pd.Timestamp(stats.min)
                    group_max = pd.Timestamp(stats.max)
                    minimum = group_min if minimum is None else min(minimum, group_min)
                    maximum = group_max if maximum is None else max(maximum, group_max)
                    break
            if minimum is None or maximum is None:
                raise ValueError("parquet index bounds are unavailable")
            result = (columns, minimum, maximum, int(metadata.num_rows))
        except Exception:
            frame = pd.read_parquet(path)
            if frame.empty or not isinstance(frame.index, pd.DatetimeIndex):
                result = (set(), None, None, 0)
            else:
                result = (
                    {str(column) for column in frame.columns},
                    pd.Timestamp(frame.index.min()),
                    pd.Timestamp(frame.index.max()),
                    len(frame),
                )
        if len(self._parquet_metadata) >= 4096:
            self._parquet_metadata.clear()
        self._parquet_metadata[cache_key] = result
        return result

    def _key(
        self,
        market: str,
        source: str,
        field: str,
        tickers,
        start,
        end,
    ) -> Path:
        """生成缓存文件路径 (ticker 全集哈希，避免文件名因顺序/数量爆炸)."""
        if isinstance(tickers, (list, pd.Index)):
            raw = "_".join(sorted(str(t) for t in tickers))
        else:
            raw = str(tickers)
        h = hashlib.md5(f"{market}_{source}_{field}_{raw}_{start}_{end}".encode()).hexdigest()[:16]
        return self.cache_dir / f"{market}_{source}_{field}_{h}.parquet"

    def get(
        self,
        market: str,
        source: str,
        field: str,
        tickers,
        start,
        end,
    ) -> Optional[pd.DataFrame]:
        """返回缓存的 DataFrame，若无缓存返回 None."""
        path = self._key(market, source, field, tickers, start, end)
        if path.exists():
            try:
                return pd.read_parquet(path)
            except Exception:
                return None

        # Reuse a cache built for a wider date range or ticker superset. This
        # keeps offline research usable when callers request the same data with
        # a different warm-up window while preserving source/field isolation.
        requested = {str(ticker) for ticker in tickers}
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        pattern = f"{market}_{source}_{field}_*.parquet"
        covering_path = None
        covering_span = None
        for candidate in self.cache_dir.glob(pattern):
            try:
                columns, minimum, maximum, span = self._inspect_parquet(candidate)
            except Exception:
                continue
            if minimum is None or maximum is None:
                continue
            if not requested.issubset(columns):
                continue
            if minimum > start_ts or maximum < end_ts:
                continue
            if covering_path is None or span < covering_span:
                covering_path = candidate
                covering_span = span

        if covering_path is not None:
            covering = pd.read_parquet(covering_path)
            sliced = covering.loc[start_ts:end_ts].reindex(columns=list(tickers))
            try:
                self.put(market, source, field, tickers, start, end, sliced)
            except Exception:
                pass
            return sliced
        return None

    def put(
        self,
        market: str,
        source: str,
        field: str,
        tickers,
        start,
        end,
        df: pd.DataFrame,
    ) -> None:
        """写入缓存."""
        path = self._key(market, source, field, tickers, start, end)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=True)

    def clear(self, pattern: str = "*") -> None:
        """清除缓存的某些部分."""
        for f in self.cache_dir.glob(pattern):
            f.unlink()

    @property
    def size(self) -> int:
        """缓存总大小 (bytes)."""
        return sum(f.stat().st_size for f in self.cache_dir.glob("*.parquet"))
