"""Canonical futures contract symbols and fail-closed alias handling."""
from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd
import polars as pl


_CZCE_EXCHANGES = {"CZC", "CZCE", "ZCE"}
_ROOT_SUFFIX_RE = re.compile(r"^([A-Z]+)(\d+)$")

CONTRACT_SYMBOL_SEMANTICS_VERSION = 2

MARKET_FIELDS = (
    "open", "high", "low", "close", "volume", "amount", "position",
    "settle_price", "pre_settle_price",
)


class ContractAliasConflictError(ValueError):
    """Raised when aliases for one contract disagree at the same timestamp."""


def contract_symbol_parts(symbols: pd.Series) -> pd.DataFrame:
    """Parse exact roots and flag real ``ROOT+YYMM`` delivery contracts.

    Plain roots, two-digit vendor aliases and continuous/index suffixes are
    retained as parseable symbols but never marked as concrete contracts.
    """
    normalized = symbols.astype(str).str.strip().str.upper()
    parsed = normalized.str.extract(_ROOT_SUFFIX_RE)
    suffix = parsed[1]
    concrete_suffix = suffix.where(suffix.str.fullmatch(r"\d{4}", na=False))
    delivery_year = pd.to_numeric(
        concrete_suffix.str[:2], errors="coerce"
    ).add(2000)
    delivery_month = pd.to_numeric(
        concrete_suffix.str[2:], errors="coerce"
    )
    is_concrete = (
        parsed[0].notna()
        & concrete_suffix.notna()
        & delivery_month.between(1, 12)
    )
    return pd.DataFrame(
        {
            "symbol": normalized,
            "root": parsed[0].str.upper(),
            "suffix": suffix,
            "delivery_year": delivery_year.where(is_concrete),
            "delivery_month": delivery_month.where(is_concrete),
            "is_concrete": is_concrete,
        },
        index=symbols.index,
    )


def contract_symbol_parts_polars(symbols: pl.Series) -> pl.DataFrame:
    """Polars equivalent of :func:`contract_symbol_parts`."""
    name = symbols.name or "symbol"
    frame = pl.DataFrame({name: symbols}).with_columns(
        pl.col(name).cast(pl.String).str.strip_chars().str.to_uppercase().alias("symbol")
    )
    root = pl.col("symbol").str.extract(r"^([A-Z]+)(\d+)$", 1)
    suffix = pl.col("symbol").str.extract(r"^([A-Z]+)(\d+)$", 2)
    concrete = suffix.str.contains(r"^\d{4}$")
    year = suffix.str.slice(0, 2).cast(pl.Int32, strict=False) + 2000
    month = suffix.str.slice(2, 2).cast(pl.Int32, strict=False)
    valid = root.is_not_null() & concrete.fill_null(False) & month.is_between(1, 12)
    return frame.select(
        "symbol",
        root.alias("root"),
        suffix.alias("suffix"),
        pl.when(valid).then(year).otherwise(None).alias("delivery_year"),
        pl.when(valid).then(month).otherwise(None).alias("delivery_month"),
        valid.alias("is_concrete"),
    )


def canonicalize_contract_aliases_polars(
    frame: pl.DataFrame,
    *,
    key_columns: Iterable[str] | None = None,
    market_fields: Iterable[str] = MARKET_FIELDS,
) -> pl.DataFrame:
    """Canonicalize aliases with the same fail-closed rules in Polars."""
    if frame.is_empty() or "symbol" not in frame.columns:
        return frame
    missing = sorted({"exchange", "trade_date"} - set(frame.columns))
    if missing:
        raise ValueError(f"contract canonicalization requires columns: {missing}")

    result = frame.with_columns(
        pl.col("symbol").cast(pl.String).str.strip_chars().str.to_uppercase(),
        pl.col("exchange").cast(pl.String).str.strip_chars().str.to_uppercase(),
        pl.col("trade_date").cast(pl.Date),
    ).with_columns(
        pl.col("symbol").alias("__original_symbol"),
        pl.col("symbol").str.extract(r"^([A-Z]+)(\d)(\d{2})$", 1).alias("__root3"),
        pl.col("symbol").str.extract(r"^([A-Z]+)(\d)(\d{2})$", 2).alias("__year_digit"),
        pl.col("symbol").str.extract(r"^([A-Z]+)(\d)(\d{2})$", 3).alias("__month2"),
    )
    czce = pl.col("exchange").is_in(list(_CZCE_EXCHANGES)) & pl.col("__root3").is_not_null()
    month = pl.col("__month2").cast(pl.Int32, strict=False)
    invalid = result.filter(czce & ~month.is_between(1, 12))
    if invalid.height:
        sample = invalid["symbol"].unique(maintain_order=True).head(5).to_list()
        raise ValueError(f"invalid CZCE contract month: {sample}")
    trade_year = pl.col("trade_date").dt.year()
    year_digit = pl.col("__year_digit").cast(pl.Int32, strict=False)
    delivery_year = (trade_year // 10) * 10 + year_digit
    delivery_year = pl.when(delivery_year < trade_year).then(
        delivery_year + 10
    ).otherwise(delivery_year)
    canonical = pl.concat_str(
        pl.col("__root3"),
        (delivery_year % 100).cast(pl.String).str.pad_start(2, "0"),
        month.cast(pl.String).str.pad_start(2, "0"),
    )
    result = result.with_columns(
        pl.when(czce).then(canonical).otherwise(pl.col("symbol")).alias("symbol")
    )

    if key_columns is None:
        time_column = "trade_datetime" if "trade_datetime" in result.columns else "trade_date"
        keys = ["exchange", "symbol", time_column]
    else:
        keys = list(key_columns)
    missing_keys = [column for column in keys if column not in result.columns]
    if missing_keys:
        raise ValueError(f"contract alias keys missing: {missing_keys}")
    fields = [column for column in market_fields if column in result.columns]
    aggregates = [pl.len().alias("__rows")]
    aggregates.extend(pl.col(field).n_unique().alias(f"__n_{field}") for field in fields)
    groups = result.group_by(keys, maintain_order=True).agg(aggregates)
    conflict_expr = pl.any_horizontal(
        [pl.col(f"__n_{field}") > 1 for field in fields]
    ) if fields else pl.lit(False)
    conflicts = groups.filter((pl.col("__rows") > 1) & conflict_expr)
    if conflicts.height:
        raise ContractAliasConflictError(
            "conflicting contract aliases after CZCE canonicalization: "
            f"groups={conflicts.height}, samples={conflicts.select(keys).head(5).to_dicts()}"
        )
    return (
        result.unique(subset=keys, keep="last", maintain_order=True)
        .drop("__original_symbol", "__root3", "__year_digit", "__month2")
    )
def _canonical_symbols(frame: pd.DataFrame) -> pd.Series:
    symbols = frame["symbol"].astype(str).str.strip().str.upper()
    exchanges = frame["exchange"].astype(str).str.strip().str.upper()
    parsed = symbols.str.extract(r"^([A-Z]+)(\d)(\d{2})$")
    months = pd.to_numeric(parsed[2], errors="coerce")
    mask = exchanges.isin(_CZCE_EXCHANGES) & parsed[0].notna()
    invalid = mask & ~months.between(1, 12)
    if invalid.any():
        sample = symbols.loc[invalid].drop_duplicates().head(5).tolist()
        raise ValueError(f"invalid CZCE contract month: {sample}")
    if not mask.any():
        return symbols

    trade_year = pd.to_datetime(frame["trade_date"], errors="raise").dt.year
    year_digit = pd.to_numeric(parsed[1], errors="coerce").fillna(0).astype(int)
    delivery_year = trade_year.floordiv(10).mul(10).add(year_digit)
    delivery_year = delivery_year.add(delivery_year.lt(trade_year).astype(int).mul(10))
    canonical = (
        parsed[0]
        + delivery_year.mod(100).astype(int).astype(str).str.zfill(2)
        + months.fillna(0).astype(int).astype(str).str.zfill(2)
    )
    symbols.loc[mask] = canonical.loc[mask]
    return symbols


def canonicalize_contract_aliases(
    frame: pd.DataFrame,
    *,
    key_columns: Iterable[str] | None = None,
    market_fields: Iterable[str] = MARKET_FIELDS,
) -> pd.DataFrame:
    """Canonicalize contract aliases and remove only identical market rows.

    Duplicate canonical keys are allowed only when every available market
    field is exactly equal (with null equal to null). Conflicting rows raise
    :class:`ContractAliasConflictError`; no row is selected heuristically.
    """
    if frame.empty or "symbol" not in frame:
        return frame
    required = {"exchange", "trade_date"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"contract canonicalization requires columns: {missing}")

    result = frame.copy()
    original_symbols = result["symbol"].astype(str).str.strip().str.upper()
    result["symbol"] = _canonical_symbols(result)

    if key_columns is None:
        time_column = "trade_datetime" if "trade_datetime" in result else "trade_date"
        keys = ["exchange", "symbol", time_column]
    else:
        keys = list(key_columns)
    missing_keys = [column for column in keys if column not in result]
    if missing_keys:
        raise ValueError(f"contract alias keys missing: {missing_keys}")

    duplicate_mask = result.duplicated(keys, keep=False)
    if not duplicate_mask.any():
        return result
    duplicate_rows = result.loc[duplicate_mask].copy()
    fields = [column for column in market_fields if column in duplicate_rows]
    grouped = duplicate_rows.groupby(keys, dropna=False, sort=False)
    conflicting_keys = None
    differing_fields: dict[tuple, list[str]] = {}
    for field in fields:
        different = grouped[field].nunique(dropna=False).gt(1)
        if different.any():
            bad = different.index[different]
            conflicting_keys = bad if conflicting_keys is None else conflicting_keys.union(bad)
            for key in bad:
                normalized_key = key if isinstance(key, tuple) else (key,)
                differing_fields.setdefault(normalized_key, []).append(field)
    if conflicting_keys is not None and len(conflicting_keys):
        samples = []
        for key in list(conflicting_keys)[:5]:
            normalized_key = key if isinstance(key, tuple) else (key,)
            mask = np.ones(len(duplicate_rows), dtype=bool)
            for column, value in zip(keys, normalized_key):
                mask &= duplicate_rows[column].eq(value).to_numpy()
            aliases = original_symbols.loc[duplicate_rows.index[mask]].drop_duplicates().tolist()
            samples.append({
                **dict(zip(keys, normalized_key)),
                "aliases": aliases,
                "differing_fields": differing_fields.get(normalized_key, []),
            })
        raise ContractAliasConflictError(
            "conflicting contract aliases after CZCE canonicalization: "
            f"groups={len(conflicting_keys)}, samples={samples}"
        )

    return result.drop_duplicates(keys, keep="last").reset_index(drop=True)


__all__ = [
    "CONTRACT_SYMBOL_SEMANTICS_VERSION",
    "ContractAliasConflictError",
    "MARKET_FIELDS",
    "canonicalize_contract_aliases",
    "canonicalize_contract_aliases_polars",
    "contract_symbol_parts",
    "contract_symbol_parts_polars",
]
