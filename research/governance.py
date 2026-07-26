"""Economic-family metadata and training-only candidate governance."""
from __future__ import annotations

from collections import Counter
from typing import Iterable, Mapping, Sequence


_CATEGORY_TO_FAMILY = {
    "momentum": "trend",
    "directional": "trend",
    "reversal": "reversal",
    "oscillator": "reversal",
    "sentiment": "sentiment",
    "term_structure": "carry",
    "basis": "carry",
    "volatility": "risk_distribution",
    "drawdown": "risk_distribution",
    "skewness": "risk_distribution",
    "liquidity": "liquidity_flow",
    "volume_oi": "liquidity_flow",
    "volume_price": "liquidity_flow",
    "volume_stat": "liquidity_flow",
    "intraday": "intraday",
    "intraday_proxy": "intraday",
    "intraday_specs": "intraday",
    "cross_frequency": "intraday",
    "cross_commodity": "cross_commodity",
    "technicals": "technical_pattern",
    "pattern": "technical_pattern",
}


def factor_family(factor_name: str, explicit_map: Mapping[str, str] | None = None) -> str:
    """Resolve a stable economic family from explicit or registry metadata."""
    name = str(factor_name)
    if explicit_map and name in explicit_map:
        return str(explicit_map[name])
    category = ""
    try:
        from core.registry import get

        category = str(get("factor", name).category or "")
    except (KeyError, AttributeError):
        category = ""
    if category:
        return _CATEGORY_TO_FAMILY.get(category, category)

    lowered = name.lower()
    fallback_prefixes = (
        (("carry", "basis", "roll_yield", "term_structure"), "carry"),
        (("mom", "trend", "breakout", "sma_slope", "ema_gap"), "trend"),
        (("vol", "atr", "drawdown", "skew"), "risk_distribution"),
        (("volume", "oi_", "dollar_volume", "amihud"), "liquidity_flow"),
        (("intraday", "overnight", "tail_", "vwap"), "intraday"),
        (("reversal", "rsi", "oscillator"), "reversal"),
    )
    for prefixes, family in fallback_prefixes:
        if lowered.startswith(prefixes):
            return family
    return "other"


def select_candidates_by_family(
    candidates: Sequence[Mapping],
    *,
    default_cap: int,
    family_caps: Mapping[str, int] | None = None,
    explicit_map: Mapping[str, str] | None = None,
    score_key: str = "best_t",
) -> tuple[list[dict], dict]:
    """Apply deterministic family caps to an already training-approved list."""
    if default_cap < 1:
        raise ValueError("default family cap must be at least one")
    caps = {str(key): int(value) for key, value in (family_caps or {}).items()}
    ranked = sorted(
        (dict(candidate) for candidate in candidates),
        key=lambda row: (-abs(float(row.get(score_key, 0.0))), str(row.get("name", row.get("factor", "")))),
    )
    selected: list[dict] = []
    rejected: list[dict] = []
    counts: Counter[str] = Counter()
    for row in ranked:
        name = str(row.get("name", row.get("factor", "")))
        family = factor_family(name, explicit_map)
        row["economic_family"] = family
        cap = caps.get(family, default_cap)
        if cap < 1 or counts[family] >= cap:
            rejected.append({"name": name, "economic_family": family, "reason": "family_cap"})
            continue
        counts[family] += 1
        selected.append(row)
    audit = {
        "input_count": len(ranked),
        "selected_count": len(selected),
        "selected_by_family": dict(sorted(counts.items())),
        "default_cap": int(default_cap),
        "family_caps": dict(sorted(caps.items())),
        "rejected": rejected,
    }
    return selected, audit
