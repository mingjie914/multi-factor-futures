"""Canonical futures sector taxonomy shared by research and execution."""
from __future__ import annotations

from typing import Iterable


SECTOR_MAP = {
    # Ferrous industrial chain.
    "RB": "ferrous", "I": "ferrous", "FG": "ferrous", "JM": "ferrous",
    "HC": "ferrous", "J": "ferrous", "SM": "ferrous", "SF": "ferrous",
    # Nonferrous industrial metals and precious metals are separate sectors.
    "CU": "nonferrous", "AL": "nonferrous", "SN": "nonferrous",
    "NI": "nonferrous", "ZN": "nonferrous",
    "AU": "precious", "AG": "precious",
    "SC": "energy", "SA": "energy", "FU": "energy", "V": "energy",
    "TA": "energy", "RU": "energy", "MA": "energy", "EG": "energy",
    "EB": "energy", "PK": "energy", "PP": "energy", "L": "energy",
    "A": "agri", "M": "agri", "Y": "agri", "P": "agri", "C": "agri",
    "CF": "agri", "SR": "agri", "OI": "agri", "RM": "agri",
    "JD": "agri", "AP": "agri", "LH": "agri",
    # Stock-index and government-bond futures are separate sectors.
    "IF": "stock_index", "IC": "stock_index", "IM": "stock_index",
    "IH": "stock_index",
    "T": "bond", "TL": "bond",
    "EC": "other", "UR": "other", "LC": "other", "PS": "other",
}

SECTOR_NAMES = tuple(sorted(set(SECTOR_MAP.values())))


def sector_for(instrument: str) -> str:
    return SECTOR_MAP.get(str(instrument), "other")


def instruments_in_sectors(
    instruments: Iterable[str], allowed_sectors: Iterable[str]
) -> list[str]:
    allowed = {str(sector) for sector in allowed_sectors}
    return [str(instrument) for instrument in instruments if sector_for(instrument) in allowed]
