"""Canonical futures sector taxonomy shared by research and execution."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


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
    "T": "bond", "TF": "bond", "TS": "bond", "TL": "bond",
    # Newer contracts are classified without changing the established sector
    # labels consumed by research artifacts and constraints.
    "UR": "energy", "SH": "energy", "BR": "energy", "PX": "energy",
    "PR": "energy", "BZ": "energy", "PL": "energy", "LG": "energy",
    "LC": "nonferrous", "SI": "nonferrous", "PS": "nonferrous",
    "AO": "nonferrous", "AD": "nonferrous",
    "EC": "other",
}

SECTOR_NAMES = tuple(sorted(set(SECTOR_MAP.values())))

ASSET_CLASS_BY_SECTOR = {
    "stock_index": "stock",
    "bond": "bond",
    "ferrous": "commodity",
    "nonferrous": "commodity",
    "precious": "commodity",
    "energy": "commodity",
    "agri": "commodity",
    "other": "commodity",
}

ASSET_CLASS_NAMES = ("stock", "bond", "commodity")


def sector_for(instrument: str) -> str:
    return SECTOR_MAP.get(str(instrument), "other")


def asset_class_for(instrument: str) -> str:
    """Return the top-level stock, bond, or commodity allocation bucket."""

    return ASSET_CLASS_BY_SECTOR[sector_for(instrument)]


def hierarchy_for(instrument: str) -> tuple[str, str]:
    """Return ``(asset_class, leaf_group)`` for portfolio construction.

    Commodity instruments retain their economic sub-sector. Stock-index and
    government-bond futures use their asset class as the leaf group because
    they do not have a second allocation layer.
    """

    asset_class = asset_class_for(instrument)
    leaf_group = sector_for(instrument) if asset_class == "commodity" else asset_class
    return asset_class, leaf_group


def instruments_in_sectors(
    instruments: Iterable[str], allowed_sectors: Iterable[str]
) -> list[str]:
    allowed = {str(sector) for sector in allowed_sectors}
    return [str(instrument) for instrument in instruments if sector_for(instrument) in allowed]


def sector_matrix(dates, instruments: Iterable[str]) -> pd.DataFrame:
    """Return the canonical point-in-time sector labels as a dense panel."""

    index = pd.DatetimeIndex(dates)
    columns = pd.Index(instruments)
    labels = np.asarray([sector_for(item) for item in columns], dtype=object)
    values = np.broadcast_to(labels, (len(index), len(columns))).copy()
    return pd.DataFrame(values, index=index, columns=columns, dtype=object)


def sector_series(instruments: Iterable[str]) -> pd.Series:
    """Return one static label per instrument without expanding over dates."""

    columns = pd.Index(instruments)
    return pd.Series(
        [sector_for(item) for item in columns], index=columns, dtype=object
    )
