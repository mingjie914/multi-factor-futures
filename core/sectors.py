"""Canonical futures sector taxonomy shared by research and execution."""
from __future__ import annotations

import hashlib
import json
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


SECTOR_MAP = {
    # Ferrous industrial chain.
    "RB": "ferrous", "I": "ferrous", "FG": "ferrous", "JM": "ferrous",
    "HC": "ferrous", "J": "ferrous", "SM": "ferrous", "SF": "ferrous",
    # Nonferrous industrial metals and precious metals are separate sectors.
    "CU": "nonferrous", "AL": "nonferrous", "SN": "nonferrous",
    "NI": "nonferrous", "ZN": "nonferrous", "PB": "nonferrous",
    "SS": "nonferrous",
    "AU": "precious", "AG": "precious",
    "SC": "energy", "SA": "energy", "FU": "energy", "V": "energy",
    "TA": "energy", "RU": "energy", "MA": "energy", "EG": "energy",
    "EB": "energy", "PK": "energy", "PP": "energy", "L": "energy",
    "BU": "energy", "PG": "energy",
    "A": "agri", "M": "agri", "Y": "agri", "P": "agri", "C": "agri",
    "CF": "agri", "SR": "agri", "OI": "agri", "RM": "agri",
    "JD": "agri", "AP": "agri", "LH": "agri", "CS": "agri",
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

# The main framework universe is one ordered contract shared by research,
# backtests, monitoring and the production-style strategy.  Broader taxonomy
# entries above remain available for classification but are not separate
# framework universes.
FRAMEWORK_UNIVERSE = (
    "A", "AG", "AL", "AU", "CU", "FU", "HC", "I", "IC", "IF", "IH", "J",
    "JM", "M", "MA", "NI", "P", "RB", "RM", "RU", "SA", "SN", "SR", "T",
    "TA", "TL", "TS", "Y", "ZN", "IM", "TF", "CF", "OI", "LH", "JD", "SC",
    "V", "UR",
)


def require_framework_universe(instruments: Iterable[str]) -> tuple[str, ...]:
    """Fail closed unless an entry point uses the ordered framework contract."""
    normalized = tuple(map(str, instruments))
    if normalized != FRAMEWORK_UNIVERSE:
        raise ValueError("universe must exactly match FRAMEWORK_UNIVERSE")
    return normalized

TAXONOMY_VERSION = "futures_taxonomy_20260728"

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

PORTFOLIO_SELECTION_GROUPS = ("有色", "黑色", "能化", "农产品", "金融")
_PORTFOLIO_GROUP_BY_SECTOR = {
    "nonferrous": "有色",
    "precious": "有色",
    "ferrous": "黑色",
    "energy": "能化",
    "agri": "农产品",
    "stock_index": "金融",
    "bond": "金融",
}


def taxonomy_sha256(sector_map: Mapping[str, str] = SECTOR_MAP) -> str:
    payload = {
        "version": TAXONOMY_VERSION,
        "sector_map": dict(sorted((str(k), str(v)) for k, v in sector_map.items())),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def taxonomy_diff(
    previous: Mapping[str, str],
    current: Mapping[str, str] = SECTOR_MAP,
) -> dict:
    """Return an auditable change set; any change requires a P0 replay."""
    instruments = sorted(set(map(str, previous)) | set(map(str, current)))
    changes = [
        {
            "instrument": instrument,
            "previous": previous.get(instrument),
            "current": current.get(instrument),
        }
        for instrument in instruments
        if previous.get(instrument) != current.get(instrument)
    ]
    return {
        "taxonomy_version": TAXONOMY_VERSION,
        "taxonomy_sha256": taxonomy_sha256(current),
        "changes": changes,
        "requires_full_p0_replay": bool(changes),
    }


def sector_for(instrument: str) -> str:
    return SECTOR_MAP.get(str(instrument), "other")


def portfolio_selection_group_for(instrument: str) -> str:
    """Return the broad bucket used only by production Top/Bottom caps."""
    return _PORTFOLIO_GROUP_BY_SECTOR.get(sector_for(instrument), "其他")


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
