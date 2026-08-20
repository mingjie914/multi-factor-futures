from __future__ import annotations

import pandas as pd
import pytest

from data.manager import DataManager


class MacroSource:
    def __init__(self):
        self.calls = 0

    def fetch_macro(self, fields, start=None, end=None):
        self.calls += 1
        frame = pd.DataFrame(
            {"pmi": [49.5, 50.2], "repo_7d": [1.8, 1.9]},
            index=pd.to_datetime(["2020-01-31", "2020-02-29"]),
        )
        return frame.reindex(columns=fields)


def test_data_manager_macro_cache_is_aligned_and_copy_safe():
    source = MacroSource()
    manager = DataManager(
        source=source,
        cache=None,
        config={"cache": {"enabled": False}},
    )

    first = manager.get_macro(
        ["pmi", "repo_7d"], "2020-01-01", "2020-03-01"
    )
    first.iloc[0, 0] = -999.0
    second = manager.get_macro(
        ["pmi", "repo_7d"], "2020-01-01", "2020-03-01"
    )

    assert source.calls == 1
    assert second.columns.tolist() == ["pmi", "repo_7d"]
    assert isinstance(second.index, pd.DatetimeIndex)
    assert second.iloc[0, 0] == pytest.approx(49.5)
