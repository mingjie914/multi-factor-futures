from __future__ import annotations

import pandas as pd
import pytest

from data.manager import DataManager
from data.mysql_source import MySQLSource


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


def test_mysql_macro_query_uses_configured_aliases_and_month_dates(monkeypatch):
    source = MySQLSource({
        "tables": {
            "macrodata": {
                "table_name": "macrodata",
                "columns": {
                    "date": "date",
                    "pmi": "PMI",
                    "repo_7d": "银行间质押式回购加权利率7天月",
                },
            }
        }
    })
    captured = {}

    def fake_read_sql(sql):
        captured["sql"] = sql
        return pd.DataFrame({
            "observation_date": ["2020-01-31", "2020-02-29"],
            "pmi": ["49.5", "50.2"],
            "repo_7d": ["1.8", "1.9"],
        })

    monkeypatch.setattr(source, "_read_sql", fake_read_sql)
    result = source.fetch_macro(
        ["pmi", "repo_7d"], "2020-01-01", "2020-03-31"
    )

    assert result.columns.tolist() == ["pmi", "repo_7d"]
    assert result.index.equals(
        pd.DatetimeIndex(["2020-01-31", "2020-02-29"], name="observation_date")
    )
    assert result.dtypes.tolist() == [float, float]
    assert "STR_TO_DATE" in captured["sql"]
    assert "`PMI` AS `pmi`" in captured["sql"]
    assert "`macrodata`" in captured["sql"]


def test_mysql_macro_rejects_fields_outside_the_configured_whitelist():
    source = MySQLSource({
        "tables": {
            "macrodata": {
                "table_name": "macrodata",
                "columns": {"date": "date", "pmi": "PMI"},
            }
        }
    })

    with pytest.raises(ValueError, match="unconfigured macro fields"):
        source.fetch_macro(["pmi", "not_allowed"])

