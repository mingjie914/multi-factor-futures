from __future__ import annotations

import pandas as pd

from workflows.walkforward import _build_rolling_folds


def test_short_walkforward_calendar_retries_without_name_error():
    calendar = pd.bdate_range("2024-01-01", periods=12)

    folds = _build_rolling_folds(
        calendar, train_bars=5, test_bars=4, step_bars=4
    )

    assert len(folds) == 2
