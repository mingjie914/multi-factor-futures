from __future__ import annotations

import numpy as np
import pandas as pd

from strategies.combined import _prepare_erc_returns


def test_combined_erc_excludes_asset_without_history():
    index = pd.bdate_range("2022-01-01", periods=20)
    returns = pd.DataFrame(
        {
            "old_a": np.linspace(-0.01, 0.01, len(index)),
            "future": np.nan,
            "old_b": np.linspace(0.02, -0.02, len(index)),
        },
        index=index,
    )

    clean = _prepare_erc_returns(returns)

    assert list(clean.columns) == ["old_a", "old_b"]
    assert len(clean) == 20
    assert clean.notna().all().all()
