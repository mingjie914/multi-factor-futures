from __future__ import annotations

import numpy as np
import pandas as pd

from workflows.experiments.supertrend_parallel import _walk_forward_weights


def test_parallel_min_variance_is_bounded_and_point_in_time():
    dates = pd.date_range("2023-01-02", periods=90, freq="B")
    step = np.arange(90, dtype=float)
    returns = pd.DataFrame(
        {
            "multi_factor": 0.002 * np.sin(step / 3.0),
            "supertrend": 0.001 * np.cos(step / 4.0),
        },
        index=dates,
    )
    base = _walk_forward_weights(returns)
    assert base["supertrend"].between(0.10, 0.30).all()
    assert np.allclose(base.sum(axis=1), 1.0)

    cutoff = dates[60]
    changed = returns.copy()
    changed.loc[changed.index > cutoff, "supertrend"] *= 50.0
    revised = _walk_forward_weights(changed)
    pd.testing.assert_frame_equal(base.loc[:cutoff], revised.loc[:cutoff])
