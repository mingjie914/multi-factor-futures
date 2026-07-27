"""Opt-in adapter from immutable mining snapshots to framework Factors."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from factor_mining.api import CandidateSpec
from factor_mining.features import FeatureEngine
from factor_mining.operators import Expr, ExpressionEvaluator
from factor_mining.repository import load_snapshot
from factor_mining.validation import ValidationConfig, prepare_signal


SNAPSHOT_ENV = "MF_MINED_CANDIDATE_SNAPSHOT"


def _load_panels(candidate: CandidateSpec, data, dates, universe) -> dict[str, pd.DataFrame]:
    panels: dict[str, pd.DataFrame] = {}
    for field in candidate.dependencies:
        try:
            if candidate.frequency != "daily" and hasattr(data, "get_at_frequency"):
                frame = data.get_at_frequency(
                    field, dates, universe, frequency=candidate.frequency
                )
            else:
                frame = data.get(field, dates, universe)
        except (KeyError, NotImplementedError, ValueError):
            frame = pd.DataFrame(np.nan, index=dates, columns=universe)
        panels[field] = frame.reindex(index=dates, columns=universe)
    return panels


def compute_symbolic_candidate(
    candidate: CandidateSpec, data, dates, universe
) -> pd.DataFrame:
    if candidate.kind != "symbolic":
        raise TypeError("the first framework bridge supports symbolic candidates only")
    panels = _load_panels(candidate, data, dates, universe)
    if "close" not in panels or panels["close"].isna().all().all():
        return pd.DataFrame(np.nan, index=dates, columns=universe)
    for dependency in candidate.dependencies:
        if dependency not in panels or panels[dependency].isna().all().all():
            return pd.DataFrame(np.nan, index=dates, columns=universe)

    expression = Expr.from_dict(candidate.payload["expression"])
    if expression.sha256 != candidate.payload.get("expression_sha256"):
        raise ValueError(f"candidate expression hash mismatch: {candidate.candidate_id}")
    postprocess: Mapping = candidate.payload.get("postprocess", {})
    volatility = None
    volatility_name = postprocess.get("volatility_feature")
    required_features = set(expression.terminals())
    if postprocess.get("neutralize_volatility") and volatility_name:
        required_features.add(str(volatility_name))
    try:
        features = FeatureEngine(candidate.feature_config).build(
            panels, required_features=required_features
        )
    except KeyError:
        return pd.DataFrame(np.nan, index=dates, columns=universe)
    raw = ExpressionEvaluator(features).evaluate(expression, copy=False)
    if postprocess.get("neutralize_volatility") and volatility_name:
        volatility = features.values.get(str(volatility_name))
    validation = ValidationConfig(
        # The framework's forward return starts at the factor row.  Shift by
        # both lags so it matches PreparedTarget's delayed entry exactly.
        decision_lag_bars=(
            int(candidate.payload.get("decision_lag_bars", 1))
            + int(candidate.target.entry_delay_bars)
        ),
        mad_clip=float(postprocess.get("mad_clip", 5.0)),
        neutralize_volatility=bool(postprocess.get("neutralize_volatility", False)),
    )
    group_mapping = candidate.payload.get("group_labels") or {}
    group_labels = None
    if group_mapping:
        if any(str(symbol) not in group_mapping for symbol in universe):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        group_labels = [group_mapping[str(symbol)] for symbol in universe]
    signal = prepare_signal(
        candidate.expected_direction * raw,
        validation,
        volatility=volatility,
        group_labels=group_labels,
    )
    return pd.DataFrame(signal, index=features.index, columns=features.symbols).reindex(
        index=dates, columns=universe
    )


def make_factor_class(candidate: CandidateSpec):
    from core.interfaces import Factor

    class SnapshotFactor(Factor):
        name = candidate.framework_name
        category = candidate.category
        frequency = candidate.frequency
        description = (
            f"Auto-mined symbolic factor {candidate.candidate_id}; "
            f"snapshot content {candidate.calculated_hash()[:12]}"
        )

        def dependencies(self) -> list[str]:
            return list(candidate.dependencies)

        def compute(self, data, dates, universe):
            return compute_symbolic_candidate(candidate, data, dates, universe)

    SnapshotFactor.__name__ = "MinedFactor_" + "".join(
        character if character.isalnum() else "_"
        for character in candidate.framework_name
    )
    SnapshotFactor.__qualname__ = SnapshotFactor.__name__
    return SnapshotFactor


def register_snapshot(path: str | Path) -> tuple[str, ...]:
    from factors.user import register_user_factor

    names: list[str] = []
    for candidate in load_snapshot(path):
        if candidate.kind != "symbolic":
            raise TypeError(
                f"candidate {candidate.candidate_id} is not supported by the symbolic bridge"
            )
        factor_class = make_factor_class(candidate)
        register_user_factor(
            candidate.framework_name, category=candidate.category
        )(factor_class)
        names.append(candidate.framework_name)
    return tuple(names)


def register_snapshot_from_environment() -> tuple[str, ...]:
    path = os.environ.get(SNAPSHOT_ENV, "").strip()
    if not path:
        return ()
    return register_snapshot(path)
