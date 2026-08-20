from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from core.interfaces import ProcessingContext, ProcessingStep
from core.registry import get as registry_get
from core.types import FactorMatrix


class FactorProcessor:
    """链式调用多个 ProcessingStep."""

    def __init__(self, steps: List[ProcessingStep]):
        self._steps = steps

    def process(
        self, factor: FactorMatrix, context: ProcessingContext
    ) -> FactorMatrix:
        result = factor.copy()
        eligibility = None
        if context.eligibility is not None:
            eligibility = context.eligibility.reindex(
                index=result.index, columns=result.columns, fill_value=False
            )
            result = result.where(eligibility)
        for step in self._steps:
            try:
                result = step.transform(result, context)
            except Exception as e:
                raise RuntimeError(
                    f"processing step {step.name!r} failed: {e}"
                ) from e
            self._validate_result(result, factor, step.name)
        if eligibility is not None:
            result = result.where(eligibility)
        self._validate_result(result, factor, "final_output")
        return result

    def process_batch(
        self, factors: Dict[str, FactorMatrix], context: ProcessingContext
    ) -> Dict[str, FactorMatrix]:
        return {name: self.process(f, context) for name, f in factors.items()}

    def process_excluding(
        self,
        factor: FactorMatrix,
        context: ProcessingContext,
        excluded_steps: set[str],
    ) -> FactorMatrix:
        """Run the declared pipeline while omitting predeclared step types."""
        result = factor.copy()
        eligibility = None
        if context.eligibility is not None:
            eligibility = context.eligibility.reindex(
                index=result.index, columns=result.columns, fill_value=False
            )
            result = result.where(eligibility)
        for step in self._steps:
            if str(getattr(step, "name", "")) in excluded_steps:
                continue
            try:
                result = step.transform(result, context)
            except Exception as exc:
                raise RuntimeError(
                    f"processing step {step.name!r} failed: {exc}"
                ) from exc
            self._validate_result(result, factor, step.name)
        if eligibility is not None:
            result = result.where(eligibility)
        self._validate_result(result, factor, "final_output")
        return result

    @staticmethod
    def _validate_result(
        result: FactorMatrix,
        original: FactorMatrix,
        step_name: str,
    ) -> None:
        if not isinstance(result, pd.DataFrame):
            raise TypeError(
                f"processing step {step_name!r} must return a DataFrame"
            )
        if not result.index.equals(original.index) or not result.columns.equals(
            original.columns
        ):
            raise ValueError(
                f"processing step {step_name!r} changed factor axes"
            )
        try:
            values = result.to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"processing step {step_name!r} returned non-numeric values"
            ) from exc
        if np.isinf(values).any():
            raise ValueError(
                f"processing step {step_name!r} returned infinite values"
            )
        if not np.isfinite(values).any():
            raise ValueError(
                f"processing step {step_name!r} returned no finite values"
            )


def build_processing_steps(configs: List[dict]) -> List[ProcessingStep]:
    """从配置构建处理步骤."""
    steps = []
    for cfg in configs:
        step_type = cfg.get("type", "")
        params = cfg.get("params", {})
        if not step_type:
            raise ValueError("processing step type must not be empty")
        cls = registry_get("processing_step", step_type)
        steps.append(cls(**params))
    return steps


def build_processing_context(
    data,
    dates,
    universe,
    universe_selection_config=None,
) -> ProcessingContext:
    """Build the shared sector labels and optional point-in-time universe mask."""

    from core.sectors import sector_series
    from data.universe_selection import build_universe_eligibility

    dates = pd.DatetimeIndex(dates)
    universe = pd.Index(universe)
    eligibility = build_universe_eligibility(
        data, dates, universe, universe_selection_config
    )
    # Snapshot factors contain cross-sectional operators and candidate-level
    # neutralization before the generic processor runs.  Publish the exact
    # point-in-time mask on this research-local provider so the bridge can use
    # the same eligible cross-section during expression evaluation.
    setattr(data, "_factor_eligibility", eligibility)
    return ProcessingContext(
        data=data,
        dates=dates,
        universe=universe,
        industry=sector_series(universe),
        eligibility=eligibility,
    )
