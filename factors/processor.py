from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from core.interfaces import ProcessingContext, ProcessingStep
from core.registry import get as registry_get, register
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
                if bool(getattr(step, "fail_closed", False)):
                    raise
                logging.getLogger(__name__).warning(
                    f"Processing step '{step.name}' failed: {e}"
                )
        if eligibility is not None:
            result = result.where(eligibility)
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
                if bool(getattr(step, "fail_closed", False)):
                    raise
                logging.getLogger(__name__).warning(
                    "Processing step '%s' failed: %s", step.name, exc
                )
        if eligibility is not None:
            result = result.where(eligibility)
        return result


def build_processing_steps(configs: List[dict]) -> List[ProcessingStep]:
    """从配置构建处理步骤."""
    steps = []
    for cfg in configs:
        step_type = cfg.get("type", "")
        params = cfg.get("params", {})
        try:
            cls = registry_get("processing_step", step_type)
            steps.append(cls(**params))
        except KeyError:
            continue
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
    return ProcessingContext(
        data=data,
        dates=dates,
        universe=universe,
        industry=sector_series(universe),
        eligibility=build_universe_eligibility(
            data, dates, universe, universe_selection_config
        ),
    )
