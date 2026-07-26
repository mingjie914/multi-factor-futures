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
        for step in self._steps:
            try:
                result = step.transform(result, context)
            except Exception as e:
                logging.getLogger(__name__).warning(
                    f"Processing step '{step.name}' failed: {e}"
                )
        return result

    def process_batch(
        self, factors: Dict[str, FactorMatrix], context: ProcessingContext
    ) -> Dict[str, FactorMatrix]:
        return {name: self.process(f, context) for name, f in factors.items()}


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
