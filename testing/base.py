from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from core.types import FactorMatrix, ReturnMatrix, UniverseSchedule


class TestResult(ABC):
    """Abstract result of a factor test evaluation."""

    @abstractmethod
    def to_dict(self) -> dict:
        ...

    @abstractmethod
    def summary(self) -> str:
        ...

    def plot(self, save_path: Optional[str] = None):
        """Optional plotting implementation."""
        pass


class FactorTest(ABC):
    """Abstract base for factor test harnesses."""

    name: str = ""

    @abstractmethod
    def run(
        self,
        factor: FactorMatrix,
        forward_returns: ReturnMatrix,
        universe: UniverseSchedule,
        **params,
    ) -> TestResult:
        ...
