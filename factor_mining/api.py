"""Stable public contracts shared by mining, storage, and framework adapters."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib
import json
from typing import Any, Dict, Tuple


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


class RunMode(str, Enum):
    DEV = "dev"
    MINE = "mine"
    SCREEN = "screen"
    OOS = "oos"


@dataclass(frozen=True)
class TargetSpec:
    """Availability-aware return label in bars at ``decision_frequency``."""

    name: str
    decision_frequency: str = "1min"
    horizon_bars: int = 15
    entry_delay_bars: int = 1
    entry_price: str = "close"
    session_policy: str = "allow_cross_session"
    cost_bps: float = 0.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("target name is required")
        if self.horizon_bars < 1:
            raise ValueError("horizon_bars must be positive")
        if self.entry_delay_bars < 1:
            raise ValueError("entry_delay_bars must be at least one")
        if self.entry_price != "close":
            raise ValueError("the first release supports close-to-close labels only")
        if self.cost_bps < 0:
            raise ValueError("cost_bps cannot be negative")
        if self.session_policy != "allow_cross_session":
            raise ValueError(
                "the first release supports allow_cross_session targets only"
            )

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "TargetSpec":
        return cls(**value)


@dataclass(frozen=True)
class FeatureConfig:
    """Feature vocabulary. Windows are bars, not calendar days."""

    source_frequency: str = "1min"
    decision_frequency: str = "1min"
    feature_horizons: Tuple[int, ...] = (1, 2, 3, 5, 10, 15, 30, 60, 120, 240)
    lag_steps: Tuple[int, ...] = (1, 2, 3, 5, 10, 15, 30, 60)
    rolling_windows: Tuple[int, ...] = (3, 5, 10, 15, 30, 60, 120, 240)
    raw_fields: Tuple[str, ...] = (
        "open", "high", "low", "close", "volume", "amount", "oi",
        "oi_change",
    )
    include_technicals: bool = True
    include_distribution: bool = True
    dtype: str = "float32"
    max_feature_memory_mb: int = 4096

    def __post_init__(self) -> None:
        for name, values in (
            ("feature_horizons", self.feature_horizons),
            ("lag_steps", self.lag_steps),
            ("rolling_windows", self.rolling_windows),
        ):
            if not values or any(int(value) < 1 for value in values):
                raise ValueError(f"{name} must contain positive bar counts")
        if 1 not in self.feature_horizons:
            raise ValueError("feature_horizons must include the one-bar feature")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be float32 or float64")
        if self.max_feature_memory_mb < 64:
            raise ValueError("max_feature_memory_mb must be at least 64")

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "FeatureConfig":
        converted = dict(value)
        for key in ("feature_horizons", "lag_steps", "rolling_windows", "raw_fields"):
            if key in converted:
                converted[key] = tuple(converted[key])
        return cls(**converted)


@dataclass(frozen=True)
class CandidateSpec:
    """Content-addressed candidate stored outside the runtime factor registry."""

    candidate_id: str
    framework_name: str
    kind: str
    category: str
    frequency: str
    target: TargetSpec
    dependencies: Tuple[str, ...]
    lookback_bars: int
    payload: Dict[str, Any]
    feature_config: FeatureConfig
    metrics: Dict[str, Any] = field(default_factory=dict)
    lineage: Dict[str, Any] = field(default_factory=dict)
    status: str = "mined_candidate"
    expected_direction: int = 1
    content_sha256: str = ""

    def __post_init__(self) -> None:
        if self.kind not in {"symbolic", "model"}:
            raise ValueError("candidate kind must be symbolic or model")
        if self.expected_direction not in {-1, 1}:
            raise ValueError("expected_direction must be -1 or 1")
        if self.lookback_bars < 0:
            raise ValueError("lookback_bars cannot be negative")
        if not self.dependencies or len(self.dependencies) != len(set(self.dependencies)):
            raise ValueError("candidate dependencies must be non-empty and unique")
        if self.frequency != self.feature_config.decision_frequency:
            raise ValueError("candidate and feature frequencies differ")
        if self.frequency != self.target.decision_frequency:
            raise ValueError("candidate and target frequencies differ")

    def hash_payload(self) -> Dict[str, Any]:
        return {
            "framework_name": self.framework_name,
            "kind": self.kind,
            "category": self.category,
            "frequency": self.frequency,
            "target": asdict(self.target),
            "dependencies": list(self.dependencies),
            "lookback_bars": self.lookback_bars,
            "payload": self.payload,
            "feature_config": asdict(self.feature_config),
            "expected_direction": self.expected_direction,
        }

    def calculated_hash(self) -> str:
        return content_hash(self.hash_payload())

    def validated(self) -> "CandidateSpec":
        expected = self.calculated_hash()
        if self.content_sha256 and self.content_sha256 != expected:
            raise ValueError(f"candidate hash mismatch: {self.candidate_id}")
        if not self.candidate_id:
            raise ValueError("candidate_id is required")
        if not self.framework_name:
            raise ValueError("framework_name is required")
        return self

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["content_sha256"] = self.content_sha256 or self.calculated_hash()
        return value

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "CandidateSpec":
        data = dict(value)
        data["target"] = TargetSpec.from_dict(data["target"])
        data["feature_config"] = FeatureConfig.from_dict(data["feature_config"])
        data["dependencies"] = tuple(data.get("dependencies", ()))
        return cls(**data).validated()


@dataclass(frozen=True)
class MiningRunSpec:
    run_id: str
    mode: RunMode
    seed: int
    start: str
    end: str
    universe: Tuple[str, ...]
    target: TargetSpec
    feature_config: FeatureConfig
    engine: str = "gp"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.run_id or not self.universe:
            raise ValueError("run_id and universe are required")
        if self.target.decision_frequency != self.feature_config.decision_frequency:
            raise ValueError("run target and feature frequencies differ")

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["mode"] = self.mode.value
        return value
