"""Frequency-aware minimum sample rules for discovery and OOS stages."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from core.factor_contract import normalise_frequency


@dataclass(frozen=True)
class SampleAssessment:
    frequency: str
    train_bars: int
    test_bars: int
    train_days: int | None
    test_days: int | None
    minimum_train_bars: int
    minimum_test_bars: int
    minimum_train_days: int
    minimum_test_days: int
    train_test_ratio: float
    train_test_day_ratio: float | None
    minimum_train_test_ratio: float
    sufficient: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "frequency": self.frequency,
            "train_bars": self.train_bars,
            "test_bars": self.test_bars,
            "train_days": self.train_days,
            "test_days": self.test_days,
            "minimum_train_bars": self.minimum_train_bars,
            "minimum_test_bars": self.minimum_test_bars,
            "minimum_train_days": self.minimum_train_days,
            "minimum_test_days": self.minimum_test_days,
            "train_test_ratio": self.train_test_ratio,
            "train_test_day_ratio": self.train_test_day_ratio,
            "minimum_train_test_ratio": self.minimum_train_test_ratio,
            "sufficient": self.sufficient,
            "reasons": list(self.reasons),
        }


def _minimum(mapping: object, original: str, canonical: str, label: str) -> int:
    values = dict(mapping or {})
    key = original if original in values else canonical
    if key not in values:
        raise ValueError(f"sample policy has no {label} for frequency {key!r}")
    value = int(values[key])
    if value < 1:
        raise ValueError(f"sample policy {label}[{key!r}] must be positive")
    return value


def assess_sample_counts(
    train_bars: int,
    test_bars: int,
    *,
    policy,
    frequency: str,
    train_days: int | None = None,
    test_days: int | None = None,
) -> SampleAssessment:
    original_freq = frequency
    frequency = normalise_frequency(frequency)
    minimum_train = _minimum(
        policy.minimum_train_bars_by_frequency, original_freq, frequency, "minimum_train_bars"
    )
    minimum_test = _minimum(
        policy.minimum_test_bars_by_frequency, original_freq, frequency, "minimum_test_bars"
    )
    minimum_train_days = _minimum(
        policy.minimum_train_days_by_frequency, original_freq, frequency, "minimum_train_days"
    )
    minimum_test_days = _minimum(
        policy.minimum_test_days_by_frequency, original_freq, frequency, "minimum_test_days"
    )
    ratio = float(train_bars) / float(test_bars) if test_bars > 0 else 0.0
    day_ratio = (
        float(train_days) / float(test_days)
        if train_days is not None and test_days is not None and test_days > 0
        else None
    )
    minimum_ratio = float(policy.minimum_train_test_ratio)
    reasons = []
    if int(train_bars) < minimum_train:
        reasons.append("insufficient_train_bars")
    if int(test_bars) < minimum_test:
        reasons.append("insufficient_test_bars")
    if train_days is not None and int(train_days) < minimum_train_days:
        reasons.append("insufficient_train_days")
    if test_days is not None and int(test_days) < minimum_test_days:
        reasons.append("insufficient_test_days")
    if ratio < minimum_ratio:
        reasons.append("insufficient_train_test_ratio")
    if day_ratio is not None and day_ratio < minimum_ratio:
        reasons.append("insufficient_train_test_day_ratio")
    return SampleAssessment(
        frequency=frequency,
        train_bars=int(train_bars),
        test_bars=int(test_bars),
        train_days=None if train_days is None else int(train_days),
        test_days=None if test_days is None else int(test_days),
        minimum_train_bars=minimum_train,
        minimum_test_bars=minimum_test,
        minimum_train_days=minimum_train_days,
        minimum_test_days=minimum_test_days,
        train_test_ratio=ratio,
        train_test_day_ratio=day_ratio,
        minimum_train_test_ratio=minimum_ratio,
        sufficient=not reasons,
        reasons=tuple(reasons),
    )


def minimum_training_days(policy, frequency: str) -> int:
    """Convert the training-bar floor to instrument trading days."""
    canonical = normalise_frequency(frequency)
    return _minimum(
        policy.minimum_train_days_by_frequency,
        canonical, canonical,
        "minimum_train_days",
    )


def chronological_split(
    series: pd.Series,
    *,
    policy,
    frequency: str,
) -> tuple[pd.Series, pd.Series, SampleAssessment]:
    """Use the minimum admissible ratio (3:1 by default) for an OOS split."""
    clean = series.dropna().sort_index()
    ratio = float(policy.minimum_train_test_ratio)
    split_fraction = ratio / (ratio + 1.0)
    split_index = int(len(clean) * split_fraction)
    train = clean.iloc[:split_index]
    test = clean.iloc[split_index:]
    assessment = assess_sample_counts(
        len(train), len(test), policy=policy, frequency=frequency,
        train_days=int(pd.DatetimeIndex(train.index).normalize().nunique()),
        test_days=int(pd.DatetimeIndex(test.index).normalize().nunique()),
    )
    return train, test, assessment
