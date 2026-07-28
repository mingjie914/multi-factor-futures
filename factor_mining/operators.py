"""Typed expression tree and protected vectorized operator runtime."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from factor_mining.api import canonical_json
from factor_mining.features import FeatureSet


NUMERIC = "numeric"
BOOLEAN = "boolean"
_EPS = 1e-8
_CLIP = 1e8


@dataclass(frozen=True)
class OperatorSpec:
    arity: int
    output_type: str = NUMERIC
    input_types: Tuple[str, ...] = ()
    uses_window: bool = False


OPERATOR_SPECS: Dict[str, OperatorSpec] = {
    "add": OperatorSpec(2),
    "sub": OperatorSpec(2),
    "mul": OperatorSpec(2),
    "div": OperatorSpec(2),
    "avg": OperatorSpec(2),
    "max": OperatorSpec(2),
    "min": OperatorSpec(2),
    "neg": OperatorSpec(1),
    "abs": OperatorSpec(1),
    "square": OperatorSpec(1),
    "cube": OperatorSpec(1),
    "signed_sqrt": OperatorSpec(1),
    "signed_log1p": OperatorSpec(1),
    "reciprocal": OperatorSpec(1),
    "delay": OperatorSpec(1, uses_window=True),
    "delta": OperatorSpec(1, uses_window=True),
    "ts_sum": OperatorSpec(1, uses_window=True),
    "ts_mean": OperatorSpec(1, uses_window=True),
    "ts_ema": OperatorSpec(1, uses_window=True),
    "ts_std": OperatorSpec(1, uses_window=True),
    "ts_min": OperatorSpec(1, uses_window=True),
    "ts_max": OperatorSpec(1, uses_window=True),
    "ts_median": OperatorSpec(1, uses_window=True),
    "ts_rank": OperatorSpec(1, uses_window=True),
    "ts_zscore": OperatorSpec(1, uses_window=True),
    "ts_skew": OperatorSpec(1, uses_window=True),
    "ts_kurt": OperatorSpec(1, uses_window=True),
    "ts_corr": OperatorSpec(2, uses_window=True),
    "ts_cov": OperatorSpec(2, uses_window=True),
    "decay_linear": OperatorSpec(1, uses_window=True),
    "cs_rank": OperatorSpec(1),
    "cs_zscore": OperatorSpec(1),
    "cs_demean": OperatorSpec(1),
    "gt": OperatorSpec(2, output_type=BOOLEAN),
    "lt": OperatorSpec(2, output_type=BOOLEAN),
    "logical_and": OperatorSpec(2, output_type=BOOLEAN, input_types=(BOOLEAN, BOOLEAN)),
    "logical_or": OperatorSpec(2, output_type=BOOLEAN, input_types=(BOOLEAN, BOOLEAN)),
    "logical_not": OperatorSpec(1, output_type=BOOLEAN, input_types=(BOOLEAN,)),
    "if_else": OperatorSpec(3, input_types=(BOOLEAN, NUMERIC, NUMERIC)),
}


@dataclass(frozen=True)
class Expr:
    op: str
    args: Tuple["Expr", ...] = ()
    name: str = ""
    value: float = 0.0
    window: int = 0
    output_type: str = NUMERIC

    @classmethod
    def terminal(cls, name: str) -> "Expr":
        if not name:
            raise ValueError("terminal name is required")
        return cls(op="terminal", name=str(name), output_type=NUMERIC)

    @classmethod
    def constant(cls, value: float) -> "Expr":
        if not np.isfinite(value):
            raise ValueError("constant must be finite")
        return cls(op="constant", value=float(value), output_type=NUMERIC)

    @classmethod
    def operation(cls, op: str, *args: "Expr", window: int = 0) -> "Expr":
        if op not in OPERATOR_SPECS:
            raise ValueError(f"unknown operator: {op}")
        spec = OPERATOR_SPECS[op]
        if len(args) != spec.arity:
            raise ValueError(f"operator {op} requires {spec.arity} arguments")
        if spec.uses_window and int(window) < 1:
            raise ValueError(f"operator {op} requires a positive window")
        if not spec.uses_window and window:
            raise ValueError(f"operator {op} does not accept a window")
        expected_inputs = spec.input_types or (NUMERIC,) * spec.arity
        for index, (argument, expected) in enumerate(zip(args, expected_inputs)):
            if argument.output_type != expected:
                raise TypeError(
                    f"operator {op} argument {index} requires {expected}, "
                    f"got {argument.output_type}"
                )
        return cls(
            op=op,
            args=tuple(args),
            window=int(window),
            output_type=spec.output_type,
        )

    def to_dict(self) -> dict:
        value = {"op": self.op, "output_type": self.output_type}
        if self.args:
            value["args"] = [argument.to_dict() for argument in self.args]
        if self.name:
            value["name"] = self.name
        if self.op == "constant":
            value["value"] = self.value
        if self.window:
            value["window"] = self.window
        return value

    @classmethod
    def from_dict(cls, value: Mapping) -> "Expr":
        op = str(value.get("op", ""))
        if op == "terminal":
            return cls.terminal(str(value.get("name", "")))
        if op == "constant":
            return cls.constant(float(value.get("value", 0.0)))
        args = tuple(cls.from_dict(item) for item in value.get("args", ()))
        expr = cls.operation(op, *args, window=int(value.get("window", 0)))
        declared = value.get("output_type")
        if declared and declared != expr.output_type:
            raise ValueError(f"expression output type mismatch for {op}")
        return expr

    @property
    def sha256(self) -> str:
        return hashlib.sha256(canonical_json(self.to_dict()).encode("utf-8")).hexdigest()

    @property
    def complexity(self) -> int:
        return 1 + sum(argument.complexity for argument in self.args)

    @property
    def depth(self) -> int:
        return 1 + max((argument.depth for argument in self.args), default=0)

    @property
    def max_window(self) -> int:
        return max(self.window, *(argument.max_window for argument in self.args), 0)

    def terminals(self) -> tuple[str, ...]:
        names = {self.name} if self.op == "terminal" else set()
        for argument in self.args:
            names.update(argument.terminals())
        return tuple(sorted(names))

    def paths(self, prefix: Tuple[int, ...] = ()) -> tuple[Tuple[int, ...], ...]:
        result = [prefix]
        for index, argument in enumerate(self.args):
            result.extend(argument.paths(prefix + (index,)))
        return tuple(result)

    def subtree(self, path: Sequence[int]) -> "Expr":
        node = self
        for index in path:
            node = node.args[int(index)]
        return node

    def replace(self, path: Sequence[int], replacement: "Expr") -> "Expr":
        if not path:
            if replacement.output_type != self.output_type:
                raise TypeError("replacement changes root output type")
            return replacement
        index = int(path[0])
        args = list(self.args)
        target = args[index]
        args[index] = target.replace(path[1:], replacement)
        return Expr.operation(self.op, *args, window=self.window)


def _sanitize(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    result = np.where(np.isfinite(result), np.clip(result, -_CLIP, _CLIP), np.nan)
    return result.astype(np.float32, copy=False)


def _frame(value: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(np.asarray(value, dtype=float))


def _rolling(value: np.ndarray, window: int, method: str) -> np.ndarray:
    min_periods = max(2, min(window, max(3, window // 2)))
    rolling = _frame(value).rolling(window, min_periods=min_periods)
    if method == "sum":
        return rolling.sum().to_numpy()
    if method == "mean":
        return rolling.mean().to_numpy()
    if method == "std":
        return rolling.std().to_numpy()
    if method == "min":
        return rolling.min().to_numpy()
    if method == "max":
        return rolling.max().to_numpy()
    if method == "median":
        return rolling.median().to_numpy()
    if method == "skew":
        return rolling.skew().to_numpy()
    if method == "kurt":
        return rolling.kurt().to_numpy()
    raise ValueError(method)


def _cross_section_rank(value: np.ndarray) -> np.ndarray:
    return _frame(value).rank(axis=1, method="average", pct=True).to_numpy(dtype=float)


def _decay_linear(value: np.ndarray, window: int) -> np.ndarray:
    """Weighted rolling mean without a Python callback per window."""

    source = np.asarray(value, dtype=float)
    result = np.full(source.shape, np.nan, dtype=float)
    if len(source) < window:
        return result
    weights = np.arange(1, window + 1, dtype=float)
    weights /= weights.sum()
    kernel = weights[::-1]
    count_kernel = np.ones(window, dtype=np.int16)
    for column in range(source.shape[1]):
        values = source[:, column]
        finite = np.isfinite(values)
        weighted = np.convolve(
            np.where(finite, values, 0.0), kernel, mode="valid"
        )
        counts = np.convolve(
            finite.astype(np.int16), count_kernel, mode="valid"
        )
        result[window - 1 :, column] = np.where(
            counts == window, weighted, np.nan
        )
    return result


class ExpressionEvaluator:
    """Evaluate expression trees with subtree memoization and protected closure."""

    def __init__(
        self,
        features: FeatureSet,
        *,
        cache_max_bytes: int = 128 * 1024 * 1024,
        cross_section_mask: np.ndarray | None = None,
    ):
        self.features = features
        self.cache_max_bytes = max(0, int(cache_max_bytes))
        self.cross_section_mask = None
        if cross_section_mask is not None:
            mask = np.asarray(cross_section_mask)
            if mask.shape != features.shape:
                raise ValueError("cross_section_mask shape differs from features")
            if mask.dtype != np.bool_:
                raise TypeError("cross_section_mask must be a boolean array")
            self.cross_section_mask = mask
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._cache_bytes = 0

    def clear(self) -> None:
        self._cache.clear()
        self._cache_bytes = 0

    def evaluate(self, expression: Expr, *, copy: bool = True) -> np.ndarray:
        if expression.output_type != NUMERIC:
            raise TypeError("factor expression root must be numeric")
        result = self._eval(expression)
        return result.copy() if copy else result

    def _remember(self, key: str, value: np.ndarray) -> None:
        size = int(value.nbytes)
        if size > self.cache_max_bytes:
            return
        while self._cache and self._cache_bytes + size > self.cache_max_bytes:
            _, removed = self._cache.popitem(last=False)
            self._cache_bytes -= int(removed.nbytes)
        self._cache[key] = value
        self._cache_bytes += size

    def _eval(self, expression: Expr) -> np.ndarray:
        key = expression.sha256
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached
        if expression.op == "terminal":
            if expression.name not in self.features.values:
                raise KeyError(f"missing terminal feature: {expression.name}")
            return np.asarray(self.features.values[expression.name], dtype=np.float32)
        elif expression.op == "constant":
            result = np.full(self.features.shape, expression.value, dtype=np.float32)
        else:
            args = [self._eval(argument) for argument in expression.args]
            result = self._apply(expression.op, args, expression.window)
        result = _sanitize(result)
        result.setflags(write=False)
        self._remember(key, result)
        return result

    def _apply(self, op: str, args: Sequence[np.ndarray], window: int) -> np.ndarray:
        with np.errstate(all="ignore"):
            if op == "add": return args[0] + args[1]
            if op == "sub": return args[0] - args[1]
            if op == "mul": return args[0] * args[1]
            if op == "div": return _protected_divide(args[0], args[1])
            if op == "avg": return 0.5 * (args[0] + args[1])
            if op == "max": return np.maximum(args[0], args[1])
            if op == "min": return np.minimum(args[0], args[1])
            if op == "neg": return -args[0]
            if op == "abs": return np.abs(args[0])
            if op == "square": return np.square(np.clip(args[0], -1e4, 1e4))
            if op == "cube": return np.power(np.clip(args[0], -400.0, 400.0), 3)
            if op == "signed_sqrt": return np.sign(args[0]) * np.sqrt(np.abs(args[0]))
            if op == "signed_log1p": return np.sign(args[0]) * np.log1p(np.abs(args[0]))
            if op == "reciprocal": return _protected_divide(np.ones_like(args[0]), args[0])
            if op == "delay": return _frame(args[0]).shift(window).to_numpy()
            if op == "delta": return _frame(args[0]).diff(window).to_numpy()
            if op == "ts_sum": return _rolling(args[0], window, "sum")
            if op == "ts_mean": return _rolling(args[0], window, "mean")
            if op == "ts_ema":
                return _frame(args[0]).ewm(
                    span=window,
                    adjust=False,
                    min_periods=max(2, window // 2),
                ).mean().to_numpy()
            if op == "ts_std": return _rolling(args[0], window, "std")
            if op == "ts_min": return _rolling(args[0], window, "min")
            if op == "ts_max": return _rolling(args[0], window, "max")
            if op == "ts_median": return _rolling(args[0], window, "median")
            if op == "ts_skew": return _rolling(args[0], window, "skew")
            if op == "ts_kurt": return _rolling(args[0], window, "kurt")
            if op == "ts_rank":
                min_periods = max(2, window // 2)
                return _frame(args[0]).rolling(
                    window, min_periods=min_periods
                ).rank(method="average", pct=True).to_numpy()
            if op == "ts_zscore":
                mean = _rolling(args[0], window, "mean")
                std = _rolling(args[0], window, "std")
                return _protected_divide(args[0] - mean, std)
            if op == "ts_corr":
                return _frame(args[0]).rolling(
                    window, min_periods=max(2, window // 2)
                ).corr(_frame(args[1])).to_numpy()
            if op == "ts_cov":
                return _frame(args[0]).rolling(
                    window, min_periods=max(2, window // 2)
                ).cov(_frame(args[1])).to_numpy()
            if op == "decay_linear":
                return _decay_linear(args[0], window)
            if op == "cs_rank":
                value = args[0]
                if self.cross_section_mask is not None:
                    value = np.where(self.cross_section_mask, value, np.nan)
                return _cross_section_rank(value)
            if op in {"cs_zscore", "cs_demean"}:
                value = args[0]
                if self.cross_section_mask is not None:
                    value = np.where(self.cross_section_mask, value, np.nan)
                valid = np.isfinite(value)
                count = valid.sum(axis=1, keepdims=True)
                mean = np.divide(
                    np.where(valid, value, 0.0).sum(axis=1, keepdims=True),
                    count,
                    out=np.full((len(args[0]), 1), np.nan, dtype=float),
                    where=count > 0,
                )
                if op == "cs_demean": return value - mean
                squared = np.where(valid, (value - mean) ** 2, 0.0).sum(
                    axis=1, keepdims=True
                )
                std = np.sqrt(np.divide(
                    squared,
                    count,
                    out=np.full_like(squared, np.nan, dtype=float),
                    where=count > 0,
                ))
                return _protected_divide(value - mean, std)
            if op == "gt": return _protected_compare(args[0], args[1], "gt")
            if op == "lt": return _protected_compare(args[0], args[1], "lt")
            if op == "logical_and": return _protected_logical(args[0], args[1], "and")
            if op == "logical_or": return _protected_logical(args[0], args[1], "or")
            if op == "logical_not":
                return np.where(np.isfinite(args[0]), args[0] <= 0, np.nan)
            if op == "if_else":
                selected = np.where(args[0] > 0, args[1], args[2])
                valid = np.isfinite(args[0]) & np.isfinite(selected)
                return np.where(valid, selected, np.nan)
        raise ValueError(f"unsupported operator: {op}")


def _protected_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    result = np.full(numerator.shape, np.nan, dtype=np.float32)
    finite = np.isfinite(numerator) & np.isfinite(denominator)
    stable = finite & (np.abs(denominator) > _EPS)
    np.divide(numerator, denominator, out=result, where=stable)
    result[finite & ~stable] = 0.0
    return result


def _protected_compare(left: np.ndarray, right: np.ndarray, op: str) -> np.ndarray:
    valid = np.isfinite(left) & np.isfinite(right)
    comparison = left > right if op == "gt" else left < right
    return np.where(valid, comparison.astype(np.float32), np.nan)


def _protected_logical(left: np.ndarray, right: np.ndarray, op: str) -> np.ndarray:
    valid = np.isfinite(left) & np.isfinite(right)
    comparison = ((left > 0) & (right > 0)) if op == "and" else ((left > 0) | (right > 0))
    return np.where(valid, comparison.astype(np.float32), np.nan)
