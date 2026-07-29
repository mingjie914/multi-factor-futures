from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from factor_mining.api import FeatureConfig, TargetSpec, canonical_json, content_hash
from factor_mining.features import FeatureSet
from factor_mining.validation import PreparedTarget, _row_nanstd, shift_signal


SCHEMA_VERSION = 1
TERMINAL_SNAPSHOT = "terminal_snapshot.npy"
METADATA_FILE = "snapshot_metadata.json"
TARGET_VALUES_FILE = "target_values.npy"
TARGET_RANKS_FILE = "target_rank_values.npy"
TARGET_FINITE_FILE = "target_finite_mask.npy"
SHIFTED_VOLATILITY_FILE = "shifted_volatility.npy"
COVERAGE_MASK_FILE = "coverage_mask.npy"


def _sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _update_array_hash(digest, value: np.ndarray) -> None:
    array = np.asarray(value)
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(canonical_json(list(array.shape)).encode("utf-8"))
    contiguous = np.ascontiguousarray(array)
    view = memoryview(contiguous).cast("B")
    chunk_bytes = 8 * 1024 * 1024
    for start in range(0, len(view), chunk_bytes):
        digest.update(view[start:start + chunk_bytes])


def source_data_fingerprint(panels: Mapping[str, pd.DataFrame]) -> str:
    """Hash aligned source panels without pickling framework objects."""

    digest = hashlib.sha256()
    for name in sorted(map(str, panels)):
        frame = panels[name]
        digest.update(name.encode("utf-8"))
        digest.update(
            np.asarray(pd.DatetimeIndex(frame.index).asi8, dtype=np.int64).tobytes()
        )
        digest.update(canonical_json(list(map(str, frame.columns))).encode("utf-8"))
        _update_array_hash(digest, frame.to_numpy(copy=False))
    return digest.hexdigest()


def snapshot_cache_key(
    *,
    feature_config: FeatureConfig,
    target_spec: TargetSpec,
    taxonomy: Mapping[str, object],
    source_fingerprint: str,
    decision_lag_bars: int,
    neutralize_volatility: bool,
) -> str:
    return content_hash({
        "feature_config": asdict(feature_config),
        "target_spec": asdict(target_spec),
        "taxonomy": dict(taxonomy),
        "source_data_fingerprint": str(source_fingerprint),
        "decision_lag_bars": int(decision_lag_bars),
        "neutralize_volatility": bool(neutralize_volatility),
        "schema_version": SCHEMA_VERSION,
    })


@dataclass(frozen=True)
class StaticResearchContext:
    """Read-only terminal snapshot and GP-invariant research state.

    The on-disk array uses ``(F, T, N)`` so each terminal is a contiguous
    zero-copy ``(T, N)`` view.  ``terminal_matrix`` exposes the requested
    logical ``(T, N, F)`` view without copying.
    """

    cache_dir: Path
    metadata: Mapping[str, object]
    terminal_storage: np.ndarray
    features: FeatureSet
    target: PreparedTarget
    target_finite_mask: np.ndarray
    target_dispersion: float
    coverage_mask: np.ndarray
    coverage_denominator: int
    volatility: np.ndarray | None
    shifted_volatility: np.ndarray | None
    group_labels: tuple[str, ...] | None
    industry_group_indices: Mapping[str, np.ndarray]

    @property
    def terminal_matrix(self) -> np.ndarray:
        return self.terminal_storage.transpose(1, 2, 0)

    def __getitem__(self, feature_name: str) -> np.ndarray:
        return self.features.values[str(feature_name)]

    @property
    def terminal_names(self) -> tuple[str, ...]:
        return tuple(map(str, self.metadata["terminal_names"]))

    @classmethod
    def create(
        cls,
        cache_dir: str | Path,
        *,
        features: FeatureSet,
        target: PreparedTarget,
        feature_config: FeatureConfig,
        source_fingerprint: str,
        taxonomy: Mapping[str, object] | None = None,
        volatility: np.ndarray | None = None,
        group_labels: Sequence[str] | None = None,
        decision_lag_bars: int = 1,
        block_rows: int = 2500,
    ) -> "StaticResearchContext":
        destination = Path(cache_dir)
        metadata_path = destination / METADATA_FILE
        if metadata_path.exists():
            raise FileExistsError(
                f"terminal snapshot already exists; validate and load it: {destination}"
            )
        destination.mkdir(parents=True, exist_ok=True)
        if features.shape != target.values.shape:
            raise ValueError("feature and target shapes differ")
        labels = None if group_labels is None else tuple(map(str, group_labels))
        if labels is not None and len(labels) != features.shape[1]:
            raise ValueError("group_labels must contain one label per feature symbol")

        names = tuple(features.feature_names)
        rows, columns = features.shape
        snapshot_path = destination / TERMINAL_SNAPSHOT
        temporary_snapshot = destination / f"{TERMINAL_SNAPSHOT}.tmp"
        storage = np.lib.format.open_memmap(
            temporary_snapshot,
            mode="w+",
            dtype=np.float32,
            shape=(len(names), rows, columns),
        )
        for index, name in enumerate(names):
            storage[index] = np.asarray(features.values[name], dtype=np.float32)
        storage.flush()
        del storage
        temporary_snapshot.replace(snapshot_path)

        target_values = np.asarray(target.values, dtype=np.float32)
        target_ranks = np.asarray(target.rank_values, dtype=np.float32)
        target_finite = np.isfinite(target_values)
        aligned_volatility = None
        if volatility is not None:
            volatility = np.asarray(volatility, dtype=np.float32)
            if volatility.shape != features.shape:
                raise ValueError("volatility control shape differs from features")
            aligned_volatility = shift_signal(volatility, int(decision_lag_bars))
        coverage_mask = target_finite.copy()
        if aligned_volatility is not None:
            coverage_mask &= np.isfinite(aligned_volatility)

        np.save(destination / TARGET_VALUES_FILE, target_values)
        np.save(destination / TARGET_RANKS_FILE, target_ranks)
        np.save(destination / TARGET_FINITE_FILE, target_finite)
        np.save(destination / COVERAGE_MASK_FILE, coverage_mask)
        if aligned_volatility is not None:
            np.save(destination / SHIFTED_VOLATILITY_FILE, aligned_volatility)

        artifact_sha256 = {
            name: _sha256_file(destination / name)
            for name in (
                TERMINAL_SNAPSHOT,
                TARGET_VALUES_FILE,
                TARGET_RANKS_FILE,
                TARGET_FINITE_FILE,
                COVERAGE_MASK_FILE,
            )
        }
        if aligned_volatility is not None:
            artifact_sha256[SHIFTED_VOLATILITY_FILE] = _sha256_file(
                destination / SHIFTED_VOLATILITY_FILE
            )
        taxonomy_value = dict(taxonomy or {})
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "storage_layout": "FTN",
            "logical_axis_order": "TNF",
            "terminal_names": list(names),
            "terminal_index": {name: index for index, name in enumerate(names)},
            "storage_shape": [len(names), rows, columns],
            "logical_shape": [rows, columns, len(names)],
            "T": rows,
            "N": columns,
            "timestamps": [timestamp.isoformat() for timestamp in features.index],
            "symbols": list(map(str, features.symbols)),
            "feature_config": asdict(feature_config),
            "feature_config_sha256": content_hash(asdict(feature_config)),
            "target_spec": asdict(target.spec),
            "target_spec_sha256": content_hash(asdict(target.spec)),
            "taxonomy": taxonomy_value,
            "taxonomy_sha256": content_hash(taxonomy_value),
            "source_data_fingerprint": str(source_fingerprint),
            "snapshot_sha256": artifact_sha256[TERMINAL_SNAPSHOT],
            "artifact_sha256": artifact_sha256,
            "raw_dependencies": {
                name: sorted(map(str, features.raw_dependencies.get(name, ())))
                for name in names
            },
            "lookbacks": {
                name: int(features.lookbacks.get(name, 0)) for name in names
            },
            "dtype": "float32",
            "decision_lag_bars": int(decision_lag_bars),
            "volatility_feature_present": aligned_volatility is not None,
            "neutralize_volatility": aligned_volatility is not None,
            "group_labels": list(labels) if labels is not None else None,
            "target_dispersion": float(np.nanmean(_row_nanstd(target_values))),
            "coverage_denominator": int(coverage_mask.sum()),
            "write_block_rows": int(block_rows),
        }
        temporary_metadata = destination / f"{METADATA_FILE}.tmp"
        temporary_metadata.write_text(
            json.dumps(metadata, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        temporary_metadata.replace(metadata_path)
        return cls.load(
            destination,
            expected_feature_config=feature_config,
            expected_target_spec=target.spec,
            expected_taxonomy=taxonomy_value,
            expected_source_fingerprint=source_fingerprint,
            expected_decision_lag_bars=decision_lag_bars,
            expected_neutralize_volatility=aligned_volatility is not None,
        )

    @classmethod
    def load(
        cls,
        cache_dir: str | Path,
        *,
        expected_feature_config: FeatureConfig | None = None,
        expected_target_spec: TargetSpec | None = None,
        expected_taxonomy: Mapping[str, object] | None = None,
        expected_source_fingerprint: str | None = None,
        expected_decision_lag_bars: int | None = None,
        expected_neutralize_volatility: bool | None = None,
    ) -> "StaticResearchContext":
        source = Path(cache_dir)
        metadata = json.loads(
            (source / METADATA_FILE).read_text(encoding="utf-8")
        )
        if int(metadata.get("schema_version", -1)) != SCHEMA_VERSION:
            raise ValueError("terminal snapshot schema version mismatch")
        if metadata.get("storage_layout") != "FTN":
            raise ValueError("unsupported terminal snapshot storage layout")
        if expected_feature_config is not None and (
            metadata.get("feature_config_sha256")
            != content_hash(asdict(expected_feature_config))
        ):
            raise ValueError("terminal snapshot FeatureConfig hash mismatch")
        if expected_target_spec is not None and (
            metadata.get("target_spec_sha256")
            != content_hash(asdict(expected_target_spec))
        ):
            raise ValueError("terminal snapshot TargetSpec hash mismatch")
        if expected_taxonomy is not None and (
            metadata.get("taxonomy_sha256")
            != content_hash(dict(expected_taxonomy))
        ):
            raise ValueError("terminal snapshot taxonomy hash mismatch")
        if expected_source_fingerprint is not None and (
            metadata.get("source_data_fingerprint")
            != str(expected_source_fingerprint)
        ):
            raise ValueError("terminal snapshot source data fingerprint mismatch")
        if expected_decision_lag_bars is not None and (
            int(metadata.get("decision_lag_bars", -1))
            != int(expected_decision_lag_bars)
        ):
            raise ValueError("terminal snapshot decision lag mismatch")
        if expected_neutralize_volatility is not None and (
            bool(metadata.get("neutralize_volatility"))
            != bool(expected_neutralize_volatility)
        ):
            raise ValueError("terminal snapshot volatility policy mismatch")

        artifacts = metadata.get("artifact_sha256", {})
        for name, expected_sha256 in artifacts.items():
            if _sha256_file(source / name) != expected_sha256:
                raise ValueError(f"terminal snapshot artifact hash mismatch: {name}")
        snapshot_path = source / TERMINAL_SNAPSHOT
        terminal_hash = artifacts.get(TERMINAL_SNAPSHOT)
        if terminal_hash is None:
            terminal_hash = _sha256_file(snapshot_path)
        if terminal_hash != metadata.get("snapshot_sha256"):
            raise ValueError("terminal snapshot SHA-256 mismatch")
        storage = np.load(snapshot_path, mmap_mode="r")
        expected_storage_shape = tuple(map(int, metadata["storage_shape"]))
        if storage.shape != expected_storage_shape or storage.dtype != np.float32:
            raise ValueError("terminal snapshot shape or dtype mismatch")
        names = tuple(map(str, metadata["terminal_names"]))
        values = {name: storage[index] for index, name in enumerate(names)}
        index = pd.DatetimeIndex(pd.to_datetime(metadata["timestamps"]))
        symbols = pd.Index(metadata["symbols"])
        features = FeatureSet(
            index=index,
            symbols=symbols,
            values=values,
            raw_dependencies={
                name: frozenset(metadata["raw_dependencies"].get(name, ()))
                for name in names
            },
            lookbacks={
                name: int(metadata["lookbacks"].get(name, 0)) for name in names
            },
            dtype="float32",
        )
        target_spec = TargetSpec.from_dict(metadata["target_spec"])
        target_values = np.load(source / TARGET_VALUES_FILE, mmap_mode="r")
        target_ranks = np.load(source / TARGET_RANKS_FILE, mmap_mode="r")
        target = PreparedTarget(
            index=index,
            symbols=symbols,
            values=target_values,
            rank_values=target_ranks,
            spec=target_spec,
        )
        target_finite = np.load(source / TARGET_FINITE_FILE, mmap_mode="r")
        coverage_mask = np.load(source / COVERAGE_MASK_FILE, mmap_mode="r")
        shifted_volatility = None
        volatility = None
        if bool(metadata.get("volatility_feature_present")):
            shifted_volatility = np.load(
                source / SHIFTED_VOLATILITY_FILE, mmap_mode="r"
            )
            for name in ("realized_vol_60p", "realized_vol_30p", "realized_vol_15p"):
                if name in values:
                    volatility = values[name]
                    break
            if volatility is None:
                raise ValueError("snapshot declares volatility but no control terminal exists")
        raw_labels = metadata.get("group_labels")
        labels = None if raw_labels is None else tuple(map(str, raw_labels))
        groups = {}
        if labels is not None:
            label_array = np.asarray(labels)
            groups = {
                str(label): np.flatnonzero(label_array == label)
                for label in np.unique(label_array)
            }
        return cls(
            cache_dir=source,
            metadata=metadata,
            terminal_storage=storage,
            features=features,
            target=target,
            target_finite_mask=target_finite,
            target_dispersion=float(metadata["target_dispersion"]),
            coverage_mask=coverage_mask,
            coverage_denominator=int(metadata["coverage_denominator"]),
            volatility=volatility,
            shifted_volatility=shifted_volatility,
            group_labels=labels,
            industry_group_indices=groups,
        )
