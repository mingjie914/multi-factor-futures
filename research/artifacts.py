from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import pandas as pd


SCHEMA_VERSION = 1
MANIFEST_NAME = "manifest.json"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def dataframe_sha256(frame: pd.DataFrame) -> str:
    """Hash dataframe labels, dtypes and values without serialising to CSV."""
    digest = hashlib.sha256()
    digest.update(json.dumps([str(value) for value in frame.columns]).encode("utf-8"))
    digest.update(json.dumps([str(dtype) for dtype in frame.dtypes]).encode("utf-8"))
    row_hashes = pd.util.hash_pandas_object(frame, index=True, categorize=True)
    digest.update(row_hashes.to_numpy(dtype="uint64").tobytes())
    return digest.hexdigest()


def dataframe_collection_sha256(frames: Mapping[str, pd.DataFrame]) -> str:
    digest = hashlib.sha256()
    for name, frame in sorted(frames.items()):
        encoded_name = str(name).encode("utf-8")
        digest.update(len(encoded_name).to_bytes(4, "big"))
        digest.update(encoded_name)
        digest.update(bytes.fromhex(dataframe_sha256(frame)))
    return digest.hexdigest()


def canonical_config_hash(config: Any) -> str:
    """Hash research-relevant config while excluding runtime artifact/date fields."""
    if hasattr(config, "model_dump"):
        raw = config.model_dump()
    elif hasattr(config, "dict"):
        raw = config.dict()
    elif isinstance(config, Mapping):
        raw = dict(config)
    else:
        raise TypeError(f"unsupported config type: {type(config).__name__}")
    raw = json.loads(json.dumps(raw, ensure_ascii=False, default=str))
    raw.pop("research_artifacts", None)
    raw.pop("date_range", None)
    # Factor lists are research outputs in nested walk-forward. Hash the
    # candidate data/model settings, not the fold-specific selected result.
    raw.pop("factors", None)
    for sub_portfolio in raw.get("sub_portfolios", []):
        if isinstance(sub_portfolio, dict):
            sub_portfolio.pop("factors", None)
    payload = json.dumps(raw, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def source_tree_hash(root: Path) -> str:
    """Hash active project Python/config sources in stable relative-path order."""
    root = root.resolve()
    ignored_parts = {
        ".git",
        ".idea",
        ".pytest_cache",
        "__pycache__",
        "cache",
        "reports",
        "runs",
        "signals_output",
        "test_cache",
        "skill-quant-factor-risk-pattern-alpha-main",
    }
    candidates = []
    for path in root.rglob("*"):
        if not path.is_file() or any(part in ignored_parts for part in path.parts):
            continue
        if path.suffix.lower() not in {".py", ".yaml", ".yml", ".json"}:
            continue
        candidates.append(path)
    digest = hashlib.sha256()
    for path in sorted(candidates, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        file_hash = bytes.fromhex(sha256_file(path))
        digest.update(file_hash)
    return digest.hexdigest()


class ResearchArtifactBundle:
    """Validated, immutable collection of point-in-time research artifacts."""

    REQUIRED_METADATA = {
        "artifact_id",
        "as_of_date",
        "train_start",
        "train_end",
        "data_sha256",
        "config_sha256",
        "code_sha256",
        "files",
    }

    def __init__(self, root: Path, manifest: Dict[str, Any]):
        self.root = root.resolve()
        self.manifest = manifest
        self._csv_cache: Dict[str, pd.DataFrame] = {}
        self._json_cache: Dict[str, Any] = {}

    @classmethod
    def load(
        cls,
        root: str | Path,
        *,
        decision_date: Optional[Any] = None,
        expected_config_hash: Optional[str] = None,
    ) -> "ResearchArtifactBundle":
        root_path = Path(root).expanduser().resolve()
        manifest_path = root_path / MANIFEST_NAME
        if not manifest_path.is_file():
            raise FileNotFoundError(f"research artifact manifest not found: {manifest_path}")
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        cls._validate_manifest(root_path, manifest, decision_date, expected_config_hash)
        return cls(root_path, manifest)

    @classmethod
    def create(
        cls,
        root: str | Path,
        *,
        artifact_id: str,
        train_start: Any,
        train_end: Any,
        data_sha256: str,
        config_sha256: str,
        code_sha256: str,
        files: Mapping[str, str | Path],
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "ResearchArtifactBundle":
        root_path = Path(root).expanduser().resolve()
        root_path.mkdir(parents=True, exist_ok=True)
        manifest_path = root_path / MANIFEST_NAME
        if manifest_path.exists():
            raise FileExistsError(f"refusing to overwrite immutable bundle: {manifest_path}")
        if not artifact_id.strip():
            raise ValueError("artifact_id must not be empty")
        start = pd.Timestamp(train_start)
        end = pd.Timestamp(train_end)
        if start > end:
            raise ValueError("train_start must be <= train_end")

        file_entries: Dict[str, Dict[str, Any]] = {}
        for logical_name, raw_path in sorted(files.items()):
            path = Path(raw_path).expanduser().resolve()
            try:
                relative = path.relative_to(root_path)
            except ValueError as exc:
                raise ValueError(f"artifact file must be inside bundle root: {path}") from exc
            if not path.is_file():
                raise FileNotFoundError(path)
            file_entries[str(logical_name)] = {
                "path": relative.as_posix(),
                "sha256": sha256_file(path),
                "size": path.stat().st_size,
            }

        manifest: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "artifact_id": artifact_id,
            "as_of_date": end.date().isoformat(),
            "train_start": start.date().isoformat(),
            "train_end": end.date().isoformat(),
            "data_sha256": cls._validate_hash(data_sha256, "data_sha256"),
            "config_sha256": cls._validate_hash(config_sha256, "config_sha256"),
            "code_sha256": cls._validate_hash(code_sha256, "code_sha256"),
            "files": file_entries,
            "metadata": dict(metadata or {}),
        }
        temporary = manifest_path.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        temporary.replace(manifest_path)
        return cls.load(root_path)

    @classmethod
    def _validate_manifest(
        cls,
        root: Path,
        manifest: Dict[str, Any],
        decision_date: Optional[Any],
        expected_config_hash: Optional[str],
    ) -> None:
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported research artifact schema: {manifest.get('schema_version')!r}"
            )
        missing = cls.REQUIRED_METADATA - set(manifest)
        if missing:
            raise ValueError(f"research artifact metadata missing: {sorted(missing)}")
        start = pd.Timestamp(manifest["train_start"])
        end = pd.Timestamp(manifest["train_end"])
        as_of = pd.Timestamp(manifest["as_of_date"])
        if start > end or as_of != end:
            raise ValueError("artifact training range/as_of_date is inconsistent")
        if decision_date is not None and as_of >= pd.Timestamp(decision_date):
            raise ValueError(
                f"artifact as_of_date {as_of.date()} must be before decision date "
                f"{pd.Timestamp(decision_date).date()}"
            )
        for field in ("data_sha256", "config_sha256", "code_sha256"):
            cls._validate_hash(manifest[field], field)
        if expected_config_hash is not None:
            cls._validate_hash(expected_config_hash, "expected_config_hash")
            if manifest["config_sha256"] != expected_config_hash:
                raise ValueError("research artifact config hash does not match runtime config")
        files = manifest["files"]
        if not isinstance(files, dict):
            raise ValueError("artifact files must be a mapping")
        for logical_name, entry in files.items():
            if not isinstance(entry, dict) or not {"path", "sha256"}.issubset(entry):
                raise ValueError(f"invalid artifact entry: {logical_name}")
            expected = cls._validate_hash(entry["sha256"], f"files.{logical_name}.sha256")
            path = cls._safe_path(root, entry["path"])
            if not path.is_file():
                raise FileNotFoundError(f"artifact file missing: {path}")
            if sha256_file(path) != expected:
                raise ValueError(f"artifact hash mismatch: {logical_name}")

    @staticmethod
    def _validate_hash(value: Any, field: str) -> str:
        text = str(value)
        if len(text) != 64 or any(char not in "0123456789abcdef" for char in text.lower()):
            raise ValueError(f"{field} must be a SHA-256 hex digest")
        return text.lower()

    @staticmethod
    def _safe_path(root: Path, relative: str) -> Path:
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"artifact path escapes bundle root: {relative}") from exc
        return path

    @property
    def artifact_id(self) -> str:
        return str(self.manifest["artifact_id"])

    @property
    def as_of_date(self) -> pd.Timestamp:
        return pd.Timestamp(self.manifest["as_of_date"])

    def has(self, logical_name: str) -> bool:
        return logical_name in self.manifest["files"]

    def path_for(self, logical_name: str) -> Path:
        if not self.has(logical_name):
            raise KeyError(f"artifact not in bundle: {logical_name}")
        return self._safe_path(self.root, self.manifest["files"][logical_name]["path"])

    def read_csv(self, logical_name: str, **kwargs) -> pd.DataFrame:
        if logical_name not in self._csv_cache:
            self._csv_cache[logical_name] = pd.read_csv(self.path_for(logical_name), **kwargs)
        return self._csv_cache[logical_name].copy(deep=True)

    def read_json(self, logical_name: str) -> Any:
        if logical_name not in self._json_cache:
            with self.path_for(logical_name).open("r", encoding="utf-8") as handle:
                self._json_cache[logical_name] = json.load(handle)
        return json.loads(json.dumps(self._json_cache[logical_name], ensure_ascii=False))
