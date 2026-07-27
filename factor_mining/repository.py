"""SQLite candidate catalog and immutable framework snapshots."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Iterable, Iterator, Mapping, Sequence

from factor_mining.api import CandidateSpec, MiningRunSpec, canonical_json, content_hash


SCHEMA_VERSION = 1
ALLOWED_STATUSES = {
    "mined_candidate",
    "development_candidate",
    "historical_candidate",
    "oos_validated",
    "rejected",
}
STATUS_TRANSITIONS = {
    "mined_candidate": {
        "development_candidate", "historical_candidate", "rejected"
    },
    "development_candidate": {
        "historical_candidate", "oos_validated", "rejected"
    },
    "historical_candidate": {"oos_validated", "rejected"},
    "oos_validated": set(),
    "rejected": set(),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class CandidateRepository:
    """Small catalog for discovery state; never used directly by live factors."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=30000")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    engine TEXT NOT NULL,
                    spec_json TEXT NOT NULL,
                    spec_sha256 TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS candidates (
                    candidate_id TEXT PRIMARY KEY,
                    framework_name TEXT NOT NULL UNIQUE,
                    content_sha256 TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    status TEXT NOT NULL,
                    run_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    candidate_json TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES runs(run_id)
                );
                CREATE TABLE IF NOT EXISTS evaluations (
                    evaluation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    candidate_id TEXT NOT NULL,
                    run_id TEXT,
                    stage TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    evidence_json TEXT NOT NULL,
                    FOREIGN KEY(candidate_id) REFERENCES candidates(candidate_id)
                );
                CREATE INDEX IF NOT EXISTS idx_candidates_status
                    ON candidates(status, updated_at);
                CREATE INDEX IF NOT EXISTS idx_evaluations_candidate
                    ON evaluations(candidate_id, created_at);
                """
            )

    def add_run(self, spec: MiningRunSpec) -> None:
        payload = spec.to_dict()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO runs(run_id, created_at, mode, engine, spec_json, spec_sha256)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    spec.run_id,
                    _utc_now(),
                    spec.mode.value,
                    spec.engine,
                    canonical_json(payload),
                    content_hash(payload),
                ),
            )

    def add_candidates(
        self, candidates: Iterable[CandidateSpec], *, run_id: str | None = None
    ) -> int:
        rows = []
        now = _utc_now()
        for candidate in candidates:
            candidate.validated()
            if candidate.status not in ALLOWED_STATUSES:
                raise ValueError(f"unsupported candidate status: {candidate.status}")
            value = candidate.to_dict()
            rows.append((
                candidate.candidate_id,
                candidate.framework_name,
                value["content_sha256"],
                candidate.kind,
                candidate.status,
                run_id or candidate.lineage.get("run_id"),
                now,
                now,
                canonical_json(value),
            ))
        if not rows:
            return 0
        inserted = 0
        with self._connect() as connection:
            for row in rows:
                existing = connection.execute(
                    "SELECT content_sha256 FROM candidates WHERE candidate_id=?",
                    (row[0],),
                ).fetchone()
                if existing is not None:
                    if existing["content_sha256"] != row[2]:
                        raise ValueError(f"candidate id collision: {row[0]}")
                    continue
                connection.execute(
                    """
                    INSERT INTO candidates(
                        candidate_id, framework_name, content_sha256, kind, status,
                        run_id, created_at, updated_at, candidate_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    row,
                )
                inserted += 1
        return inserted

    def list_candidates(
        self,
        *,
        statuses: Sequence[str] | None = None,
        limit: int | None = None,
    ) -> tuple[CandidateSpec, ...]:
        query = "SELECT candidate_json FROM candidates"
        params: list[object] = []
        if statuses:
            unknown = sorted(set(statuses) - ALLOWED_STATUSES)
            if unknown:
                raise ValueError(f"unsupported statuses: {unknown}")
            placeholders = ",".join("?" for _ in statuses)
            query += f" WHERE status IN ({placeholders})"
            params.extend(statuses)
        query += " ORDER BY updated_at DESC, candidate_id"
        if limit is not None:
            if limit < 1:
                raise ValueError("limit must be positive")
            query += " LIMIT ?"
            params.append(int(limit))
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return tuple(CandidateSpec.from_dict(json.loads(row["candidate_json"])) for row in rows)

    def get_candidate(self, candidate_id: str) -> CandidateSpec:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT candidate_json FROM candidates WHERE candidate_id=?",
                (candidate_id,),
            ).fetchone()
        if row is None:
            raise KeyError(candidate_id)
        return CandidateSpec.from_dict(json.loads(row["candidate_json"]))

    def record_evaluation(
        self,
        candidate_id: str,
        *,
        stage: str,
        metrics: Mapping,
        evidence: Mapping,
        run_id: str | None = None,
    ) -> int:
        """Append evidence without changing the candidate's status."""
        if not stage.strip():
            raise ValueError("evaluation stage is required")
        self.get_candidate(candidate_id)
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO evaluations(
                    candidate_id, run_id, stage, created_at, metrics_json, evidence_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    candidate_id,
                    run_id,
                    stage,
                    _utc_now(),
                    canonical_json(dict(metrics)),
                    canonical_json(dict(evidence)),
                ),
            )
            return int(cursor.lastrowid)

    def promote(
        self,
        candidate_id: str,
        status: str,
        *,
        evidence: Mapping | None = None,
        run_id: str | None = None,
    ) -> CandidateSpec:
        if status not in ALLOWED_STATUSES:
            raise ValueError(f"unsupported candidate status: {status}")
        current = self.get_candidate(candidate_id)
        if status not in STATUS_TRANSITIONS[current.status]:
            raise ValueError(
                f"invalid candidate status transition: {current.status} -> {status}"
            )
        if status in {
            "development_candidate", "historical_candidate", "oos_validated"
        } and evidence is None:
            raise ValueError(f"status {status} requires audit evidence")
        if status in {
            "development_candidate", "historical_candidate", "oos_validated"
        } and evidence.get("valid") is not True:
            raise ValueError(f"status {status} requires valid audit evidence")
        value = current.to_dict()
        value["status"] = status
        promoted = CandidateSpec.from_dict(value)
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE candidates
                SET status=?, updated_at=?, candidate_json=?
                WHERE candidate_id=?
                """,
                (status, _utc_now(), canonical_json(promoted.to_dict()), candidate_id),
            )
            if evidence is not None:
                connection.execute(
                    """
                    INSERT INTO evaluations(
                        candidate_id, run_id, stage, created_at, metrics_json, evidence_json
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        candidate_id,
                        run_id,
                        status,
                        _utc_now(),
                        canonical_json(promoted.metrics),
                        canonical_json(evidence),
                    ),
                )
        return promoted

    def write_snapshot(
        self,
        output: str | Path,
        *,
        candidate_ids: Sequence[str] | None = None,
        statuses: Sequence[str] | None = None,
        refuse_existing: bool = True,
    ) -> Path:
        if bool(candidate_ids) == bool(statuses):
            raise ValueError("select snapshot candidates by ids or statuses, exclusively")
        if candidate_ids:
            candidates = tuple(self.get_candidate(item) for item in candidate_ids)
        else:
            candidates = self.list_candidates(statuses=statuses)
        if not candidates:
            raise ValueError("snapshot selection is empty")
        names = [candidate.framework_name for candidate in candidates]
        if len(names) != len(set(names)):
            raise ValueError("snapshot has duplicate framework names")
        payload = {
            "schema_version": SCHEMA_VERSION,
            "created_at": _utc_now(),
            "candidate_count": len(candidates),
            "candidates": [candidate.to_dict() for candidate in candidates],
        }
        snapshot = dict(payload)
        snapshot["snapshot_sha256"] = content_hash(payload)
        path = Path(output).expanduser().resolve()
        if refuse_existing and path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        mode = "x" if refuse_existing else "w"
        with path.open(mode, encoding="utf-8") as handle:
            handle.write(
                json.dumps(snapshot, ensure_ascii=False, sort_keys=True, indent=2)
                + "\n"
            )
        return path


def load_snapshot(path: str | Path) -> tuple[CandidateSpec, ...]:
    snapshot_path = Path(path).expanduser().resolve()
    value = json.loads(snapshot_path.read_text(encoding="utf-8"))
    if int(value.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("unsupported candidate snapshot schema")
    declared = value.pop("snapshot_sha256", "")
    if not declared or declared != content_hash(value):
        raise ValueError("candidate snapshot hash mismatch")
    candidates = tuple(
        CandidateSpec.from_dict(item) for item in value.get("candidates", ())
    )
    if len(candidates) != int(value.get("candidate_count", -1)):
        raise ValueError("candidate snapshot count mismatch")
    names = [candidate.framework_name for candidate in candidates]
    if len(names) != len(set(names)):
        raise ValueError("candidate snapshot contains duplicate names")
    return candidates
