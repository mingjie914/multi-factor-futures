"""Content-addressed immutable handoffs; completion is a manifest, not a CSV alone."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile


def resolve(value) -> Path:
    """Resolve a trading path without importing the research runtime."""
    path = Path(value)
    return (path if path.is_absolute() else Path(__file__).resolve().parents[1] / path).resolve()


def encoded(value) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                       separators=(",", ":")) + "\n").encode("utf-8")


def digest(value) -> str:
    return hashlib.sha256(encoded(value)).hexdigest()


def file_hash(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_artifact(directory) -> dict:
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or digest(manifest["identity"]) != directory.name:
        raise ValueError("artifact identity hash mismatch")
    for name, expected in manifest["files"].items():
        if Path(name).name != name or file_hash(directory / name) != expected:
            raise ValueError(f"artifact file hash mismatch: {name}")
    return manifest


def read_csv(directory, name) -> list[dict]:
    manifest = read_artifact(directory)
    if name not in manifest["files"]:
        raise ValueError(f"unregistered artifact file: {name}")
    with (Path(directory) / name).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_artifact(root, identity: dict, files: dict) -> Path:
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / digest(identity)
    payloads = {}
    for name, value in files.items():
        if Path(name).name != name or name == "manifest.json":
            raise ValueError("artifact names must be plain filenames")
        if name.endswith(".csv"):
            stream = io.StringIO(newline="")
            fields = list(dict.fromkeys(key for row in value for key in row))
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(value)
            payloads[name] = stream.getvalue().encode("utf-8-sig")
        else:
            payloads[name] = encoded(value)
    manifest = {"schema_version": 1, "identity": identity,
                "files": {name: hashlib.sha256(body).hexdigest() for name, body in payloads.items()}}
    if destination.exists():
        if read_artifact(destination) != manifest:
            raise ValueError("same input identity produced different output hashes")
        return destination
    # Rename a complete sibling directory. Readers never see partially written handoffs.
    with tempfile.TemporaryDirectory(prefix=".building-", dir=root) as temp:
        staged = Path(temp) / "ready"
        staged.mkdir()
        for name, body in {**payloads, "manifest.json": encoded(manifest)}.items():
            with (staged / name).open("wb") as handle:
                handle.write(body)
                handle.flush()
                os.fsync(handle.fileno())
        try:
            staged.rename(destination)
        except FileExistsError:
            if read_artifact(destination) != manifest:
                raise ValueError("concurrent artifact hash mismatch")
    return destination
