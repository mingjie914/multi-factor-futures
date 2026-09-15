from __future__ import annotations

import json
import runpy
from pathlib import Path

import yaml


def test_retired_6f_snapshot_configs_match_their_frozen_definitions():
    root = Path(__file__).resolve().parents[1]
    catalog = yaml.safe_load((root / "config/strategy_library.yaml").read_text(encoding="utf-8"))
    entry = next(row for row in catalog["strategies"] if row["id"] == "snapshot_6f_icir")
    assert entry["status"] == "archived"
    assert entry["source"] == "legacy_observation"
    assert entry["factor_definition_path"] == "config/factor_sets/legacy_6f.json"
    frozen = json.loads((root / entry["factor_definition_path"]).read_text(encoding="utf-8"))["FACTORS"]
    assert len(frozen) == 6
    assert set(frozen.values()) == {-1, 1}
    for name in ("6f", "6f_icir"):
        snapshot = root / "snapshot" / name
        # Local audit archives are intentionally untracked, not required fixtures.
        if not snapshot.exists():
            continue
        factors = runpy.run_path(str(snapshot / "combined.py"))["FACTORS"]
        config = yaml.safe_load((snapshot / "config.yaml").read_text(encoding="utf-8"))
        assert factors == frozen
        assert config["factors"] == list(factors)
