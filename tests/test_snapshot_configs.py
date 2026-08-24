from __future__ import annotations

import runpy
from pathlib import Path

import yaml


def test_retired_6f_snapshot_configs_match_their_frozen_definitions():
    root = Path(__file__).resolve().parents[1]
    for name in ("6f", "6f_icir"):
        snapshot = root / "snapshot" / name
        factors = runpy.run_path(str(snapshot / "combined.py"))["FACTORS"]
        config = yaml.safe_load((snapshot / "config.yaml").read_text(encoding="utf-8"))
        assert config["factors"] == list(factors)
