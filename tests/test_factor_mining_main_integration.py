from __future__ import annotations

import os

import pytest

import main
from core.sectors import FRAMEWORK_UNIVERSE
from factor_mining.api import CandidateSpec, FeatureConfig, TargetSpec
from factor_mining.bridge import SNAPSHOT_ENV
from factor_mining.operators import Expr
from factor_mining.repository import CandidateRepository


def _snapshot(tmp_path):
    expression = Expr.operation("cs_rank", Expr.terminal("return_1p"))
    candidate = CandidateSpec(
        candidate_id="gp_main_gateway_test",
        framework_name="mined_gp_main_gateway_test",
        kind="symbolic",
        category="auto_mined",
        frequency="1min",
        target=TargetSpec(name="forward_5p", horizon_bars=5),
        dependencies=("close",),
        lookback_bars=2,
        payload={
            "expression": expression.to_dict(),
            "expression_sha256": expression.sha256,
            "decision_lag_bars": 1,
            "postprocess": {
                "mad_clip": 5.0,
                "neutralize_volatility": False,
                "volatility_feature": None,
            },
        },
        feature_config=FeatureConfig(
            raw_fields=("close",),
            feature_horizons=(1, 5),
            lag_steps=(1,),
            rolling_windows=(3, 5),
        ),
    )
    repository = CandidateRepository(tmp_path / "pool.sqlite3")
    repository.add_candidates((candidate,))
    return repository.write_snapshot(
        tmp_path / "snapshot.json", candidate_ids=(candidate.candidate_id,),
        framework_universe=FRAMEWORK_UNIVERSE,
    )


def test_main_exposes_factor_mining_as_first_class_command():
    assert main.WORKFLOW_COMMANDS["mining"][0] == "factor_mining.cli"


def test_main_validates_and_strips_mined_snapshot_before_workflow_import(
    tmp_path, monkeypatch
):
    snapshot = _snapshot(tmp_path)
    monkeypatch.delenv(SNAPSHOT_ENV, raising=False)
    argv = [
        "main.py",
        "research",
        "--factors",
        "mined_gp_main_gateway_test",
        "--mined-snapshot",
        str(snapshot),
        "--multi-period",
    ]

    cleaned, count = main._configure_mined_snapshot(argv)

    assert count == 1
    assert "--mined-snapshot" not in cleaned
    assert cleaned[-1] == "--multi-period"
    assert os.environ[SNAPSHOT_ENV] == str(snapshot.resolve())


def test_main_rejects_snapshot_gateway_option_for_mining_command(tmp_path):
    snapshot = _snapshot(tmp_path)
    with pytest.raises(ValueError, match="not a mining-command option"):
        main._configure_mined_snapshot(
            ["main.py", "mining", "--mined-snapshot", str(snapshot), "pool-list"]
        )


def test_main_dispatches_synthetic_mining_smoke(monkeypatch, capsys):
    monkeypatch.setattr(
        main.sys,
        "argv",
        [
            "main.py", "mining", "dev-smoke",
            "--periods", "80", "--symbols", "5",
            "--population", "8", "--generations", "1",
            "--max-candidates", "2",
            "--candidate-prefix", "smoke",
            "--sector-neutralization",
            "--rolling-windows", "3,15",
            "--feature-horizons", "1,5,15",
        ],
    )

    main.main()

    output = capsys.readouterr().out
    assert '"run_id":"synthetic_dev"' in output
    assert '"candidate_count"' in output
    assert '"candidate_id":"gp_smoke_h15_' in output
