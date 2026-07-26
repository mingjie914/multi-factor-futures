from __future__ import annotations

import yaml

from workflows.close_decision import build_close_decision


def test_close_decision_is_no_trade_when_disabled(tmp_path):
    config = tmp_path / "trading.yaml"
    config.write_text(
        yaml.safe_dump({"enabled": False, "approval_status": "disabled"}),
        encoding="utf-8",
    )

    decision = build_close_decision(config, "2026-07-24")

    assert decision["status"] == "NO_TRADE"
    assert decision["reason_code"] == "TRADING_DISABLED"
    assert decision["target_weights"] == {}
    assert decision["orders"] == []


def test_close_decision_does_not_promote_unapproved_package(tmp_path):
    package = tmp_path / "candidate.yaml"
    package.write_text("factors: [candidate]\n", encoding="utf-8")
    config = tmp_path / "trading.yaml"
    config.write_text(
        yaml.safe_dump({
            "enabled": True,
            "approval_status": "historical_candidate",
            "deployment_package": str(package),
        }),
        encoding="utf-8",
    )

    decision = build_close_decision(config, "2026-07-24")

    assert decision["status"] == "NO_TRADE"
    assert decision["reason_code"] == "DEPLOYMENT_NOT_APPROVED"


def test_close_decision_requires_iso_date(tmp_path):
    config = tmp_path / "trading.yaml"
    config.write_text("enabled: false\n", encoding="utf-8")

    try:
        build_close_decision(config, "2026/07/24")
    except ValueError:
        pass
    else:
        raise AssertionError("invalid decision dates must fail closed")
