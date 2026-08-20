from __future__ import annotations

import yaml

from workflows.close_decision import build_close_target_publication


def test_close_publication_has_no_targets_when_disabled(tmp_path):
    config = tmp_path / "target_publication.yaml"
    config.write_text(
        yaml.safe_dump({"enabled": False, "approval_status": "disabled"}),
        encoding="utf-8",
    )

    decision = build_close_target_publication(config, "2026-07-24")

    assert decision["status"] == "NO_TARGETS"
    assert decision["reason_code"] == "TARGET_PUBLICATION_DISABLED"
    assert decision["target_weights"] == {}


def test_close_publication_does_not_promote_unapproved_package(tmp_path):
    package = tmp_path / "candidate.yaml"
    package.write_text("factors: [candidate]\n", encoding="utf-8")
    config = tmp_path / "target_publication.yaml"
    config.write_text(
        yaml.safe_dump({
            "enabled": True,
            "approval_status": "historical_candidate",
            "deployment_package": str(package),
        }),
        encoding="utf-8",
    )

    decision = build_close_target_publication(config, "2026-07-24")

    assert decision["status"] == "NO_TARGETS"
    assert decision["reason_code"] == "TARGET_PUBLICATION_NOT_APPROVED"


def test_observation_approval_does_not_authorize_target_publication(tmp_path):
    package = tmp_path / "observation.yaml"
    package.write_text("factors: [observed]\n", encoding="utf-8")
    config = tmp_path / "target_publication.yaml"
    config.write_text(
        yaml.safe_dump({
            "enabled": True,
            "approval_status": "approved_for_observation",
            "deployment_package": str(package),
        }),
        encoding="utf-8",
    )

    decision = build_close_target_publication(config, "2026-07-24")

    assert decision["status"] == "NO_TARGETS"
    assert decision["reason_code"] == "TARGET_PUBLICATION_NOT_APPROVED"


def test_close_publication_requires_iso_date(tmp_path):
    config = tmp_path / "target_publication.yaml"
    config.write_text("enabled: false\n", encoding="utf-8")

    try:
        build_close_target_publication(config, "2026/07/24")
    except ValueError:
        pass
    else:
        raise AssertionError("invalid decision dates must fail closed")
