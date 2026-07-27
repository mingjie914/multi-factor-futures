"""Versioned factor-discovery policy and multiplicity control.

The discovery unit is an economic factor.  Horizons and preprocessing
variants are hypotheses within that factor.  Factor families are selected by
BH-adjusted Simes p-values; hypotheses in selected families use the
Benjamini-Bogomolov selection-adjusted BH level ``q * R / M``.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

import numpy as np

from research.statistics import benjamini_hochberg, simes_p_value


def policy_dict(policy: Any) -> dict:
    if hasattr(policy, "model_dump"):
        value = policy.model_dump()
    elif hasattr(policy, "dict"):
        value = policy.dict()
    elif isinstance(policy, Mapping):
        value = dict(policy)
    else:
        raise TypeError(f"unsupported validation policy: {type(policy).__name__}")
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def validate_policy(policy: Any) -> dict:
    value = policy_dict(policy)
    if value.get("discovery_method") != "hierarchical_fdr":
        raise ValueError("validation_policy.discovery_method must be hierarchical_fdr")
    for key in (
        "discovery_q", "fwer_report_alpha", "annual_direction_ratio",
        "annual_effect_ratio", "oos_fold_sign_ratio", "observation_weight_cap",
    ):
        number = float(value[key])
        if not 0.0 < number <= 1.0:
            raise ValueError(f"validation_policy.{key} must be in (0, 1]")
    if int(value["n_return_groups"]) < 2:
        raise ValueError("validation_policy.n_return_groups must be at least 2")
    if int(value["minimum_calendar_years"]) < 1:
        raise ValueError("minimum_calendar_years must be positive")
    if int(value["minimum_year_observations"]) < 1:
        raise ValueError("minimum_year_observations must be positive")
    if int(value["single_instrument_min_trading_days"]) < 1:
        raise ValueError("single_instrument_min_trading_days must be positive")
    if int(value["single_instrument_bootstrap_samples"]) < 1:
        raise ValueError("single_instrument_bootstrap_samples must be positive")
    for key in ("min_abs_ic", "min_abs_t", "max_monthly_turnover", "cost_safety_margin"):
        if float(value[key]) < 0.0:
            raise ValueError(f"validation_policy.{key} must be non-negative")
    directions = value.get("expected_directions", {})
    if any(int(direction) not in {-1, 1} for direction in directions.values()):
        raise ValueError("expected_directions values must be -1 or 1")

    scorecard = value.get("scorecard", {})
    weights = scorecard.get("weights", {})
    if scorecard.get("enabled", False):
        if not weights or any(float(weight) < 0.0 for weight in weights.values()):
            raise ValueError("scorecard weights must be non-negative and non-empty")
        if not math.isclose(sum(map(float, weights.values())), 1.0, abs_tol=1e-9):
            raise ValueError("scorecard weights must sum to 1")
        if float(scorecard.get("ir_std_max", 0.30)) < 0.0:
            raise ValueError("scorecard.ir_std_max must be non-negative")
        if not 0.0 <= float(scorecard.get("threshold", 0.0)) <= 1.0:
            raise ValueError("scorecard.threshold must be in [0, 1]")
    if scorecard.get("calibrated", False) or scorecard.get("enforced", False):
        digest = str(scorecard.get("calibration_sha256", ""))
        if scorecard.get("enforced", False) and not scorecard.get("calibrated", False):
            raise ValueError("an enforced scorecard must be calibrated")
        if not scorecard.get("calibration_source"):
            raise ValueError("a calibrated scorecard requires calibration_source")
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest.lower()):
            raise ValueError("scorecard calibration_sha256 must be a SHA-256 digest")
    horizons = value.get("family_horizons", {})
    for family, periods in horizons.items():
        normalized = [int(period) for period in periods]
        if not normalized or any(period < 1 for period in normalized):
            raise ValueError(f"family_horizons[{family!r}] must contain positive periods")
        if len(normalized) != len(set(normalized)):
            raise ValueError(f"family_horizons[{family!r}] contains duplicates")
    return value


def validation_policy_sha256(policy: Any) -> str:
    value = validate_policy(policy)
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _finite_p(value: Any, *, estimable: bool = True) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 1.0
    if not estimable or not np.isfinite(number):
        return 1.0
    return float(np.clip(number, 0.0, 1.0))


def apply_hierarchical_fdr(
    factors: Mapping[str, Sequence[MutableMapping[str, Any]]],
    *,
    q: float = 0.10,
    fwer_alpha: float = 0.05,
    p_key: str = "p_value",
    estimable_key: str | None = None,
) -> dict:
    """Annotate factor hypotheses with factor-level and local decisions.

    Empty or non-estimable families remain in the audit with p=1 but do not
    inflate the number of tested factor families.  FWER is annotation-only.
    """
    if not 0.0 < q < 1.0 or not 0.0 < fwer_alpha < 1.0:
        raise ValueError("q and fwer_alpha must be in (0, 1)")

    names = list(factors)
    valid_names: list[str] = []
    p_by_factor: dict[str, list[float]] = {}
    for name in names:
        p_values = []
        any_estimable = False
        for entry in factors[name]:
            estimable = bool(entry.get(estimable_key, True)) if estimable_key else True
            any_estimable = any_estimable or estimable
            p_values.append(_finite_p(entry.get(p_key, 1.0), estimable=estimable))
        p_by_factor[name] = p_values
        if p_values and any_estimable:
            valid_names.append(name)

    factor_p = [simes_p_value(p_by_factor[name]) for name in valid_names]
    factor_q, factor_rejected = benjamini_hochberg(factor_p, alpha=q)
    factor_decisions = {
        name: (float(p), float(q_value), bool(rejected))
        for name, p, q_value, rejected in zip(
            valid_names, factor_p, factor_q, factor_rejected
        )
    }
    selected_count = sum(decision[2] for decision in factor_decisions.values())
    family_count = len(valid_names)
    local_alpha = q * selected_count / family_count if family_count else 0.0
    total_hypotheses = sum(len(values) for values in p_by_factor.values())
    fwer_cutoff = fwer_alpha / max(total_hypotheses, 1)

    local_selected_count = 0
    for name in names:
        factor_p_value, factor_q_value, factor_selected = factor_decisions.get(
            name, (1.0, 1.0, False)
        )
        local_p = p_by_factor[name]
        if local_p:
            local_q, local_rejected = benjamini_hochberg(
                local_p, alpha=max(local_alpha, np.finfo(float).eps)
            )
        else:
            local_q = np.zeros(0, dtype=float)
            local_rejected = np.zeros(0, dtype=bool)
        for entry, p_value, local_value, local_pass in zip(
            factors[name], local_p, local_q, local_rejected
        ):
            selected = bool(factor_selected and local_alpha > 0.0 and local_pass)
            local_selected_count += int(selected)
            entry.update({
                "factor_simes_p_value": factor_p_value,
                "factor_q_value": factor_q_value,
                "factor_fdr_significant": factor_selected,
                "local_q_value": float(local_value),
                "local_fdr_alpha": float(local_alpha),
                "hierarchical_fdr_significant": selected,
                "fwer_cutoff": float(fwer_cutoff),
                "fwer_significant": bool(p_value <= fwer_cutoff),
                "evidence_level": (
                    "FWER" if p_value <= fwer_cutoff
                    else "FDR" if selected
                    else "not_discovered"
                ),
            })

    return {
        "method": "simes_bh_then_benjamini_bogomolov_bh",
        "q": float(q),
        "factor_family_count": family_count,
        "selected_factor_count": selected_count,
        "local_alpha": float(local_alpha),
        "total_hypotheses": total_hypotheses,
        "selected_hypothesis_count": local_selected_count,
        "fwer_report_alpha": float(fwer_alpha),
        "fwer_cutoff": float(fwer_cutoff),
    }


def locked_oos_direction_gate(
    train_ic: float,
    fold_ics: Iterable[float],
    locked_oos_ic: float,
    *,
    minimum_fold_ratio: float = 0.60,
) -> dict:
    """Apply the frozen training orientation to fold and locked-OOS ICs."""
    orientation = 1.0 if float(train_ic) >= 0.0 else -1.0
    folds = np.asarray(list(fold_ics), dtype=float)
    folds = folds[np.isfinite(folds)]
    fold_ratio = float(np.mean(folds * orientation > 0.0)) if len(folds) else 0.0
    oriented_locked_ic = float(locked_oos_ic) * orientation
    return {
        "training_orientation": int(orientation),
        "fold_sign_ratio": fold_ratio,
        "oriented_locked_oos_ic": oriented_locked_ic,
        "passes": bool(
            len(folds) > 0
            and fold_ratio >= minimum_fold_ratio
            and oriented_locked_ic > 0.0
        ),
    }


def compare_taxonomy_replay(previous: Mapping[str, Any], current: Mapping[str, Any]) -> dict:
    """Compare factor decisions after a taxonomy-triggered full P0 replay."""
    previous_rows = {
        str(row.get("name", "")): row
        for row in previous.get("all_results", [])
        if row.get("name")
    }
    current_rows = {
        str(row.get("name", "")): row
        for row in current.get("all_results", [])
        if row.get("name")
    }
    previous_final = set(map(str, previous.get("final_factors", [])))
    current_final = set(map(str, current.get("final_factors", [])))
    changes = []
    for name in sorted(set(previous_rows) | set(current_rows)):
        old = previous_rows.get(name, {})
        new = current_rows.get(name, {})
        comparison = {
            "factor": name,
            "previous_final": name in previous_final,
            "current_final": name in current_final,
            "previous_local_fdr": bool(
                old.get("hierarchical_fdr_significant", False)
            ),
            "current_local_fdr": bool(
                new.get("hierarchical_fdr_significant", False)
            ),
            "previous_best_period": int(old.get("best_period", 0) or 0),
            "current_best_period": int(new.get("best_period", 0) or 0),
            "previous_best_variant": str(old.get("best_variant", "")),
            "current_best_variant": str(new.get("best_variant", "")),
            "previous_best_ic": float(old.get("best_ic", 0.0) or 0.0),
            "current_best_ic": float(new.get("best_ic", 0.0) or 0.0),
        }
        if (
            comparison["previous_final"] != comparison["current_final"]
            or comparison["previous_local_fdr"] != comparison["current_local_fdr"]
            or comparison["previous_best_period"] != comparison["current_best_period"]
            or comparison["previous_best_variant"] != comparison["current_best_variant"]
            or not math.isclose(
                comparison["previous_best_ic"],
                comparison["current_best_ic"],
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            changes.append(comparison)
    previous_hash = str(
        previous.get("config", {}).get("taxonomy_sha256", "")
    )
    current_hash = str(current.get("config", {}).get("taxonomy_sha256", ""))
    return {
        "previous_taxonomy_sha256": previous_hash,
        "current_taxonomy_sha256": current_hash,
        "taxonomy_changed": previous_hash != current_hash,
        "previous_final_count": len(previous_final),
        "current_final_count": len(current_final),
        "added_final_factors": sorted(current_final - previous_final),
        "removed_final_factors": sorted(previous_final - current_final),
        "factor_differences": changes,
    }
