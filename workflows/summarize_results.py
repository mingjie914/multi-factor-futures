"""Summarize frozen adaptive nested walk-forward outputs without retuning."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from alpha.ols import SectorGroupedOLSModel
from backtest.metrics import compute_all_metrics
from research.statistics import deflated_sharpe_ratio


def _finite_float(value) -> float:
    number = float(value)
    return number if np.isfinite(number) else 0.0


def _serializable(value):
    if isinstance(value, dict):
        return {str(key): _serializable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serializable(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return _finite_float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _weight_concentration(path: Path) -> dict:
    weights = pd.read_csv(path, index_col=0).apply(pd.to_numeric, errors="coerce")
    weights = weights.fillna(0.0)
    absolute = weights.abs()
    gross = absolute.sum(axis=1)
    active = gross > 1e-12
    absolute = absolute.loc[active]
    gross = gross.loc[active]
    if absolute.empty:
        return {"active_days": 0}

    instrument_shares = absolute.div(gross, axis=0)
    sector_weights = pd.DataFrame(index=absolute.index)
    sector_map = SectorGroupedOLSModel._SECTOR_MAP
    for column in absolute.columns:
        sector = sector_map.get(str(column), "other")
        if sector not in sector_weights:
            sector_weights[sector] = 0.0
        sector_weights[sector] = sector_weights[sector] + absolute[column]
    sector_shares = sector_weights.div(gross, axis=0)
    dominant_sector = sector_shares.idxmax(axis=1)

    return {
        "active_days": int(len(absolute)),
        "average_gross_exposure": _finite_float(gross.mean()),
        "average_max_instrument_share": _finite_float(
            instrument_shares.max(axis=1).mean()
        ),
        "p95_max_instrument_share": _finite_float(
            instrument_shares.max(axis=1).quantile(0.95)
        ),
        "average_max_sector_share": _finite_float(sector_shares.max(axis=1).mean()),
        "p95_max_sector_share": _finite_float(
            sector_shares.max(axis=1).quantile(0.95)
        ),
        "average_sector_hhi": _finite_float((sector_shares ** 2).sum(axis=1).mean()),
        "dominant_sector_frequency": {
            str(key): _finite_float(value)
            for key, value in dominant_sector.value_counts(normalize=True).items()
        },
        "average_sector_shares": {
            str(key): _finite_float(value)
            for key, value in sector_shares.mean().sort_values(ascending=False).items()
        },
    }


def _meta_weights(path: Path) -> dict:
    weights = pd.read_csv(path, index_col=0).apply(pd.to_numeric, errors="coerce")
    weights = weights.fillna(0.0).clip(lower=0.0)
    totals = weights.sum(axis=1)
    normalized = weights.loc[totals > 1e-12].div(totals[totals > 1e-12], axis=0)
    if normalized.empty:
        return {}
    return {
        str(column): _finite_float(normalized[column].mean())
        for column in normalized.columns
    }


def _sub_portfolios(portfolio_dir: Path) -> dict:
    output = {}
    root = portfolio_dir / "sub_portfolios"
    for path in sorted(root.glob("*/metrics.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        output[path.parent.name] = {
            "metrics": payload.get("metrics", {}),
            "failure_count": int(payload.get("failure_count", 0)),
        }
    return output


def summarize(run_root: Path) -> dict:
    validation_path = run_root / "walkforward_validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    folds = validation.get("walk_forward", [])
    if not folds or any("error" in fold for fold in folds):
        raise RuntimeError("walk-forward output is missing a complete valid fold")

    all_returns = []
    factor_sets = {}
    fold_details = {}
    for fold in folds:
        segment = str(fold["segment"])
        portfolio_dir = Path(fold["portfolio_path"])
        nav = pd.read_csv(
            portfolio_dir / "combined_nav.csv", index_col=0, parse_dates=True
        ).iloc[:, 0]
        all_returns.append(nav.pct_change().dropna().rename(segment))
        selected = {
            factor
            for factors in fold.get("selected_factors", {}).values()
            for factor in factors
        }
        factor_sets[segment] = selected
        manifest = json.loads(
            (Path(fold["artifact_path"]) / "manifest.json").read_text(encoding="utf-8")
        )
        metrics_payload = json.loads(
            (portfolio_dir / "metrics.json").read_text(encoding="utf-8")
        )
        fold_details[segment] = {
            "train_start": fold["train_start"],
            "train_end": fold["train_end"],
            "test_start": fold["test_start"],
            "test_end": fold["test_end"],
            "actual_test_start": fold["actual_test_start"],
            "actual_test_end": fold["actual_test_end"],
            "research_approved_before_cluster_dedup": int(
                manifest.get("metadata", {}).get("selected_factors", 0)
            ),
            "selected_after_cluster_dedup": int(len(selected)),
            "selected_factors": fold.get("selected_factors", {}),
            "metrics": metrics_payload.get("combined_metrics", {}),
            "dsr": fold.get("dsr", {}),
            "combined_failure_count": int(metrics_payload.get("failure_count", 0)),
            "sub_portfolios": _sub_portfolios(portfolio_dir),
            "average_normalized_meta_weights": _meta_weights(
                portfolio_dir / "meta_weights.csv"
            ),
            "concentration": _weight_concentration(
                portfolio_dir / "underlying_weights.csv"
            ),
        }

    combined_returns = pd.concat(all_returns).sort_index()
    if combined_returns.index.has_duplicates:
        raise RuntimeError("walk-forward return dates overlap across folds")
    combined_nav = (1.0 + combined_returns).cumprod()
    aggregate_metrics = {
        str(key): _finite_float(value)
        for key, value in compute_all_metrics(combined_nav).items()
    }

    segments = list(factor_sets)
    pairwise = {}
    for left_index, left in enumerate(segments):
        for right in segments[left_index + 1:]:
            union = factor_sets[left] | factor_sets[right]
            pairwise[f"{left}|{right}"] = {
                "intersection": sorted(factor_sets[left] & factor_sets[right]),
                "jaccard": _finite_float(
                    len(factor_sets[left] & factor_sets[right]) / len(union)
                    if union else 1.0
                ),
            }
    all_fold_intersection = set.intersection(*factor_sets.values())

    result = {
        "method": {
            "candidate_factor_count": int(folds[0]["candidate_factor_count"]),
            "fdr_method": "hierarchical",
            "alpha_type": str(folds[0]["alpha_type"]),
            "meta_optimizer": "shrinkage_min_variance",
            "cluster_deduplication": True,
            "unmapped_sector_policy": "zero",
        },
        "aggregate_oos": {
            "start": combined_returns.index.min().date().isoformat(),
            "end": combined_returns.index.max().date().isoformat(),
            "observations": int(len(combined_returns)),
            "metrics": aggregate_metrics,
            "dsr": deflated_sharpe_ratio(
                combined_returns,
                n_trials=int(folds[0]["candidate_factor_count"]),
                risk_free_rate=0.0,
            ),
            "positive_fold_count": int(
                sum(float(fold["sharpe"]) > 0.0 for fold in folds)
            ),
            "fold_count": int(len(folds)),
        },
        "factor_stability": {
            "all_fold_intersection": sorted(all_fold_intersection),
            "all_fold_intersection_count": int(len(all_fold_intersection)),
            "pairwise": pairwise,
        },
        "folds": fold_details,
    }
    return _serializable(result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    output = args.output.resolve() if args.output else run_root / "diagnostics.json"
    result = summarize(run_root)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
