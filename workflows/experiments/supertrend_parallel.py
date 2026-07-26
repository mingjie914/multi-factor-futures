"""Compare a saved multi-factor portfolio with the Supertrend rule sleeve."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.metrics import compute_all_metrics, compute_split_metrics


FIXED_SUPERTREND_WEIGHTS = (0.10, 0.20, 0.30)


def _read_series(path: Path, preferred: str) -> pd.Series:
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    column = preferred if preferred in frame.columns else frame.columns[0]
    return frame[column].astype(float).sort_index()


def _walk_forward_weights(
    returns: pd.DataFrame,
    *,
    reweight_freq: int = 20,
    estimation_window: int = 252,
    shrinkage: float = 0.30,
    min_supertrend_weight: float = 0.10,
    max_supertrend_weight: float = 0.30,
) -> pd.DataFrame:
    """Point-in-time two-sleeve minimum-variance weights."""
    current_supertrend = 0.20
    rows = np.empty((len(returns), 2), dtype=float)
    last_reweight = -reweight_freq

    for index, date in enumerate(returns.index):
        if index >= 20 and index - last_reweight >= reweight_freq:
            history = returns.iloc[max(0, index - estimation_window):index]
            covariance = history.cov().to_numpy(dtype=float) * 252.0
            covariance = np.nan_to_num(
                covariance, nan=0.0, posinf=0.0, neginf=0.0
            )
            diagonal = np.maximum(np.diag(covariance), 1e-8)
            covariance = (
                (1.0 - shrinkage) * covariance
                + shrinkage * np.diag(diagonal)
            )
            covariance = (covariance + covariance.T) / 2.0
            try:
                raw = np.linalg.solve(covariance, np.ones(2, dtype=float))
                raw /= raw.sum()
                target_supertrend = float(np.clip(
                    raw[1], min_supertrend_weight, max_supertrend_weight
                ))
                current_supertrend = (
                    0.5 * target_supertrend + 0.5 * current_supertrend
                )
            except (np.linalg.LinAlgError, ValueError, FloatingPointError):
                pass
            last_reweight = index
        rows[index] = [1.0 - current_supertrend, current_supertrend]

    return pd.DataFrame(
        rows, index=returns.index, columns=["multi_factor", "supertrend"]
    )


def _evaluate_variant(
    returns: pd.DataFrame,
    costs: pd.DataFrame,
    weights: pd.DataFrame,
    *,
    allocation_cost_rate: float = 0.0011,
) -> tuple[pd.Series, dict]:
    weights = weights.reindex(returns.index).ffill()
    allocation_turnover = weights.diff().abs().sum(axis=1).fillna(0.0)
    allocation_cost = allocation_turnover * allocation_cost_rate
    net_return = (returns * weights).sum(axis=1) - allocation_cost
    nav = (1.0 + net_return).cumprod()
    metrics = compute_all_metrics(nav, returns=net_return)
    metrics["return_correlation"] = float(
        returns["multi_factor"].corr(returns["supertrend"])
    )
    metrics["embedded_cost_proxy"] = float((costs * weights).sum().sum())
    metrics["allocation_cost_proxy"] = float(allocation_cost.sum())
    metrics["total_cost_proxy"] = (
        metrics["embedded_cost_proxy"] + metrics["allocation_cost_proxy"]
    )
    metrics["avg_supertrend_weight"] = float(weights["supertrend"].mean())
    metrics["min_supertrend_weight"] = float(weights["supertrend"].min())
    metrics["max_supertrend_weight"] = float(weights["supertrend"].max())
    metrics["split_diagnostic"] = compute_split_metrics(
        nav, returns=net_return, train_ratio=0.6
    )
    return nav, metrics


def analyse_segment(
    baseline_dir: Path,
    supertrend_dir: Path,
    output_dir: Path,
) -> list[dict]:
    base_nav = _read_series(baseline_dir / "combined_nav.csv", "nav")
    trend_nav = _read_series(supertrend_dir / "nav.csv", "nav")
    base_cost = _read_series(baseline_dir / "costs.csv", "cost")
    trend_cost = _read_series(supertrend_dir / "costs.csv", "cost")
    returns = pd.concat(
        {
            "multi_factor": base_nav.pct_change(fill_method=None),
            "supertrend": trend_nav.pct_change(fill_method=None),
        },
        axis=1,
        join="inner",
    ).dropna()
    costs = pd.concat(
        {"multi_factor": base_cost, "supertrend": trend_cost},
        axis=1,
    ).reindex(returns.index).fillna(0.0)
    if len(returns) < 20:
        raise ValueError(f"insufficient overlapping returns in {baseline_dir}")

    variants = {
        "baseline_100": pd.DataFrame(
            {"multi_factor": 1.0, "supertrend": 0.0}, index=returns.index
        )
    }
    for trend_weight in FIXED_SUPERTREND_WEIGHTS:
        variants[f"fixed_{int((1-trend_weight)*100)}_{int(trend_weight*100)}"] = (
            pd.DataFrame(
                {
                    "multi_factor": 1.0 - trend_weight,
                    "supertrend": trend_weight,
                },
                index=returns.index,
            )
        )
    variants["walkforward_minvar_10_30"] = _walk_forward_weights(returns)

    output_dir.mkdir(parents=True, exist_ok=False)
    navs = {}
    rows = []
    for name, weights in variants.items():
        nav, metrics = _evaluate_variant(returns, costs, weights)
        navs[name] = nav
        weights.to_csv(output_dir / f"weights_{name}.csv")
        row = {
            "variant": name,
            "start": str(returns.index[0].date()),
            "end": str(returns.index[-1].date()),
            "n_days": len(returns),
            **{key: value for key, value in metrics.items() if key != "split_diagnostic"},
            "late_segment_sharpe": metrics["split_diagnostic"].get("test", {}).get(
                "sharpe", 0.0
            ),
        }
        rows.append(row)

    comparison = pd.DataFrame(rows).set_index("variant")
    comparison.to_csv(output_dir / "comparison.csv", encoding="utf-8-sig")
    pd.DataFrame(navs).to_csv(output_dir / "navs.csv")
    (output_dir / "comparison.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        action="append",
        required=True,
        help="Repeat as LABEL=PATH for each saved historical segment",
    )
    parser.add_argument("--supertrend-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=False)
    all_rows = []
    for value in args.baseline:
        if "=" not in value:
            raise ValueError("--baseline must use LABEL=PATH")
        label, path = value.split("=", 1)
        rows = analyse_segment(
            Path(path), Path(args.supertrend_dir), root / label
        )
        all_rows.extend({"segment": label, **row} for row in rows)

    full = pd.DataFrame(all_rows)
    full.to_csv(root / "all_segments.csv", index=False, encoding="utf-8-sig")
    summary = (
        full.groupby("variant")
        .agg(
            mean_annual_return=("annual_return", "mean"),
            mean_sharpe=("sharpe", "mean"),
            worst_sharpe=("sharpe", "min"),
            mean_max_drawdown=("max_drawdown", "mean"),
            worst_max_drawdown=("max_drawdown", "min"),
            mean_late_segment_sharpe=("late_segment_sharpe", "mean"),
            mean_supertrend_weight=("avg_supertrend_weight", "mean"),
            mean_total_cost_proxy=("total_cost_proxy", "mean"),
        )
        .sort_values(["worst_sharpe", "mean_sharpe"], ascending=False)
    )
    summary.to_csv(root / "summary.csv", encoding="utf-8-sig")
    (root / "methodology.json").write_text(
        json.dumps(
            {
                "evidence_level": "historical_non_independent_diagnostic",
                "fixed_supertrend_weights": list(FIXED_SUPERTREND_WEIGHTS),
                "dynamic_method": "20-day point-in-time shrinkage minimum variance",
                "dynamic_supertrend_bounds": [0.10, 0.30],
                "estimation_window": 252,
                "covariance_shrinkage": 0.30,
                "allocation_cost_rate": 0.0011,
                "cost_note": (
                    "Sleeve costs stay embedded; dynamic sleeve-weight turnover "
                    "pays an additional conservative cost proxy without cross-sleeve netting."
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(float_format=lambda value: f"{value:.6f}"))


if __name__ == "__main__":
    main()
