"""Convert formal target weights to indicative integer contract lots.

This is a hand-off utility, not an order router. Prices and concrete contracts
come from the configured point-in-time Parquet source; no same-day dominant
inference or machine-specific data path is allowed.
"""
from __future__ import annotations

import argparse
from math import isfinite
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import load_config
from data.contract_specs import CONTRACT_SPECS
from data.manager import DataManager


def _contract_snapshot(source, roots, as_of=None) -> tuple[pd.Timestamp, dict]:
    """Return exact scheduled contracts and raw closes for one trading day."""
    roots = tuple(dict.fromkeys(str(root).strip().upper() for root in roots))
    if not roots or any(not root for root in roots):
        raise ValueError("at least one non-empty root is required")

    if as_of is None:
        latest = getattr(source, "fetch_latest_trade_date", None)
        if not callable(latest):
            raise NotImplementedError(
                "configured data source does not expose its latest trade date"
            )
        date = pd.Timestamp(latest()).normalize()
    else:
        date = pd.Timestamp(as_of).normalize()

    calendar = pd.DatetimeIndex(source.fetch_calendar(date, date)).normalize()
    if date not in calendar:
        raise ValueError(f"{date.date()} is not a published trading day")

    schedule = source.fetch_contract_schedule(roots, date, date)
    if not isinstance(schedule, pd.DataFrame):
        raise TypeError("contract schedule must be a DataFrame")
    schedule = schedule.reindex(index=[date], columns=roots)
    curve = source.fetch_contract_curve_at_frequency(
        roots, date, date, ["close"], frequency="daily"
    )
    if not isinstance(curve, pd.DataFrame) or curve.empty:
        raise RuntimeError(f"no concrete contract closes on {date.date()}")
    curve = curve.copy()
    curve["trade_date"] = pd.to_datetime(curve["trade_date"]).dt.normalize()
    curve = curve.loc[curve["trade_date"].eq(date)]
    if curve.duplicated(["root", "symbol"]).any():
        raise RuntimeError("concrete close snapshot contains duplicate contracts")

    snapshot = {}
    for root in roots:
        contract = schedule.at[date, root]
        if pd.isna(contract) or not str(contract).strip():
            raise RuntimeError(
                f"formal contract schedule is missing {root} on {date.date()}"
            )
        contract = str(contract)
        rows = curve.loc[
            curve["root"].eq(root) & curve["symbol"].eq(contract)
        ]
        if len(rows) != 1:
            raise RuntimeError(
                "scheduled contract close is missing or ambiguous: "
                f"{contract}@{date.date()}"
            )
        price = float(rows.iloc[0]["close"])
        if not isfinite(price) or price <= 0.0:
            raise RuntimeError(
                f"scheduled contract close is invalid: {contract}={price}"
            )
        snapshot[root] = {"contract": contract, "price": price}
    return date, snapshot


def _weights_from_signal(date: str, config_path: str):
    from strategies.combined import CombinedStrategy

    strategy = CombinedStrategy(config_path)
    weights = strategy.signal(date)
    selected = {
        str(root).upper(): float(weight)
        for root, weight in weights.items()
        if abs(float(weight)) > 1e-12
    }
    return selected, strategy.data_manager.source


def _parse_weights(value: str) -> dict[str, float]:
    parsed = {}
    for pair in str(value).split(","):
        root, separator, weight = pair.partition(":")
        root = root.strip().upper()
        if not separator or not root:
            raise ValueError(f"invalid weight item: {pair!r}")
        numeric = float(weight)
        if not isfinite(numeric):
            raise ValueError(f"weight must be finite: {pair!r}")
        parsed[root] = numeric
    return parsed


def lots_for(weights: dict, capital: float, snapshot: dict) -> list[dict]:
    capital = float(capital)
    if not isfinite(capital) or capital <= 0.0:
        raise ValueError("capital must be positive and finite")

    rows = []
    for root, weight in sorted(weights.items(), key=lambda item: -abs(item[1])):
        root = str(root).upper()
        weight = float(weight)
        if not isfinite(weight):
            raise ValueError(f"weight must be finite: {root}")
        if abs(weight) <= 1e-12:
            continue
        spec = CONTRACT_SPECS.get(root)
        quote = snapshot.get(root)
        if spec is None:
            raise KeyError(f"contract specification is missing: {root}")
        if quote is None:
            raise KeyError(f"formal contract quote is missing: {root}")

        price = float(quote["price"])
        multiplier = float(spec["multiplier"])
        notional = weight * capital
        raw_lots = abs(notional) / (price * multiplier)
        lots = int(raw_lots)
        actual_notional = lots * price * multiplier * (1 if weight > 0 else -1)
        rows.append({
            "symbol": root,
            "contract": str(quote["contract"]),
            "direction": "多" if weight > 0 else "空",
            "weight": weight,
            "notional": notional,
            "price": price,
            "multiplier": multiplier,
            "unit": spec["unit"],
            "raw_lots": raw_lots,
            "lots": lots,
            "actual_notional": actual_notional,
            "margin": spec.get("margin"),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="正式目标权重→参考合约手数（不发送订单）"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--date", help="固定观察策略的信号日期")
    group.add_argument("--weights", help='手动权重，例如 "TL:0.246,A:0.1891"')
    parser.add_argument(
        "--as-of", default=None,
        help="手动权重的价格日期；省略时使用最新本地交易日",
    )
    parser.add_argument("--capital", type=float, default=1_000_000.0)
    parser.add_argument("--config", default="config/intraday_backtest.yaml")
    args = parser.parse_args()

    if args.date:
        if args.as_of is not None:
            parser.error("--as-of is only valid with --weights")
        weights, source = _weights_from_signal(args.date, args.config)
        as_of = args.date
    else:
        weights = _parse_weights(args.weights)
        manager = DataManager.from_config(load_config(args.config))
        source = manager.source
        as_of = args.as_of

    quote_date, snapshot = _contract_snapshot(source, weights, as_of=as_of)
    rows = lots_for(weights, args.capital, snapshot)
    gross = sum(abs(weight) for weight in weights.values())
    print(
        f"价格日 {quote_date.date()} | 资金 {args.capital:,.0f} 元 | "
        f"目标总杠杆 {gross:.2f}"
    )
    print(
        f'{"品种":<4} {"合约":<8} {"向":<2} {"权重":>7} {"名义(元)":>12} '
        f'{"价格":>10} {"乘数":>7} {"理论手":>7} {"实手":>4} {"保证金(元)":>12}'
    )
    print("-" * 105)

    total_notional = 0.0
    total_margin = 0.0
    for row in rows:
        deviation = (
            (row["actual_notional"] - row["notional"]) / row["notional"] * 100.0
        )
        total_notional += abs(row["actual_notional"])
        margin = row["margin"]
        margin_amount = (
            abs(row["actual_notional"]) * float(margin)
            if margin is not None and row["lots"] > 0 else 0.0
        )
        total_margin += margin_amount
        print(
            f'{row["symbol"]:<4} {row["contract"]:<8} {row["direction"]:<2} '
            f'{row["weight"]:>7.4f} {row["notional"]:>12,.0f} '
            f'{row["price"]:>10.2f} {row["multiplier"]:>7,.0f} '
            f'{row["raw_lots"]:>7.2f} {row["lots"]:>4} '
            f'{margin_amount:>12,.0f}  偏差 {deviation:>6.1f}%'
        )

    print(f"\n实际总名义（多空绝对值）: {total_notional:,.0f} 元")
    print(
        f"实际保证金占用: {total_margin:,.0f} 元 "
        f"(占总资金 {total_margin / args.capital:.1%})"
    )


if __name__ == "__main__":
    main()
