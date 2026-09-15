"""Pure, diagnostic-first conversion from portfolio weights to integer lots.

The module deliberately has no data access or execution side effects.  Inputs are
validated before any account result is produced; once validated, sizing keeps
route attribution and account level merged targets separate so that a zero net
target remains visible in the diagnostic output.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal, InvalidOperation, ROUND_DOWN, ROUND_HALF_UP
import numbers
import re
from typing import Any, Mapping


_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_CONTRACT_RE = re.compile(r"^([A-Z]+)(\d{4})$")
_WEIGHT_FIELDS = ("root", "contract", "close", "weight", "data_date")
_SPEC_FIELDS = (
    "contract",
    "exchange",
    "multiplier",
    "margin_long",
    "margin_short",
    "tick",
    "as_of",
    "source",
    "verified",
)
_ACCOUNT_FIELDS = (
    "equity",
    "available",
    "margin_used",
    "frozen_margin",
    "reserve",
    "max_margin_ratio",
    "max_gross_exposure",
    "max_abs_net_exposure",
    "require_one_lot",
)
_ROUTE_FIELDS = ("strategy_id", "account_id", "capital_basis", "amount")
_BASIS = {"equity", "notional", "margin"}


def size_targets(
    weight_sets: dict[str, list[dict]],
    specs: list[dict],
    accounts: dict[str, dict],
    routes: list[dict],
    rounding: str = "half_up",
) -> dict:
    """Convert routed strategy weights to account level integer lot targets.

    ``Decimal`` is used for every monetary and lot calculation.  The returned
    structure contains ordinary Python numbers to remain easy to serialize and
    inspect.  Invalid input is rejected with ``ValueError``; valid but
    non-tradable results return complete account diagnostics with blockers.
    """

    if not isinstance(rounding, str) or rounding not in {"half_up", "toward_zero"}:
        raise ValueError("rounding must be 'half_up' or 'toward_zero'")
    if not isinstance(weight_sets, dict):
        raise ValueError("weight_sets must be a dict")
    if not isinstance(specs, list):
        raise ValueError("specs must be a list")
    if not isinstance(accounts, dict):
        raise ValueError("accounts must be a dict")
    if not isinstance(routes, list):
        raise ValueError("routes must be a list")

    parsed_specs = _parse_specs(specs)
    parsed_weights = _parse_weight_sets(weight_sets, parsed_specs)
    parsed_accounts = _parse_accounts(accounts)
    parsed_routes = _parse_routes(routes, parsed_weights, parsed_accounts)

    # A strategy needs a non-zero normalizing weight sum for every route.  This
    # check is deferred until after all structural checks so malformed input
    # still reports its concrete field rather than a secondary arithmetic error.
    for strategy_id, rows in parsed_weights.items():
        if any(row["weight"] != 0 for row in rows) is False:
            if any(route["strategy_id"] == strategy_id for route in parsed_routes):
                raise ValueError(f"strategy {strategy_id!r} has no non-zero weight")

    capital_requirements = _serialize_capital_requirements(
        _capital_requirements(parsed_weights, parsed_specs, rounding)
    )
    account_results = _size_accounts(
        parsed_weights,
        parsed_specs,
        parsed_accounts,
        parsed_routes,
        rounding,
    )
    return {"accounts": account_results, "capital_requirements": capital_requirements}


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _missing(row: Mapping[str, Any], fields: tuple[str, ...], label: str) -> None:
    missing = [field for field in fields if field not in row]
    if missing:
        raise ValueError(f"{label} missing required field(s): {', '.join(missing)}")


def _decimal(value: Any, field: str, *, positive: bool = False, nonnegative: bool = False) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (numbers.Real, Decimal)):
        raise ValueError(f"{field} must be a finite numeric value")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError(f"{field} must be a finite numeric value") from exc
    if not result.is_finite():
        raise ValueError(f"{field} must be a finite numeric value")
    if positive and result <= 0:
        raise ValueError(f"{field} must be positive")
    if nonnegative and result < 0:
        raise ValueError(f"{field} must be non-negative")
    return result


def _date(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _DATE_RE.fullmatch(value):
        raise ValueError(f"{field} must be YYYY-MM-DD")
    try:
        date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be YYYY-MM-DD") from exc
    return value


def _contract(value: Any, field: str = "contract") -> tuple[str, str]:
    if not isinstance(value, str) or value != value.upper():
        raise ValueError(f"{field} must be an uppercase concrete contract")
    match = _CONTRACT_RE.fullmatch(value)
    if match is None:
        raise ValueError(f"{field} must be an uppercase ROOT+YYMM concrete contract")
    root, suffix = match.groups()
    if suffix in {"8888", "9999"}:
        raise ValueError(f"{field} continuous code is not allowed")
    month = int(suffix[2:])
    if month < 1 or month > 12:
        raise ValueError(f"{field} has an invalid delivery month")
    return root, value


def _parse_specs(specs: list[dict]) -> dict[str, dict[str, Any]]:
    parsed: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(specs):
        row = _require_mapping(raw, f"spec[{index}]")
        _missing(row, _SPEC_FIELDS, f"spec[{index}]")
        _, contract = _contract(row["contract"], f"spec[{index}].contract")
        if contract in parsed:
            raise ValueError(f"duplicate spec for contract {contract}")
        exchange = row["exchange"]
        source = row["source"]
        if not isinstance(exchange, str) or not exchange:
            raise ValueError(f"spec[{index}].exchange must be a non-empty string")
        if not isinstance(source, str) or not source:
            raise ValueError(f"spec[{index}].source must be a non-empty string")
        if not isinstance(row["verified"], bool):
            raise ValueError(f"spec[{index}].verified must be bool")
        verified = row["verified"]
        tick = (
            None
            if row["tick"] is None and not verified
            else _decimal(row["tick"], f"spec[{index}].tick", positive=True)
        )
        max_order_lots: int | None = None
        if "max_order_lots" in row:
            value = _decimal(row["max_order_lots"], f"spec[{index}].max_order_lots", positive=True)
            if value != value.to_integral_value():
                raise ValueError(f"spec[{index}].max_order_lots must be an integer")
            max_order_lots = int(value)
        max_position_lots: int | None = None
        if "max_position_lots" in row:
            value = _decimal(row["max_position_lots"], f"spec[{index}].max_position_lots", positive=True)
            if value != value.to_integral_value():
                raise ValueError(f"spec[{index}].max_position_lots must be an integer")
            max_position_lots = int(value)
        parsed[contract] = {
            "contract": contract,
            "exchange": exchange,
            "multiplier": _decimal(row["multiplier"], f"spec[{index}].multiplier", positive=True),
            "margin_long": _decimal(row["margin_long"], f"spec[{index}].margin_long", positive=True),
            "margin_short": _decimal(row["margin_short"], f"spec[{index}].margin_short", positive=True),
            "tick": tick,
            "as_of": _date(row["as_of"], f"spec[{index}].as_of"),
            "source": source,
            "verified": verified,
            "max_order_lots": max_order_lots,
            "max_position_lots": max_position_lots,
        }
    return parsed


def _parse_weight_sets(
    weight_sets: dict[str, list[dict]], specs: dict[str, dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    parsed: dict[str, list[dict[str, Any]]] = {}
    seen_prices: dict[str, Decimal] = {}
    for strategy_id, raw_rows in weight_sets.items():
        if not isinstance(strategy_id, str) or not strategy_id:
            raise ValueError("strategy_id must be a non-empty string")
        if not isinstance(raw_rows, list) or not raw_rows:
            raise ValueError(f"strategy {strategy_id!r} must have a non-empty row list")
        rows: list[dict[str, Any]] = []
        seen_contracts: set[str] = set()
        for index, raw in enumerate(raw_rows):
            row = _require_mapping(raw, f"weight[{strategy_id}][{index}]")
            _missing(row, _WEIGHT_FIELDS, f"weight[{strategy_id}][{index}]")
            root, contract = _contract(row["contract"], f"weight[{strategy_id}][{index}].contract")
            if row["root"] != root or not isinstance(row["root"], str) or row["root"] != row["root"].upper():
                raise ValueError(f"weight[{strategy_id}][{index}].root does not match contract")
            if contract in seen_contracts:
                raise ValueError(f"duplicate weight contract {contract} in strategy {strategy_id}")
            seen_contracts.add(contract)
            close = _decimal(row["close"], f"weight[{strategy_id}][{index}].close", positive=True)
            weight = _decimal(row["weight"], f"weight[{strategy_id}][{index}].weight")
            data_date = _date(row["data_date"], f"weight[{strategy_id}][{index}].data_date")
            if contract not in specs:
                raise ValueError(f"missing spec for contract {contract}")
            if "exchange" in row:
                exchange = row["exchange"]
                if not isinstance(exchange, str) or not exchange:
                    raise ValueError(
                        f"weight[{strategy_id}][{index}].exchange must be a non-empty string"
                    )
                if exchange != specs[contract]["exchange"]:
                    raise ValueError(
                        f"exchange conflict for contract {contract}: "
                        f"weight has {exchange}, spec has {specs[contract]['exchange']}"
                    )
            prior_price = seen_prices.get(contract)
            if prior_price is not None and prior_price != close:
                raise ValueError(f"price conflict for contract {contract}")
            seen_prices[contract] = close
            rows.append(
                {
                    "root": root,
                    "contract": contract,
                    "close": close,
                    "weight": weight,
                    "data_date": data_date,
                }
            )
        parsed[strategy_id] = rows
    data_dates = {row["data_date"] for rows in parsed.values() for row in rows}
    if len(data_dates) > 1:
        raise ValueError("all weight rows in one sizing run must use one data_date")
    return parsed


def _parse_accounts(accounts: dict[str, dict]) -> dict[str, dict[str, Any]]:
    parsed: dict[str, dict[str, Any]] = {}
    for account_id, raw in accounts.items():
        if not isinstance(account_id, str) or not account_id:
            raise ValueError("account_id must be a non-empty string")
        row = _require_mapping(raw, f"account[{account_id}]")
        _missing(row, _ACCOUNT_FIELDS, f"account[{account_id}]")
        values = {
            "equity": _decimal(row["equity"], f"account[{account_id}].equity", positive=True),
            "available": _decimal(row["available"], f"account[{account_id}].available", nonnegative=True),
            "margin_used": _decimal(row["margin_used"], f"account[{account_id}].margin_used", nonnegative=True),
            "frozen_margin": _decimal(row["frozen_margin"], f"account[{account_id}].frozen_margin", nonnegative=True),
            "reserve": _decimal(row["reserve"], f"account[{account_id}].reserve", nonnegative=True),
            "max_margin_ratio": _decimal(
                row["max_margin_ratio"], f"account[{account_id}].max_margin_ratio", positive=True
            ),
            "max_gross_exposure": _decimal(
                row["max_gross_exposure"], f"account[{account_id}].max_gross_exposure", positive=True
            ),
            "max_abs_net_exposure": _decimal(
                row["max_abs_net_exposure"], f"account[{account_id}].max_abs_net_exposure", positive=True
            ),
        }
        if values["max_margin_ratio"] > 1:
            raise ValueError(f"account[{account_id}].max_margin_ratio must be <= 1")
        if not isinstance(row["require_one_lot"], bool):
            raise ValueError(f"account[{account_id}].require_one_lot must be bool")
        values["require_one_lot"] = row["require_one_lot"]
        values["original"] = dict(row)
        parsed[account_id] = values
    return parsed


def _parse_routes(
    routes: list[dict],
    weight_sets: dict[str, list[dict[str, Any]]],
    accounts: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    parsed: list[dict[str, Any]] = []
    seen_routes: set[tuple[str, str, str]] = set()
    for index, raw in enumerate(routes):
        row = _require_mapping(raw, f"route[{index}]")
        _missing(row, _ROUTE_FIELDS, f"route[{index}]")
        strategy_id = row["strategy_id"]
        account_id = row["account_id"]
        if not isinstance(strategy_id, str) or strategy_id not in weight_sets:
            raise ValueError(f"route[{index}] references missing strategy {strategy_id!r}")
        if not isinstance(account_id, str) or account_id not in accounts:
            raise ValueError(f"route[{index}] references missing account {account_id!r}")
        basis = row["capital_basis"]
        if not isinstance(basis, str) or basis not in _BASIS:
            raise ValueError(f"route[{index}].capital_basis must be equity, notional, or margin")
        route_key = (strategy_id, account_id, basis)
        if route_key in seen_routes:
            raise ValueError(f"duplicate route for strategy {strategy_id} and account {account_id}")
        seen_routes.add(route_key)
        amount = _decimal(row["amount"], f"route[{index}].amount", positive=True)
        parsed.append(
            {
                "route_index": index,
                "strategy_id": strategy_id,
                "account_id": account_id,
                "capital_basis": basis,
                "amount": amount,
            }
        )
    if not parsed:
        raise ValueError("routes must not be empty")
    routed_accounts = {route["account_id"] for route in parsed}
    missing_accounts = sorted(set(accounts) - routed_accounts)
    if missing_accounts:
        raise ValueError(
            "every configured account requires an explicit route; missing route for "
            + ", ".join(missing_accounts)
        )
    return parsed


def _margin_rate(weight: Decimal, spec: dict[str, Any]) -> Decimal:
    return spec["margin_long"] if weight >= 0 else spec["margin_short"]


def _capital_requirements(
    weight_sets: dict[str, list[dict[str, Any]]],
    specs: dict[str, dict[str, Any]],
    rounding: str,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for strategy_id, rows in weight_sets.items():
        output_rows: list[dict[str, Any]] = []
        proportional: list[Decimal] = []
        rounded: list[Decimal] = []
        for row in rows:
            spec = specs[row["contract"]]
            notional_per_lot = row["close"] * spec["multiplier"]
            if row["weight"] == 0:
                output_rows.append(
                    {
                        "root": row["root"],
                        "contract": row["contract"],
                        "weight": row["weight"],
                        "notional_per_lot": notional_per_lot,
                        "proportional_one_lot_equity": None,
                        "rounded_one_lot_equity": None,
                    }
                )
                continue
            proportional_value = notional_per_lot / abs(row["weight"])
            rounded_value = proportional_value * (Decimal("0.5") if rounding == "half_up" else Decimal(1))
            proportional.append(proportional_value)
            rounded.append(rounded_value)
            output_rows.append(
                {
                    "root": row["root"],
                    "contract": row["contract"],
                    "weight": row["weight"],
                    "notional_per_lot": notional_per_lot,
                    "proportional_one_lot_equity": proportional_value,
                    "rounded_one_lot_equity": rounded_value,
                }
            )
        if not proportional:
            # Unrouted all-zero strategies are still represented in the
            # diagnostics, but there is no meaningful one-lot requirement.
            result[strategy_id] = {
                "proportional_one_lot_equity": None,
                "rounded_one_lot_equity": None,
                "rows": output_rows,
            }
        else:
            result[strategy_id] = {
                "proportional_one_lot_equity": max(proportional),
                "rounded_one_lot_equity": max(rounded),
                "rows": output_rows,
            }
    return result


def _serialize_capital_requirements(
    values: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for strategy_id, requirement in values.items():
        result[strategy_id] = {
            "proportional_one_lot_equity": _number(requirement["proportional_one_lot_equity"]),
            "rounded_one_lot_equity": _number(requirement["rounded_one_lot_equity"]),
            "rows": [
                {
                    key: _number(value) if isinstance(value, Decimal) else value
                    for key, value in row.items()
                }
                for row in requirement["rows"]
            ],
        }
    return result


def _round_lots(value: Decimal, rounding: str) -> Decimal:
    mode = ROUND_HALF_UP if rounding == "half_up" else ROUND_DOWN
    return value.quantize(Decimal("1"), rounding=mode)


def _number(value: Decimal | int | float | None) -> int | float | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value
    if value == value.to_integral_value():
        return int(value)
    return float(value)


def _size_accounts(
    weight_sets: dict[str, list[dict[str, Any]]],
    specs: dict[str, dict[str, Any]],
    accounts: dict[str, dict[str, Any]],
    routes: list[dict[str, Any]],
    rounding: str,
) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, dict[str, Any]]] = {account_id: {} for account_id in accounts}
    attributions: dict[str, list[dict[str, Any]]] = {account_id: [] for account_id in accounts}
    equity_route_amounts: dict[str, Decimal] = {account_id: Decimal(0) for account_id in accounts}

    for route in routes:
        strategy_rows = weight_sets[route["strategy_id"]]
        abs_weight_sum = sum((abs(row["weight"]) for row in strategy_rows), Decimal(0))
        if route["capital_basis"] == "notional":
            denominator = abs_weight_sum
        elif route["capital_basis"] == "margin":
            denominator = sum(
                (abs(row["weight"]) * _margin_rate(row["weight"], specs[row["contract"]]) for row in strategy_rows),
                Decimal(0),
            )
        else:
            denominator = Decimal(1)
        if denominator <= 0:
            raise ValueError(f"strategy {route['strategy_id']!r} has no usable budget denominator")

        account_id = route["account_id"]
        if route["capital_basis"] == "equity":
            equity_route_amounts[account_id] += route["amount"]
        for row in strategy_rows:
            spec = specs[row["contract"]]
            if route["capital_basis"] == "equity":
                target_notional = row["weight"] * route["amount"]
            else:
                target_notional = row["weight"] * route["amount"] / denominator
            attribution = {
                "route_index": route["route_index"],
                "strategy_id": route["strategy_id"],
                "account_id": account_id,
                "capital_basis": route["capital_basis"],
                "amount": route["amount"],
                "root": row["root"],
                "contract": row["contract"],
                "weight": row["weight"],
                "target_notional": target_notional,
                "notional": target_notional,
            }
            attributions[account_id].append(attribution)
            target = merged[account_id].setdefault(
                row["contract"],
                {
                    "root": row["root"],
                    "contract": row["contract"],
                    "exchange": spec["exchange"],
                    "close": row["close"],
                    "multiplier": spec["multiplier"],
                    "margin_long": spec["margin_long"],
                    "margin_short": spec["margin_short"],
                    "tick": spec["tick"],
                    "target_notional": Decimal(0),
                    "metadata_unverified": False,
                    "data_dates": set(),
                },
            )
            target["target_notional"] += target_notional
            target["data_dates"].add(row["data_date"])
            if not spec["verified"] or spec["as_of"] != row["data_date"]:
                target["metadata_unverified"] = True

    results: dict[str, dict[str, Any]] = {}
    for account_id, account in accounts.items():
        targets: list[dict[str, Any]] = []
        blockers: list[str] = []
        for contract, merged_target in merged[account_id].items():
            spec = specs[contract]
            desired = merged_target["target_notional"]
            notional_per_lot = merged_target["close"] * merged_target["multiplier"]
            raw_lots = desired / notional_per_lot
            target_lots_decimal = _round_lots(raw_lots, rounding)
            target_lots = int(target_lots_decimal)
            actual_notional = target_lots_decimal * notional_per_lot
            margin_rate = (
                merged_target["margin_long"] if desired >= 0 else merged_target["margin_short"]
            )
            estimated_margin = abs(actual_notional) * margin_rate
            equity = account["equity"]
            if equity == 0:
                weight_error: Decimal | None = None if desired != 0 or actual_notional != 0 else Decimal(0)
            else:
                weight_error = actual_notional / equity - desired / equity
            target_row = {
                "root": merged_target["root"],
                "contract": contract,
                "exchange": merged_target["exchange"],
                "close": merged_target["close"],
                "multiplier": merged_target["multiplier"],
                "margin_rate": margin_rate,
                "tick": merged_target["tick"],
                "target_notional": desired,
                "raw_lots": raw_lots,
                "target_lots": target_lots,
                "actual_notional": actual_notional,
                "estimated_margin": estimated_margin,
                "weight_error": weight_error,
                "max_order_lots": spec["max_order_lots"],
            }
            targets.append(target_row)
            if merged_target["metadata_unverified"]:
                blockers.append(f"METADATA_UNVERIFIED: {contract}")
            if account["require_one_lot"] and desired != 0 and target_lots == 0:
                blockers.append(f"REQUIRE_ONE_LOT: {contract} non-zero target rounded to zero")
            max_position_lots = spec["max_position_lots"]
            if max_position_lots is not None and abs(target_lots) > max_position_lots:
                blockers.append(
                    f"MAX_POSITION_LOTS: {contract} target {target_lots} exceeds {max_position_lots}"
                )

        gross_notional = sum((abs(row["actual_notional"]) for row in targets), Decimal(0))
        net_notional = sum((row["actual_notional"] for row in targets), Decimal(0))
        estimated_margin = sum((row["estimated_margin"] for row in targets), Decimal(0))
        gross_exposure = gross_notional / account["equity"]
        net_exposure = net_notional / account["equity"]
        summary = {
            "gross_notional": gross_notional,
            "net_notional": net_notional,
            "estimated_margin": estimated_margin,
            "gross_exposure": gross_exposure,
            "net_exposure": net_exposure,
        }

        if gross_exposure > account["max_gross_exposure"]:
            blockers.append(
                f"GROSS_EXPOSURE: {gross_exposure} exceeds {account['max_gross_exposure']}"
            )
        if equity_route_amounts[account_id] > account["equity"]:
            blockers.append(
                "EQUITY_ROUTE_BUDGET: equity route amounts "
                f"{equity_route_amounts[account_id]} exceed equity {account['equity']}"
            )
        if abs(net_exposure) > account["max_abs_net_exposure"]:
            blockers.append(
                "NET_EXPOSURE: absolute net exposure "
                f"{abs(net_exposure)} exceeds {account['max_abs_net_exposure']}"
            )
        margin_limit = account["equity"] * account["max_margin_ratio"]
        if estimated_margin + account["frozen_margin"] + account["reserve"] > margin_limit:
            blockers.append(
                "MARGIN: estimated margin + frozen margin + reserve "
                f"{estimated_margin + account['frozen_margin'] + account['reserve']} "
                f"exceeds equity margin limit {margin_limit}"
            )
        required_available = max(Decimal(0), estimated_margin - account["margin_used"])
        # The broker's available balance already excludes frozen margin.  Add
        # only the new margin requirement and configured reserve here; frozen
        # margin remains part of the independent total margin cap above.
        required_available += account["reserve"]
        if required_available > account["available"]:
            blockers.append(
                "AVAILABLE: estimated incremental margin + reserve "
                f"{required_available} exceeds available {account['available']}"
            )

        results[account_id] = {
            "targets": [_serialize_target(row) for row in targets],
            "attribution": [_serialize_attribution(row) for row in attributions[account_id]],
            "blockers": _dedupe(blockers),
            "tradable": not blockers,
            "summary": {key: _number(value) for key, value in summary.items()},
            "account": dict(account["original"]),
        }
    return results


def _serialize_target(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: _number(value) if isinstance(value, Decimal) or value is None else value
        for key, value in row.items()
    }


def _serialize_attribution(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: _number(value) if isinstance(value, Decimal) else value
        for key, value in row.items()
    }


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result
