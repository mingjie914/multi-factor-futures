"""Pure, diagnostic-first conversion from portfolio weights to integer lots.

The module deliberately has no data access or execution side effects.  Inputs are
validated before any account result is produced; once validated, sizing keeps
route attribution and account level merged targets separate so that a zero net
target remains visible in the diagnostic output.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal, InvalidOperation, ROUND_CEILING, ROUND_DOWN, ROUND_FLOOR, ROUND_HALF_UP
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
_ROUTE_FIELDS = ("strategy_id", "account_id", "capital_basis")
_BASIS = {"equity", "notional", "margin", "reference", "complete_unit"}


def size_targets(
    weight_sets: dict[str, list[dict]],
    specs: list[dict],
    accounts: dict[str, dict],
    routes: list[dict],
    rounding: str = "half_up",
    *,
    specs_as_of: str | None = None,
) -> dict:
    """Convert routed strategy weights to account level integer lot targets.

    Final monetary and lot calculations use ``Decimal``; balanced candidate
    search uses bounded NumPy arrays.  The returned
    structure contains ordinary Python numbers to remain easy to serialize and
    inspect.  Invalid input is rejected with ``ValueError``; valid but
    non-tradable results return complete account diagnostics with blockers.
    """

    if not isinstance(rounding, str) or rounding not in {"half_up", "toward_zero", "balanced"}:
        raise ValueError("rounding must be 'half_up', 'toward_zero' or 'balanced'")
    if not isinstance(weight_sets, dict):
        raise ValueError("weight_sets must be a dict")
    if not isinstance(specs, list):
        raise ValueError("specs must be a list")
    if not isinstance(accounts, dict):
        raise ValueError("accounts must be a dict")
    if not isinstance(routes, list):
        raise ValueError("routes must be a list")
    validated_specs_as_of = None if specs_as_of is None else _date(specs_as_of, "specs_as_of")

    parsed_specs = _parse_specs(specs)
    parsed_weights = _parse_weight_sets(weight_sets, parsed_specs)
    if validated_specs_as_of is not None:
        if any(
            validated_specs_as_of < row["data_date"]
            for rows in parsed_weights.values()
            for row in rows
        ):
            raise ValueError("specs_as_of cannot be earlier than weight data_date")
    parsed_accounts = _parse_accounts(accounts)
    parsed_routes = _parse_routes(routes, parsed_weights, parsed_accounts)

    # A strategy needs a non-zero normalizing weight sum for every route.  This
    # check is deferred until after all structural checks so malformed input
    # still reports its concrete field rather than a secondary arithmetic error.
    for strategy_id, rows in parsed_weights.items():
        if any(row["weight"] != 0 for row in rows) is False:
            if any(route["strategy_id"] == strategy_id for route in parsed_routes):
                raise ValueError(f"strategy {strategy_id!r} has no non-zero weight")

    requirements = _capital_requirements(parsed_weights, parsed_specs, rounding)
    account_results = _size_accounts(
        parsed_weights,
        parsed_specs,
        parsed_accounts,
        parsed_routes,
        rounding,
        requirements,
        validated_specs_as_of,
    )
    return {"accounts": account_results, "capital_requirements": _serialize_capital_requirements(requirements)}


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
            raise ValueError(f"route[{index}].capital_basis must be one of {sorted(_BASIS)}")
        route_key = (strategy_id, account_id, basis)
        if route_key in seen_routes:
            raise ValueError(f"duplicate route for strategy {strategy_id} and account {account_id}")
        seen_routes.add(route_key)
        amount = units = None
        if basis == "complete_unit":
            _missing(row, ("units",), f"route[{index}]")
            if "amount" in row:
                raise ValueError(f"route[{index}] complete_unit uses units, not amount")
            units = _decimal(row["units"], f"route[{index}].units", positive=True)
            if units < 1 or units != units.to_integral_value():
                raise ValueError(f"route[{index}].units must be an integer >= 1")
        else:
            _missing(row, ("amount",), f"route[{index}]")
            if "units" in row:
                raise ValueError(f"route[{index}].units requires complete_unit")
            amount = _decimal(row["amount"], f"route[{index}].amount", positive=True)
        parsed.append(
            {
                "route_index": index,
                "strategy_id": strategy_id,
                "account_id": account_id,
                "capital_basis": basis,
                "amount": amount,
                "units": units,
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
            rounded_value = proportional_value * (Decimal("0.5") if rounding != "toward_zero" else Decimal(1))
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
    mode = ROUND_DOWN if rounding == "toward_zero" else ROUND_HALF_UP
    return value.quantize(Decimal("1"), rounding=mode)


def _balance_rounding(targets, account, specs):
    """Choose adjacent integers closest to theoretical net, then to leg weights.

    Search account targets once, before position reconciliation. Candidate arrays
    are bounded in size; final lots, money and risk checks still use Decimal.
    """
    import numpy as np

    limit = account["equity"] * account["max_abs_net_exposure"]
    raw_net = sum((r["target_notional"] for r in targets), Decimal(0))
    if abs(raw_net) > limit:
        return []  # Do not optimize away a genuine strategy-level risk breach.
    totals = [sum((r["actual_notional"] for r in targets), Decimal(0)),
              sum((abs(r["actual_notional"]) for r in targets), Decimal(0)),
              sum((r["estimated_margin"] for r in targets), Decimal(0)),
              sum((abs(r["actual_notional"] - r["target_notional"]) for r in targets), Decimal(0))]
    choices = []
    for row in sorted(targets, key=lambda r: r["contract"]):
        raw, lots = row["raw_lots"], row["target_lots"]
        position_limit = specs[row["contract"]]["max_position_lots"]
        base_allowed = position_limit is None or abs(lots) <= position_limit
        # A complete-unit capital rounded to cents can produce 1.0000000001 lots.
        if abs(raw) < 1 or abs(raw - raw.to_integral_value()) < Decimal("1e-8"):
            if not base_allowed:
                return []
            continue
        lower = int(raw.to_integral_value(rounding=ROUND_FLOOR))
        other = lower + 1 if lots == lower else lower
        if not other or (other > 0) != (lots > 0):
            continue
        unit = row["close"] * row["multiplier"]
        new_value = other * unit
        gross_delta = abs(new_value) - abs(row["actual_notional"])
        error_delta = abs(new_value - row["target_notional"]) - abs(row["actual_notional"] - row["target_notional"])
        choices.append((row, other, new_value - row["actual_notional"], gross_delta,
                        gross_delta * row["margin_rate"], error_delta, base_allowed,
                        position_limit is None or abs(other) <= position_limit))
    # Exact search is fast for this 20-leg portfolio; never begin unbounded 2**N work.
    if len(choices) > 24:
        raise ValueError("BALANCED_ROUNDING_SEARCH_LIMIT: more than 24 variable contracts")
    if not choices:
        return []
    raw_reference = 0.0 if abs(raw_net) < Decimal("1e-8") else float(raw_net)
    gross_limit = float(account["equity"] * account["max_gross_exposure"])
    margin_limit = float(account["equity"] * account["max_margin_ratio"] - account["frozen_margin"] - account["reserve"])
    cash_limit = float(account["available"] - account["reserve"])
    best = None
    for start in range(0, 1 << len(choices), 65536):
        masks = np.arange(start, min(start + 65536, 1 << len(choices)), dtype=np.uint32)
        net, gross, margin, error = [np.full(len(masks), float(value)) for value in totals]
        allowed = np.ones(len(masks), dtype=bool)
        changes = np.zeros(len(masks), dtype=np.uint8)
        for bit, (_, _, dn, dg, dm, de, base_ok, other_ok) in enumerate(choices):
            use_other = ((masks >> bit) & 1).astype(bool)
            net += use_other * float(dn)
            gross += use_other * float(dg)
            margin += use_other * float(dm)
            error += use_other * float(de)
            changes += use_other
            allowed &= np.where(use_other, other_ok, base_ok)
        allowed &= ((np.abs(net) <= float(limit)) & (gross <= gross_limit)
                    & (margin <= margin_limit)
                    & (np.maximum(0, margin - float(account["margin_used"])) <= cash_limit))
        candidates = np.flatnonzero(allowed)
        if not len(candidates):
            continue
        distances = np.round(np.abs(net[candidates] - raw_reference), 8)
        distance = distances.min()
        candidates = candidates[distances == distance]
        errors = np.round(error[candidates], 8)
        weight_error = errors.min()
        candidates = candidates[errors == weight_error]
        change_count = changes[candidates].min()
        chosen = candidates[changes[candidates] == change_count][0]
        key = (float(distance), float(weight_error), int(change_count), int(masks[chosen]))
        if best is None or key < best:
            best = key
    if best is None:
        return []  # Retain original diagnostics and existing risk blockers.
    adjustments = []
    for bit, (row, other, _, _, _, extra_error, _, _) in enumerate(choices):
        if not (best[3] >> bit) & 1:
            continue
        adjustments.append({"contract": row["contract"], "from_lots": row["target_lots"], "to_lots": other,
                            "additional_abs_notional_error": extra_error})
        row["target_lots"] = other
        row["actual_notional"] = other * row["close"] * row["multiplier"]
        row["estimated_margin"] = abs(row["actual_notional"]) * row["margin_rate"]
        row["weight_error"] = (row["actual_notional"] - row["target_notional"]) / account["equity"]
    return adjustments


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
    requirements: dict[str, dict[str, Any]],
    specs_as_of: str | None = None,
) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, dict[str, Any]]] = {account_id: {} for account_id in accounts}
    attributions: dict[str, list[dict[str, Any]]] = {account_id: [] for account_id in accounts}
    equity_route_amounts: dict[str, Decimal] = {account_id: Decimal(0) for account_id in accounts}
    resolved_routes: dict[str, list[dict]] = {account_id: [] for account_id in accounts}
    strategy_notionals: dict[str, dict[str, dict[str, Decimal]]] = {account_id: {} for account_id in accounts}

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
        minimum = requirements[route["strategy_id"]]["proportional_one_lot_equity"]
        if route["capital_basis"] == "complete_unit":
            # Round the common scaling amount UP to a cent before sizing; never
            # independently raise an underweight leg to one lot.
            reference_capital = minimum.quantize(Decimal("0.01"), rounding=ROUND_CEILING) * route["units"]
        else:
            reference_capital = route["amount"] / denominator
        resolved_routes[account_id].append({**route, "reference_capital": reference_capital,
            "minimum_complete_reference_capital": minimum})
        notionals = strategy_notionals[account_id].setdefault(route["strategy_id"], {})
        if route["capital_basis"] == "equity":
            equity_route_amounts[account_id] += route["amount"]
        for row in strategy_rows:
            spec = specs[row["contract"]]
            if route["capital_basis"] in {"complete_unit", "reference"}:
                target_notional = row["weight"] * reference_capital
            elif route["capital_basis"] == "equity":
                target_notional = row["weight"] * route["amount"]
            else:
                target_notional = row["weight"] * route["amount"] / denominator
            notionals[row["contract"]] = notionals.get(row["contract"], Decimal(0)) + target_notional
            notional_per_lot = row["close"] * spec["multiplier"]
            raw_lots = target_notional / notional_per_lot
            margin_rate = _margin_rate(row["weight"], spec)
            rounded_lots = int(_round_lots(raw_lots, rounding))
            attribution = {
                "route_index": route["route_index"],
                "strategy_id": route["strategy_id"],
                "account_id": account_id,
                "capital_basis": route["capital_basis"],
                "amount": route["amount"],
                "units": route["units"],
                "reference_capital": reference_capital,
                "root": row["root"],
                "contract": row["contract"],
                "weight": row["weight"],
                "target_notional": target_notional,
                "notional": target_notional,
                "data_date": row["data_date"],
                "close": row["close"],
                "multiplier": spec["multiplier"],
                "margin_rate": margin_rate,
                "notional_per_lot": notional_per_lot,
                "margin_per_lot": notional_per_lot * margin_rate,
                "raw_lots": raw_lots,
                # Standalone route diagnostics only; executable targets are
                # still rounded once after the account's notionals are merged.
                "rounded_lots": rounded_lots,
                "independent_rounded_lots": rounded_lots,
                "estimated_margin": abs(rounded_lots) * notional_per_lot * margin_rate,
                "rounded_weight": rounded_lots * notional_per_lot / reference_capital,
                "weight_error": rounded_lots * notional_per_lot / reference_capital - row["weight"],
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
            if not spec["verified"] or spec["as_of"] != (specs_as_of or row["data_date"]):
                target["metadata_unverified"] = True

    results: dict[str, dict[str, Any]] = {}
    for account_id, account in accounts.items():
        targets: list[dict[str, Any]] = []
        blockers: list[str] = []
        completeness = {}
        # Check each strategy before cross-strategy netting. Multiple capital
        # routes for the same strategy are one allocation, so combine them first.
        for strategy_id, notionals in strategy_notionals[account_id].items():
            selected = [row for row in weight_sets[strategy_id] if row["weight"] != 0]
            raw = {row["contract"]: abs(notionals[row["contract"]])
                   / (row["close"] * specs[row["contract"]]["multiplier"]) for row in selected}
            below = [contract for contract, lots in raw.items() if lots < 1]
            completeness[strategy_id] = {"selected_count": len(selected),
                "min_abs_raw_lots": _number(min(raw.values())),
                "below_one_contracts": below, "complete": not below}
            if account["require_one_lot"] and below:
                blockers.append(f"REQUIRE_ONE_LOT: {strategy_id} pre-round lots below one: {', '.join(below)}")
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
                "independent_rounded_lots": target_lots,
                "actual_notional": actual_notional,
                "estimated_margin": estimated_margin,
                "weight_error": weight_error,
                "max_order_lots": spec["max_order_lots"],
                "metadata_verified": not merged_target["metadata_unverified"],
            }
            targets.append(target_row)
        adjustments = _balance_rounding(targets, account, specs) if rounding == "balanced" else []
        for target in targets:
            contract, target_lots = target["contract"], target["target_lots"]
            spec = specs[contract]
            if target_lots and not target["metadata_verified"]:
                blockers.append(f"METADATA_UNVERIFIED: {contract}")
            max_position_lots = spec["max_position_lots"]
            if max_position_lots is not None and abs(target_lots) > max_position_lots:
                blockers.append(
                    f"MAX_POSITION_LOTS: {contract} target {target_lots} exceeds {max_position_lots}"
                )
            contributions = [r for r in attributions[account_id] if r["contract"] == contract]
            if adjustments and len(contributions) == 1:
                row = contributions[0]
                row["rounded_lots"] = target_lots
                row["estimated_margin"] = target["estimated_margin"]
                row["rounded_weight"] = target["actual_notional"] / row["reference_capital"]
                row["weight_error"] = row["rounded_weight"] - row["weight"]

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
            "routes": [_serialize_attribution(row) for row in resolved_routes[account_id]],
            "completeness": completeness,
            "rounding_adjustments": [_serialize_target(row) for row in adjustments],
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
