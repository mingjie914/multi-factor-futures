"""Generate catalog-bound close weights with the existing native portfolio method."""
from __future__ import annotations

from datetime import date, datetime
import hashlib
from importlib.metadata import version
from importlib.util import find_spec
import json
import platform
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from core.config import load_config, load_strategy_library
from data.manager import DataManager
from factors.numerics import factor_kernel_contract
from research.effective_factor_library import validate_effective_factor_membership
from research.historical_portfolio_search import PortfolioEvaluator, PortfolioRecipe
from research.portfolio_experiment_support import FactorPanelRunner
from scripts.contract_lots import _contract_snapshot
from trading.artifacts import digest, file_hash, read_artifact, write_artifact, resolve


ROOT = Path(__file__).resolve().parents[1]


def require_closed_date(as_of: str, latest: str, *, now=None) -> None:
    decision = date.fromisoformat(as_of)
    if decision > date.fromisoformat(latest):
        raise ValueError("decision date exceeds published data")
    now = now or datetime.now(ZoneInfo("Asia/Shanghai"))
    if decision > now.date() or (decision == now.date() and now.hour < 16):
        raise ValueError("decision requires completed close (Shanghai 16:00 or later)")


def strategy_contracts(strategy_ids, catalog_path="config/strategy_library.yaml") -> list[dict]:
    if not strategy_ids or len(set(strategy_ids)) != len(strategy_ids):
        raise ValueError("explicit unique strategy IDs are required")
    catalog = load_strategy_library(resolve(catalog_path))
    entries = {s.id: s for s in catalog.strategies}
    subsets = {s.id: s for s in catalog.factor_sets}
    library = resolve(catalog.effective_factor_library)
    records = json.loads(library.read_text(encoding="utf-8"))["factors"]
    directions = {r["factor"]: int(r["direction"]) for r in records}
    default = load_config(resolve("config/default.yaml"))
    result = []
    instances = strategy_ids if isinstance(strategy_ids, dict) else {sid: sid for sid in strategy_ids}
    for instance_id, strategy_id in instances.items():
        if strategy_id == "@formal":
            approved = [s.id for s in catalog.strategies if s.formal]
            if len(approved) != 1:
                raise ValueError("@formal requires exactly one formal strategy; select explicit IDs for multiple strategies")
            strategy_id = approved[0]
        if strategy_id not in entries:
            raise ValueError(f"unknown strategy: {strategy_id}")
        entry = entries[strategy_id]
        if entry.status == "archived" or entry.source != "effective_library":
            raise ValueError(f"archived/non-effective strategy cannot deploy: {strategy_id}")
        subset = subsets[entry.factor_set_id]
        if subset.status != "active":
            raise ValueError(f"inactive factor set: {entry.factor_set_id}")
        config = load_config(resolve(entry.config_path))
        if (config.data.source != "duckdb_futures" or entry.mode != "single"
                or config.production_portfolio != default.production_portfolio):
            raise ValueError("deployment requires the shared native daily production recipe")
        factors = list(subset.factors)
        validate_effective_factor_membership(library, {1: factors})
        frozen = {name: directions[name] for name in factors}
        if frozen != subset.selection_context.get("directions"):
            raise ValueError(f"frozen factor directions changed: {strategy_id}")
        if entry.rank_exit_buffer is not None:
            config.production_portfolio.rank_exit_buffer = entry.rank_exit_buffer
        result.append({"strategy_id": instance_id, "catalog_strategy_id": strategy_id,
                       "formal": bool(entry.formal),
                       "name": entry.name, "factors": factors,
                       "directions": frozen, "recipe": config.production_portfolio.model_dump(),
                       "panel_start": str((pd.Timestamp(default.date_range.start)
                                           - pd.Timedelta(days=365)).date()),
                       "config_path": str(resolve(entry.config_path)),
                       "factor_library_sha256": file_hash(library)})
    return result


def code_fingerprint() -> str:
    sha = hashlib.sha256()
    files = [ROOT / "requirements.txt", ROOT / "scripts/contract_lots.py"]
    for directory in ("core", "data", "factors", "optimization", "trading", "native/mf_factor_kernels/src"):
        files.extend(sorted((ROOT / directory).rglob("*.py")))
        files.extend(sorted((ROOT / directory).rglob("*.rs")))
    files += [ROOT / "research" / name for name in (
        "portfolio_experiment_support.py", "historical_portfolio_search.py", "return_ledger.py")
        if (ROOT / "research" / name).exists()]
    for path in files:
        sha.update(path.relative_to(ROOT).as_posix().encode())
        sha.update(path.read_bytes())
    return sha.hexdigest()


def runtime_contract() -> dict:
    contract = {n: version(n) for n in ("numpy", "pandas", "polars", "scipy")}
    contract.update(python=platform.python_version(), **factor_kernel_contract())
    if contract["factor_kernel_mode"] != "reference":
        module = find_spec("_mf_factor_kernels")
        contract["native_binary_sha256"] = file_hash(module.origin) if module and module.origin else None
    return contract


def panel_cache_key(identity, factors, compute_start):
    """Raw factors do not depend on broker holdings or the portfolio route."""
    return digest({**{key: identity[key] for key in (
        "data_date", "code_sha256", "data_fingerprint", "runtime", "config")},
        "factors": factors, "compute_start": str(compute_start)})


def check_publication_gate(gate: dict, contracts: list[dict]) -> None:
    if gate.get("enabled") is not True or gate.get("approval_status") != "approved_for_target_publication":
        raise ValueError("production target publication is disabled or not approved")
    path = resolve(gate.get("deployment_package") or "__missing_deployment__")
    package = json.loads(path.read_text(encoding="utf-8"))
    if (package.get("strategies") != contracts or package.get("code_sha256") != code_fingerprint()
            or package.get("runtime") != runtime_contract()
            or gate.get("protocol_sha256") != digest(package)):
        raise ValueError("approved deployment package hash or strategy contract changed")


def generate_weights(strategy_ids, *, as_of=None, output_root="runs/trading/weights",
                     catalog_path="config/strategy_library.yaml", mode="simulation", gate=None,
                     full_panel=False, holdings=None) -> Path:
    if mode not in {"simulation", "production"}:
        raise ValueError("weight mode must be simulation or production")
    contracts = strategy_contracts(strategy_ids, catalog_path)
    if mode == "production":
        if not all(row["formal"] for row in contracts):
            raise ValueError("production target publication requires formal strategies")
        check_publication_gate(gate or {}, contracts)
    cfg = load_config(resolve("config/default.yaml"))
    manager = DataManager.from_config(cfg)
    source = manager.source
    try:
        latest = source.fetch_latest_trade_date().date().isoformat()
        decision = str(as_of or latest)
        require_closed_date(decision, latest)
        states = holdings or {}
        for contract in contracts:
            state = states.get(contract["strategy_id"])
            if contract["recipe"]["rank_exit_buffer"] and (
                    not isinstance(state, dict) or state.get("data_date") != decision
                    or not state.get("source") or not isinstance(state.get("holdings"), dict)):
                raise ValueError("rank_exit_buffer requires dated, reconciled holdings for each strategy instance")
            if state and (set(state["holdings"]) - set(cfg.universe)
                          or not all(np.isfinite(float(v)) for v in state["holdings"].values())):
                raise ValueError("holding state has unknown roots or nonfinite values")
        # A daily date alone does not establish a complete multi-frequency release.
        for table in ("bars_1m", "bars_5m", "bars_15m"):
            end = source._execute_polars(f"SELECT MAX(trade_date) FROM market.{table}").item(0, 0)
            if end is None or str(end)[:10] < decision:
                raise ValueError(f"published {table} does not cover {decision}")
        _, quotes = _contract_snapshot(source, cfg.universe, as_of=decision)
        exchanges = source._execute_polars(
            "SELECT DISTINCT symbol, exchange FROM market.bars_1d WHERE trade_date = ?", [decision]
        ).to_dicts()
        for quote in quotes.values():
            matches = {str(r["exchange"]).upper() for r in exchanges
                       if str(r["symbol"]).upper() == quote["contract"].upper()}
            if len(matches) != 1:
                raise ValueError(f"missing or ambiguous exchange for {quote['contract']}")
            exchange = matches.pop()
            quote["exchange"] = {"SHF": "SHFE", "CZC": "CZCE", "CFE": "CFFEX"}.get(exchange, exchange)
        calendar = pd.DatetimeIndex(manager.get_calendar(contracts[0]["panel_start"], decision))
        required = max(c["recipe"]["ic_window"] for c in contracts) + 2
        if len(calendar) < required:
            raise ValueError("insufficient mature IC history")
        compute_start = calendar[-required]
        identity = {"stage": "weights", "mode": mode, "data_date": decision,
                    "strategies": contracts, "code_sha256": code_fingerprint(),
                    "config": cfg.model_dump(mode="json"),
                    "data_fingerprint": source.checkpoint_source_fingerprint(calendar[0], decision),
                    "runtime": runtime_contract(),
                    "quotes": quotes, "holding_states": states}
        destination = resolve(output_root) / digest(identity)
        if destination.exists() and not full_panel:
            read_artifact(destination)
            return destination
        factors = list(dict.fromkeys(f for c in contracts for f in c["factors"]))
        tail_start = None if full_panel else compute_start
        checkpoint = resolve(output_root).parent / "factor_checkpoints" / panel_cache_key(identity, factors, tail_start)
        runner = FactorPanelRunner(factors, start=calendar[0], end=decision,
                                   compute_start=tail_start, checkpoint_dir=checkpoint)
        try:
            rows = []
            for contract in contracts:
                view = runner.for_factors(contract["factors"], factor_directions=contract["directions"])
                recipe = PortfolioRecipe.from_config(type(cfg.production_portfolio)(**contract["recipe"]))
                evaluator = PortfolioEvaluator(view, start=decision, end=decision,
                    ic_window=contract["recipe"]["ic_window"],
                    risk_lookback_calendar_days=contract["recipe"]["risk_lookback_calendar_days"])
                weights = (evaluator.target_for_holdings(contract["factors"], recipe, decision,
                           states[contract["strategy_id"]]["holdings"]) if recipe.rank_exit_buffer
                           else evaluator.weights(contract["factors"], recipe).loc[pd.Timestamp(decision)])
                weights = weights.reindex(view.u, fill_value=0.0)
                if not np.isfinite(weights.to_numpy()).all() or weights.abs().sum() <= 0:
                    raise ValueError("invalid or empty target weights")
                for root, weight in weights.items():
                    rows.append({"strategy_id": contract["strategy_id"], "data_date": decision,
                                 "root": root, "contract": quotes[root]["contract"],
                                 "exchange": quotes[root]["exchange"],
                                 "close": quotes[root]["price"], "weight": float(weight)})
            if (code_fingerprint() != identity["code_sha256"]
                    or runtime_contract() != identity["runtime"]
                    or strategy_contracts(strategy_ids, catalog_path) != contracts):
                raise ValueError("strategy or computation code changed during generation; run again")
            return write_artifact(resolve(output_root), identity,
                {"weights.csv": rows, "deployment.json": {"strategies": contracts,
                  "code_sha256": identity["code_sha256"], "runtime": identity["runtime"]}})
        finally:
            runner.env.data_manager.source.close()
    finally:
        source.close()
