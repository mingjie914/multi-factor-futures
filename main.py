"""Small, explicit command gateway for the multi-factor framework."""
from __future__ import annotations

import importlib
import os
from pathlib import Path
import sys


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


WORKFLOW_COMMANDS = {
    "research": ("workflows.research", "factor tests and census studies"),
    "adaptivity": ("workflows.factor_adaptivity", "sector/horizon diagnostics"),
    "backtest": ("workflows.backtest", "single-portfolio backtest"),
    "multi": ("workflows.multi_backtest", "multi-sleeve portfolio backtest"),
    "walkforward": ("workflows.walkforward", "frozen walk-forward validation"),
    "summarize": (
        "workflows.summarize_results", "summarize frozen walk-forward outputs"
    ),
    "data-health": ("workflows.diagnostics.data_health", "read-only data checks"),
    "close": ("workflows.close_decision", "fail-closed end-of-day target publication"),
    "mining": ("factor_mining.cli", "local factor mining and candidate pool"),
}

MINED_SNAPSHOT_OPTION = "--mined-snapshot"


def _check_dependencies() -> list[str]:
    missing = []
    for module_name in ("numpy", "pandas", "yaml", "pydantic"):
        try:
            importlib.import_module(module_name)
        except ImportError:
            missing.append(module_name)
    return missing


def _print_help() -> None:
    print("Multi-factor futures research and target-weight framework")
    print(
        "\nUsage: python main.py <command> "
        "[--mined-snapshot PATH] [options]\n"
    )
    print("Commands:")
    for command, (_, description) in WORKFLOW_COMMANDS.items():
        print(f"  {command:<12} {description}")
    print("\nRun 'python main.py <command> --help' for command options.")
    print(
        "Use '--mined-snapshot PATH' to load an immutable mined-factor "
        "snapshot before a workflow starts."
    )
    print(
        "Experimental workflows remain under workflows/experiments and are "
        "not production entry points."
    )


def _dispatch(command: str) -> None:
    module_name = WORKFLOW_COMMANDS[command][0]
    module = importlib.import_module(module_name)
    original_argv = sys.argv
    sys.argv = [f"{original_argv[0]} {command}", *original_argv[2:]]
    try:
        result = module.main()
        if isinstance(result, int) and result:
            raise SystemExit(result)
    finally:
        sys.argv = original_argv


def _configure_mined_snapshot(argv: list[str]) -> tuple[list[str], int]:
    """Strip the gateway option and validate its snapshot before imports."""
    cleaned = list(argv[:2])
    snapshot_values: list[str] = []
    index = 2
    while index < len(argv):
        argument = argv[index]
        if argument == MINED_SNAPSHOT_OPTION:
            if index + 1 >= len(argv):
                raise ValueError(f"{MINED_SNAPSHOT_OPTION} requires a path")
            snapshot_values.append(argv[index + 1])
            index += 2
            continue
        prefix = MINED_SNAPSHOT_OPTION + "="
        if argument.startswith(prefix):
            snapshot_values.append(argument[len(prefix):])
            index += 1
            continue
        cleaned.append(argument)
        index += 1

    if not snapshot_values:
        return cleaned, 0
    if len(snapshot_values) != 1 or not snapshot_values[0].strip():
        raise ValueError(f"{MINED_SNAPSHOT_OPTION} must be provided exactly once")
    if argv[1] == "mining":
        raise ValueError(
            f"{MINED_SNAPSHOT_OPTION} loads factors into framework workflows; "
            "it is not a mining-command option"
        )

    path = Path(snapshot_values[0]).expanduser()
    if not path.is_absolute():
        path = Path(PROJECT_ROOT) / path
    path = path.resolve()
    from factor_mining.bridge import SNAPSHOT_ENV
    from factor_mining.repository import load_snapshot

    candidates = load_snapshot(path, require_framework=True)
    os.environ[SNAPSHOT_ENV] = str(path)
    return cleaned, len(candidates)


def main() -> None:
    if len(sys.argv) == 1 or sys.argv[1] in {"-h", "--help"}:
        _print_help()
        return

    command = sys.argv[1]
    if command not in WORKFLOW_COMMANDS:
        print(f"Unknown command: {command}", file=sys.stderr)
        _print_help()
        raise SystemExit(2)

    missing = _check_dependencies()
    if missing:
        print(f"Missing dependencies: {', '.join(missing)}", file=sys.stderr)
        print("Install with: python -m pip install -r requirements.txt")
        raise SystemExit(1)
    try:
        configured_argv, candidate_count = _configure_mined_snapshot(sys.argv)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Mined-factor snapshot error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    sys.argv = configured_argv
    if candidate_count:
        print(f"Validated mined-factor snapshot: {candidate_count} candidates")
    _dispatch(command)


if __name__ == "__main__":
    main()
