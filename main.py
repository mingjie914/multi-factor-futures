"""Small, explicit command gateway for the multi-factor framework."""
from __future__ import annotations

import importlib
import os
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
    "data-health": ("workflows.diagnostics.data_health", "read-only data checks"),
    "close": ("workflows.close_decision", "fail-closed end-of-day decision"),
}


def _check_dependencies() -> list[str]:
    missing = []
    for module_name in ("numpy", "pandas", "yaml", "pydantic"):
        try:
            importlib.import_module(module_name)
        except ImportError:
            missing.append(module_name)
    return missing


def _print_help() -> None:
    print("Multi-factor futures research and trading framework")
    print("\nUsage: python main.py <command> [options]\n")
    print("Commands:")
    for command, (_, description) in WORKFLOW_COMMANDS.items():
        print(f"  {command:<12} {description}")
    print("\nRun 'python main.py <command> --help' for command options.")
    print("Experimental workflows remain under workflows/experiments and are not production entry points.")


def _dispatch(command: str) -> None:
    module_name = WORKFLOW_COMMANDS[command][0]
    module = importlib.import_module(module_name)
    original_argv = sys.argv
    sys.argv = [f"{original_argv[0]} {command}", *original_argv[2:]]
    try:
        module.main()
    finally:
        sys.argv = original_argv


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
        print("Install with: python -m pip install -r requirements-minimal.txt")
        raise SystemExit(1)
    _dispatch(command)


if __name__ == "__main__":
    main()
