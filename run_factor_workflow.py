"""IDE entrypoint for factor validation and explicit library admission.

Edit only the constants in ``IDE SETTINGS`` and press Run.  Full-history
research is intentionally not a branch in this file.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path

from core.config import load_config
from research.effective_factor_library import admit_validation_run
from workflows.factor_validation import run_default_factor_validation


class FactorWorkflow(Enum):
    VALIDATE_ALL_INTRADAY = "validate_all_intraday"
    ADMIT_COMPLETED_RUN = "admit_completed_run"


# ============================== IDE SETTINGS ==============================
WORKFLOW = FactorWorkflow.VALIDATE_ALL_INTRADAY
CONFIG_PATH = "config/default.yaml"

# VALIDATE_ALL_INTRADAY: None creates a unique timestamped standard run.
VALIDATION_RUN_ID: str | None = None

# ADMIT_COMPLETED_RUN: set both values after reviewing a completed run.
ADMISSION_RUN_DIR: str | None = None
ADMITTED_AT: str | None = None
# ========================================================================


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def _library_path(config_path: str) -> Path:
    configured = Path(load_config(config_path).factor_library.path)
    if not configured.is_absolute():
        configured = _project_root() / configured
    return configured.resolve()


def main() -> None:
    if WORKFLOW is FactorWorkflow.VALIDATE_ALL_INTRADAY:
        run_id = VALIDATION_RUN_ID or datetime.now().strftime(
            "%Y%m%d_%H%M%S_intraday_default_window"
        )
        run_default_factor_validation(run_id=run_id, config_path=CONFIG_PATH)
        return

    if WORKFLOW is FactorWorkflow.ADMIT_COMPLETED_RUN:
        if not ADMISSION_RUN_DIR or not ADMITTED_AT:
            raise ValueError(
                "ADMIT_COMPLETED_RUN requires ADMISSION_RUN_DIR and ADMITTED_AT"
            )
        payload = admit_validation_run(
            ADMISSION_RUN_DIR,
            _library_path(CONFIG_PATH),
            admitted_at=ADMITTED_AT,
        )
        print(f"有效因子库已更新，当前记录数: {len(payload['factors'])}")
        return

    raise ValueError(f"unsupported factor workflow: {WORKFLOW!r}")


if __name__ == "__main__":
    main()
