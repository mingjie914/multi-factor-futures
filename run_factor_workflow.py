"""Single IDE entrypoint for factor admission, observation, and subset selection.

Edit only ``IDE SETTINGS`` and press Run.  The default validates one explicit
development batch.  Full-pool rebuilding is a separate, explicit choice.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path

from core.config import load_config
from research.effective_factor_library import admit_validation_run
from workflows.factor_selection import run_effective_factor_selection
from workflows.factor_validation import (
    run_common_horizon_factor_validation,
    run_default_factor_validation,
)


class FactorWorkflow(Enum):
    VALIDATE_FACTOR_BATCH = "validate_factor_batch"
    VALIDATE_ALL_INTRADAY = "validate_all_intraday"
    OBSERVE_COMMON_HORIZON = "observe_common_horizon"
    ADMIT_COMPLETED_RUN = "admit_completed_run"
    SELECT_EFFECTIVE_SUBSETS = "select_effective_subsets"


# ============================== IDE SETTINGS ==============================
WORKFLOW = FactorWorkflow.VALIDATE_FACTOR_BATCH
CONFIG_PATH = "config/default.yaml"

# Daily default: list every factor created in one predeclared development batch.
# Leave empty only when another WORKFLOW is selected.  Duplicates are removed
# without changing first-seen order; unknown/non-intraday names fail closed.
FACTOR_NAMES: tuple[str, ...] = ()
VALIDATION_RUN_ID: str | None = None

# OBSERVE_COMMON_HORIZON is non-admissible sensitivity evidence.
COMMON_HORIZON = 5

# ADMIT_COMPLETED_RUN: set both after reviewing a formal admission run.
ADMISSION_RUN_DIR: str | None = None
ADMITTED_AT: str | None = None

# SELECT_EFFECTIVE_SUBSETS consumes only the current admitted library.
SELECTION_RUN_ID: str | None = None
# ========================================================================


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def _library_path(config_path: str) -> Path:
    configured = Path(load_config(config_path).factor_library.path)
    if not configured.is_absolute():
        configured = _project_root() / configured
    return configured.resolve()


def main() -> None:
    if WORKFLOW is FactorWorkflow.VALIDATE_FACTOR_BATCH:
        if not FACTOR_NAMES:
            raise ValueError(
                "VALIDATE_FACTOR_BATCH requires an explicit non-empty FACTOR_NAMES"
            )
        run_id = VALIDATION_RUN_ID or datetime.now().strftime(
            "%Y%m%d_%H%M%S_factor_batch_admission"
        )
        run_default_factor_validation(
            run_id=run_id,
            config_path=CONFIG_PATH,
            factor_names=FACTOR_NAMES,
        )
        return

    if WORKFLOW is FactorWorkflow.VALIDATE_ALL_INTRADAY:
        run_id = VALIDATION_RUN_ID or datetime.now().strftime(
            "%Y%m%d_%H%M%S_full_intraday_admission"
        )
        run_default_factor_validation(
            run_id=run_id,
            config_path=CONFIG_PATH,
            all_registered=True,
        )
        return

    if WORKFLOW is FactorWorkflow.OBSERVE_COMMON_HORIZON:
        run_id = VALIDATION_RUN_ID or datetime.now().strftime(
            f"%Y%m%d_%H%M%S_common_h{int(COMMON_HORIZON)}_observation"
        )
        run_common_horizon_factor_validation(
            run_id=run_id,
            common_horizon=COMMON_HORIZON,
            config_path=CONFIG_PATH,
        )
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

    if WORKFLOW is FactorWorkflow.SELECT_EFFECTIVE_SUBSETS:
        run_id = SELECTION_RUN_ID or datetime.now().strftime(
            "%Y%m%d_%H%M%S_effective_factor_selection"
        )
        output = run_effective_factor_selection(
            run_id=run_id,
            config_path=CONFIG_PATH,
        )
        print(f"因子子集筛选产物: {output}")
        return

    raise ValueError(f"unsupported factor workflow: {WORKFLOW!r}")


if __name__ == "__main__":
    main()
