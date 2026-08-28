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
from workflows.factor_validation import (
    run_common_horizon_factor_validation,
    run_default_factor_validation,
)
from workflows.factor_selection import run_effective_factor_selection


class FactorWorkflow(Enum):
    VALIDATE_ALL_INTRADAY = "validate_all_intraday"
    # Explicit legacy-parity comparison: all registered factors use one
    # next-bar daily return label; it never replaces the standard library.
    VALIDATE_ALL_INTRADAY_COMMON_H1 = "validate_all_intraday_common_h1"
    # Explicit common five-bar sensitivity comparison; independent evidence.
    VALIDATE_ALL_INTRADAY_COMMON_H5 = "validate_all_intraday_common_h5"
    ADMIT_COMPLETED_RUN = "admit_completed_run"
    SELECT_EFFECTIVE_SUBSETS = "select_effective_subsets"
    # Explicit common-H5 subset comparison source.  This consumes a finalized
    # common-horizon validation run and never mutates the effective library.
    SELECT_COMMON_H5_SUBSETS = "select_common_h5_subsets"


# ============================== IDE SETTINGS ==============================
WORKFLOW = FactorWorkflow.VALIDATE_ALL_INTRADAY
CONFIG_PATH = "config/default.yaml"

# VALIDATE_ALL_INTRADAY: validates the complete registered intraday discovery
# set (not the effective library). The count is discovered from the registry;
# e.g. a future 688-factor discovery set produces a 688-row full-detail run.
VALIDATION_RUN_ID: str | None = None

# ADMIT_COMPLETED_RUN: set both values after reviewing a completed run.
ADMISSION_RUN_DIR: str | None = None
ADMITTED_AT: str | None = None

# SELECT_EFFECTIVE_SUBSETS: derive parallel subsets from the current effective
# library (not the complete discovery set) using the locked warmup + 126 IS +
# 42 OOS contract. The library size is discovered from factor_library.path;
# e.g. after 25 new admissions it consumes 100 effective factors, not 688
# unadmitted candidates.
SELECTION_RUN_ID: str | None = None
# SELECT_COMMON_H5_SUBSETS: derive parallel candidates from the finalized
# common-H5 validation evidence.  Keep this path explicit so an IDE Run never
# silently switches the ordinary effective-library selection input.
COMMON_H5_VALIDATION_RUN_DIR = (
    "runs/factor_validation/20260826_intraday588_common_h5_is126_oos42_cutoff_20260515"
)
COMMON_H5_SELECTION_RUN_ID: str | None = None
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

    if WORKFLOW in {
        FactorWorkflow.VALIDATE_ALL_INTRADAY_COMMON_H1,
        FactorWorkflow.VALIDATE_ALL_INTRADAY_COMMON_H5,
    }:
        horizon = (
            1
            if WORKFLOW is FactorWorkflow.VALIDATE_ALL_INTRADAY_COMMON_H1
            else 5
        )
        suffix = f"common_h{horizon}"
        run_id = VALIDATION_RUN_ID or datetime.now().strftime(
            f"%Y%m%d_%H%M%S_intraday_{suffix}"
        )
        run_common_horizon_factor_validation(
            run_id=run_id,
            common_horizon=horizon,
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

    if WORKFLOW is FactorWorkflow.SELECT_COMMON_H5_SUBSETS:
        run_id = COMMON_H5_SELECTION_RUN_ID or datetime.now().strftime(
            "%Y%m%d_%H%M%S_common_h5_factor_selection"
        )
        output = run_effective_factor_selection(
            run_id=run_id,
            config_path=CONFIG_PATH,
            source_run_dir=COMMON_H5_VALIDATION_RUN_DIR,
            common_horizon=5,
        )
        print(f"共同H5因子子集筛选产物: {output}")
        return

    raise ValueError(f"unsupported factor workflow: {WORKFLOW!r}")


if __name__ == "__main__":
    main()
