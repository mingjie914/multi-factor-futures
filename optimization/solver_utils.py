from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


class SolverValidationError(RuntimeError):
    """Raised when no solver produces a finite, constraint-valid optimum."""


def validated_psd_covariance(
    matrix: np.ndarray,
    *,
    relative_tolerance: float = 1e-8,
    eigenvalue_floor: float = 0.0,
) -> np.ndarray:
    """Validate a covariance matrix and clip numerical negative eigenvalues."""
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1] or values.size == 0:
        raise ValueError("covariance must be a non-empty square matrix")
    if not np.isfinite(eigenvalue_floor) or eigenvalue_floor < 0.0:
        raise ValueError("eigenvalue_floor must be finite and non-negative")
    if not np.isfinite(values).all():
        raise ValueError("covariance contains NaN/Inf")
    scale = max(float(np.max(np.abs(values))), 1e-12)
    if float(np.max(np.abs(values - values.T))) > relative_tolerance * scale:
        raise ValueError("covariance is materially asymmetric")
    symmetric = (values + values.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    if float(eigenvalues.min()) < -relative_tolerance * scale:
        raise ValueError("covariance is not positive semidefinite")
    if float(eigenvalues.min()) < eigenvalue_floor:
        eigenvalues = np.maximum(eigenvalues, eigenvalue_floor)
        symmetric = (eigenvectors * eigenvalues) @ eigenvectors.T
        symmetric = (symmetric + symmetric.T) / 2.0
    return symmetric


@dataclass(frozen=True)
class SolverOutcome:
    solver: str
    status: str
    objective_value: float
    max_constraint_violation: float


def _max_constraint_violation(constraints: Iterable[object]) -> float:
    maximum = 0.0
    for constraint in constraints:
        violation = constraint.violation()
        if violation is None:
            raise SolverValidationError("constraint violation is unavailable")
        values = np.asarray(violation, dtype=float)
        if values.size == 0:
            continue
        if not np.isfinite(values).all():
            raise SolverValidationError("constraint violation contains NaN/Inf")
        maximum = max(maximum, float(np.max(np.abs(values))))
    return maximum


def solve_validated(
    problem,
    variable,
    solver_chain: Sequence[str],
    *,
    constraint_tolerance: float = 1e-5,
    accept_inaccurate: bool = False,
) -> SolverOutcome:
    """Solve a CVXPY problem and accept only a finite, constraint-valid result.

    A non-None ``variable.value`` is not sufficient evidence that an optimizer
    succeeded. This helper validates the CVXPY status, decision vector,
    objective value, and every applied constraint before returning.
    """
    import cvxpy as cp

    allowed_statuses = {cp.OPTIMAL}
    if accept_inaccurate:
        allowed_statuses.add(cp.OPTIMAL_INACCURATE)

    available = set(cp.installed_solvers())
    attempts: list[str] = []
    for solver_name in dict.fromkeys(solver_chain):
        if solver_name not in available:
            attempts.append(f"{solver_name}:not-installed")
            continue
        try:
            problem.solve(solver=solver_name, verbose=False, warm_start=False)
            status = str(problem.status)
            if problem.status not in allowed_statuses:
                attempts.append(f"{solver_name}:{status}")
                continue

            values = np.asarray(variable.value, dtype=float).reshape(-1)
            if values.size == 0 or not np.isfinite(values).all():
                attempts.append(f"{solver_name}:{status}:non-finite-weights")
                continue

            objective = float(problem.value)
            if not np.isfinite(objective):
                attempts.append(f"{solver_name}:{status}:non-finite-objective")
                continue

            violation = _max_constraint_violation(problem.constraints)
            if violation > constraint_tolerance:
                attempts.append(
                    f"{solver_name}:{status}:constraint-violation={violation:.3e}"
                )
                continue

            return SolverOutcome(
                solver=solver_name,
                status=status,
                objective_value=objective,
                max_constraint_violation=violation,
            )
        except Exception as exc:
            attempts.append(f"{solver_name}:{type(exc).__name__}:{exc}")

    detail = "; ".join(attempts) if attempts else "no solver candidates"
    raise SolverValidationError(f"no validated optimal solution ({detail})")
