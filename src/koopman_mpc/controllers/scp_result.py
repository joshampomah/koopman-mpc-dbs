"""SCPResult dataclass — local stub to avoid dependency on dcnn-tube-mpc-dbs.

The KoopmanController returns an SCPResult with n_iterations=1 and
converged=True to match the interface expected by the simulation harness,
even though Koopman MPC solves a single QP (no SCP iterations).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class SCPResult:
    """Result container shared by SCP-based and single-QP controllers.

    Attributes:
        u_optimal: Optimal control sequence, shape (N,).
        y_nominal: Predicted outputs at optimal u, shape (N,).
        u_nominal: Nominal (warm-start) control sequence, shape (N,).
        J_optimal: Final optimal cost value.
        n_iterations: Number of SCP iterations (1 for Koopman QP).
        converged: True if solution is valid.
        iteration_costs: Cost per iteration (single entry for Koopman).
        iteration_times: Solve time per iteration.
        status: Solver status string.
        s_max_final: Upper tube bounds at final solution, shape (N,).
        s_min_final: Lower tube bounds at final solution, shape (N,).
    """

    u_optimal: np.ndarray
    y_nominal: np.ndarray
    u_nominal: np.ndarray
    J_optimal: float
    n_iterations: int
    converged: bool
    iteration_costs: List[float] = field(default_factory=list)
    iteration_times: List[float] = field(default_factory=list)
    status: str = "SUCCESS"
    s_max_final: Optional[np.ndarray] = None
    s_min_final: Optional[np.ndarray] = None
