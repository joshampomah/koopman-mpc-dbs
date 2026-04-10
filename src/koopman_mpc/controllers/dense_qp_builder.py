"""Dense QP builder for Koopman MPC.

Uses PIQP DenseSolver with pre-allocated matrices for maximum speed.
For N=7, the QP has 14 variables — sparse overhead is wasteful.

Pre-allocates P, G, h arrays once, then updates only the entries that
change per timestep (Jacobian rows and RHS values). This eliminates
scipy sparse matrix construction, which was 84% of the QP time.

Typical timing (N=7): 0.074ms total (0.014ms update + 0.060ms solve).
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np


@dataclass
class DenseQPResult:
    """Result of the dense QP solve."""
    u_optimal: np.ndarray
    t_optimal: np.ndarray
    cost: float
    status: str
    solve_time: float
    y_linearised: np.ndarray


class DenseQPBuilder:
    """Pre-allocated dense QP builder for Koopman Tube MPC.

    QP structure (2N variables: [u(N), t(N)]):
        min  R·||u||² + Q·||t||²
        s.t. t_i >= y_nom_i + J_i·(u-u_nom) + s_max_i - beta_0
             t_i >= 0
             u_min <= u_i <= u_max
             |u_i - u_{i-1}| <= delta_u_max
    """

    def __init__(
        self,
        N: int,
        Q: float,
        R: float,
        beta_0: float,
        u_min: float,
        u_max: float,
        delta_u_max: float,
    ):
        self.N = N
        self.Q = Q
        self.R = R
        self.beta_0 = beta_0
        self.u_min = u_min
        self.u_max = u_max
        self.delta_u_max = delta_u_max

        n_vars = 2 * N
        n_ineq = 6 * N  # tracking + nonneg + box_lo + box_hi + rate_up + rate_dn

        # Pre-allocate cost matrix (diagonal, never changes)
        self._P = np.zeros((n_vars, n_vars))
        np.fill_diagonal(self._P[:N, :N], 2.0 * R)
        np.fill_diagonal(self._P[N:, N:], 2.0 * Q)
        self._c = np.zeros(n_vars)

        # No equality constraints
        self._A_eq = np.zeros((0, n_vars))
        self._b_eq = np.array([])

        # Pre-allocate inequality constraint matrix
        self._G = np.zeros((n_ineq, n_vars))
        self._h = np.zeros(n_ineq)
        self._h_l = np.full(n_ineq, -1e30)

        # Fixed entries in G (only tracking rows change per step)
        # Nonneg: -t_i <= 0
        for i in range(N):
            self._G[N + i, N + i] = -1.0

        # Box lower: -u_i <= -u_min
        for i in range(N):
            self._G[2 * N + i, i] = -1.0
            self._h[2 * N + i] = -u_min

        # Box upper: u_i <= u_max
        for i in range(N):
            self._G[3 * N + i, i] = 1.0
            self._h[3 * N + i] = u_max

        # Rate up: u_i - u_{i-1} <= delta_u_max
        for i in range(N):
            self._G[4 * N + i, i] = 1.0
            if i > 0:
                self._G[4 * N + i, i - 1] = -1.0
                self._h[4 * N + i] = delta_u_max

        # Rate down: -(u_i - u_{i-1}) <= delta_u_max
        for i in range(N):
            self._G[5 * N + i, i] = -1.0
            if i > 0:
                self._G[5 * N + i, i - 1] = 1.0
                self._h[5 * N + i] = delta_u_max

        # Tracking rows: G[0:N] updated per step (J and -t columns)
        for i in range(N):
            self._G[i, N + i] = -1.0  # -t_i coefficient (fixed)

        self._solver = None
        self._first_solve = True

    def solve(
        self,
        y_nominal: np.ndarray,
        J: np.ndarray,
        u_nominal: np.ndarray,
        u_prev: float,
        s_max: np.ndarray,
    ) -> DenseQPResult:
        """Solve the QP with updated predictions and Jacobian.

        Args:
            y_nominal: Nominal predictions, shape (N,).
            J: Jacobian dy/du, shape (N, N), lower-triangular.
            u_nominal: Nominal control sequence, shape (N,).
            u_prev: Previous applied control.
            s_max: Tube upper bounds, shape (N,).

        Returns:
            DenseQPResult with optimal u and diagnostics.
        """
        import piqp

        t_start = time.time()
        N = self.N

        # Update tracking rows (J changes per step)
        self._G[:N, :N] = J

        # Update tracking RHS
        for i in range(N):
            self._h[i] = J[i, :] @ u_nominal - y_nominal[i] - s_max[i] + self.beta_0

        # Update rate constraints for u_prev (step 0 only)
        self._h[4 * N] = self.delta_u_max + u_prev
        self._h[5 * N] = self.delta_u_max - u_prev

        # Solve
        if self._solver is None:
            self._solver = piqp.DenseSolver()
            self._solver.settings.verbose = False
            self._solver.settings.eps_abs = 1e-6
            self._solver.settings.eps_rel = 1e-6
            self._solver.settings.max_iter = 500

        if self._first_solve:
            self._solver.setup(
                self._P, self._c, self._A_eq, self._b_eq,
                self._G, self._h_l, self._h,
            )
            self._first_solve = False
        else:
            self._solver.update(
                P=self._P, c=self._c,
                A=self._A_eq, b=self._b_eq,
                G=self._G, h_l=self._h_l, h_u=self._h,
            )

        status = self._solver.solve()
        solve_time = time.time() - t_start

        is_solved = (status == piqp.PIQP_SOLVED)

        if is_solved:
            x_sol = np.array(self._solver.result.x)
            u_opt = x_sol[:N]
            t_opt = x_sol[N:]
            cost = float(self._solver.result.info.primal_obj)
        else:
            u_opt = u_nominal.copy()
            t_opt = np.zeros(N)
            cost = float("inf")

        # Linearised predictions at optimal u
        delta_u = u_opt - u_nominal
        y_lin = y_nominal + J @ delta_u

        return DenseQPResult(
            u_optimal=u_opt,
            t_optimal=t_opt,
            cost=cost,
            status="Solved" if is_solved else str(status),
            solve_time=solve_time,
            y_linearised=y_lin,
        )
