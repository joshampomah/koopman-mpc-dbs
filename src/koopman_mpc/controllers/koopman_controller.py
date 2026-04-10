"""Koopman MPC controller — single QP per timestep, no SCP.

The Koopman model makes predictions that are affine in future control:
    y_k = e_k + F_k @ u[0:k]
where e_k = C @ A_k @ psi(z_t) is a constant given the current state.

This means the MPC problem is a standard QP (no nonlinear terms in u),
eliminating the need for SCP iterations. One QP solve per timestep.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from koopman_mpc.config.device_config import get_device_config
from koopman_mpc.controllers.scp_result import SCPResult

_DEVICE_CONFIG = get_device_config()


@dataclass
class KoopmanMPCConfig:
    """Configuration for Koopman MPC controller."""

    prediction_horizon: int = 5
    Q: float = 50000.0
    R: float = 1.0
    beta_0: float = field(
        default_factory=lambda: _DEVICE_CONFIG.beta.default_threshold
    )
    u_min: float = field(default_factory=lambda: _DEVICE_CONFIG.constraints.u_min)
    u_max: float = field(default_factory=lambda: _DEVICE_CONFIG.constraints.u_max)
    delta_u_max: float = field(
        default_factory=lambda: _DEVICE_CONFIG.constraints.delta_u_max
    )
    use_warm_start: bool = True
    n_state_y: int = 15
    n_state_u: int = 15

    def __post_init__(self):
        MIN_R_COST = 0.01
        r_cost = self.R * self.u_max ** 2
        if r_cost > 0 and r_cost < MIN_R_COST:
            scale = MIN_R_COST / r_cost
            self.Q *= scale
            self.R *= scale


class KoopmanController:
    """Koopman MPC controller — single QP, no SCP iterations.

    Computes QP matrices (e, F) from the Koopman model at each step,
    then solves one QP. Supports ACI online tube adaptation.
    """

    def __init__(
        self,
        predictor,
        config: KoopmanMPCConfig,
        W_bounds: Optional[np.ndarray] = None,
    ):
        self.predictor = predictor
        self.config = config

        N = config.prediction_horizon
        if W_bounds is not None and W_bounds.shape[0] >= N:
            self.s_max = W_bounds[:N, 1]
            self._offline_bounds = W_bounds[:N]
        else:
            self.s_max = np.full(N, 0.1)
            self._offline_bounds = np.column_stack([-self.s_max, self.s_max])

        self._last_result: Optional[SCPResult] = None
        self._last_u_applied: float = 0.0
        self._step_count: int = 0
        self._total_solve_time: float = 0.0

        self._qp_builder = None

        # Precompute constant matrices:
        # F[k,j] = C @ B_k[:, j] is state-independent (only e depends on state)
        import torch
        with torch.no_grad():
            C_np = predictor.C.squeeze(0).detach().numpy()
            self._F = np.zeros((N, N))
            self._CA = []
            for k in range(N):
                A_k = predictor.A[k].detach().numpy()
                B_k = predictor.B[k].detach().numpy()
                self._CA.append(C_np @ A_k)  # (d_lift,)
                for j in range(k + 1):
                    self._F[k, j] = C_np @ B_k[:, j]

    def _get_qp_builder(self):
        """Lazy-init dense QP builder (PIQP DenseSolver, no sparse overhead)."""
        if self._qp_builder is None:
            from koopman_mpc.controllers.dense_qp_builder import DenseQPBuilder
            self._qp_builder = DenseQPBuilder(
                N=self.config.prediction_horizon,
                Q=self.config.Q,
                R=self.config.R,
                beta_0=self.config.beta_0,
                u_min=self.config.u_min,
                u_max=self.config.u_max,
                delta_u_max=self.config.delta_u_max,
            )
        return self._qp_builder

    def compute_control(
        self,
        y_history: np.ndarray,
        u_history: np.ndarray,
        u_prev: float,
    ) -> Tuple[float, SCPResult]:
        """Compute optimal control via single QP.

        Args:
            y_history: Past beta values, shape (n_state_y,). Newest first.
            u_history: Past control inputs, shape (n_state_u,). Newest first.
            u_prev: Previous applied control.

        Returns:
            Tuple (u_k, SCPResult).
        """
        import time

        N = self.config.prediction_horizon

        # Build state z = [y_past, u_past] (oldest to newest)
        y_past = y_history[::-1]
        u_past = u_history[::-1]
        z = np.concatenate([y_past, u_past]).astype(np.float32)

        # Lift state and compute e (F is precomputed, constant)
        psi = self.predictor.lift(z)  # encoder forward pass only
        e = np.array([ca @ psi for ca in self._CA])  # e_k = CA_k @ psi
        F = self._F  # precomputed, state-independent

        # Warm-start
        if self._last_result is not None and self.config.use_warm_start:
            u_nom = np.append(
                self._last_result.u_optimal[1:],
                self._last_result.u_optimal[-1],
            )
            u_nom = np.clip(u_nom, self.config.u_min, self.config.u_max)
        else:
            u_nom = np.full(N, u_prev)

        # Compute nominal predictions at u_nom
        y_nom = e + F @ u_nom

        # Solve single QP
        t_start = time.time()
        qp = self._get_qp_builder()
        qp_result = qp.solve(y_nom, F, u_nom, u_prev, self.s_max)
        solve_time = time.time() - t_start

        u_k = float(qp_result.u_optimal[0])

        # Predictions at optimal u
        y_opt = e + F @ qp_result.u_optimal

        result = SCPResult(
            u_optimal=qp_result.u_optimal,
            y_nominal=y_opt,
            u_nominal=u_nom,
            J_optimal=qp_result.cost,
            n_iterations=1,  # Single QP, no SCP
            converged=True,
            iteration_costs=[qp_result.cost],
            iteration_times=[solve_time],
            status=qp_result.status,
            s_max_final=self.s_max,
            s_min_final=-self.s_max,
        )

        self._last_result = result
        self._last_u_applied = u_k
        self._step_count += 1
        self._total_solve_time += solve_time

        return u_k, result

    def reset(self) -> None:
        """Reset controller state."""
        self._last_result = None
        self._last_u_applied = 0.0
        self._step_count = 0
        self._total_solve_time = 0.0
        self._qp_builder = None

    def get_average_solve_time(self) -> float:
        if self._step_count == 0:
            return 0.0
        return self._total_solve_time / self._step_count
