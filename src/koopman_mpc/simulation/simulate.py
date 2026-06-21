"""Koopman MPC simulation integration with dbs_bench.

Shows how to wrap KoopmanController as a ControllerProtocol
and plug it into the bench SimulationRunner.

Example:
    >>> from koopman_mpc.simulation.simulate import run_koopman_mpc
    >>> from koopman_mpc.models.koopman_model import load_koopman_model
    >>> predictor = load_koopman_model("models/koopman_lasso46")
    >>> result = run_koopman_mpc(predictor, duration=60.0)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from koopman_mpc.config.device_config import get_device_config
from koopman_mpc.controllers.koopman_controller import KoopmanController, KoopmanMPCConfig
from koopman_mpc.controllers.scp_result import SCPResult

_CFG = get_device_config()


class KoopmanControllerAdapter:
    """Wraps KoopmanController to satisfy dbs_bench ControllerProtocol.

    dbs_bench.simulation.simulate.ControllerProtocol requires:
        compute_control(y_history, u_history, u_prev) -> (float, dict)
        reset() -> None

    KoopmanController.compute_control returns (float, SCPResult).
    This adapter converts SCPResult to a plain dict.
    """

    def __init__(
        self,
        predictor,
        config: Optional[KoopmanMPCConfig] = None,
        w_bounds: Optional[np.ndarray] = None,
    ):
        if config is None:
            config = KoopmanMPCConfig()
        self._ctrl = KoopmanController(predictor, config, W_bounds=w_bounds)

    def compute_control(
        self,
        y_history: np.ndarray,
        u_history: np.ndarray,
        u_prev: float,
    ):
        u, result = self._ctrl.compute_control(y_history, u_history, u_prev)
        info = {
            "solve_time": sum(result.iteration_times),
            "cost": result.J_optimal,
            "converged": result.converged,
            "status": result.status,
        }
        return u, info

    def reset(self) -> None:
        self._ctrl.reset()


def run_koopman_mpc(
    predictor,
    duration: float = 60.0,
    dt: float = 0.02,
    beta_0: Optional[float] = None,
    w_bounds: Optional[np.ndarray] = None,
    n_state_y: int = 15,
    seed: int = 42,
    show_progress: bool = True,
):
    """Run Koopman MPC on synthetic beta data.

    Requires dbs_bench to be installed:
        pip install closed-loop-dbs-bench  # or pip install -e ../closed-loop-dbs-bench

    Args:
        predictor: Trained MultiStepKoopman model.
        duration: Simulation duration in seconds.
        dt: Sample period in seconds.
        beta_0: Control threshold. Defaults to device config value.
        w_bounds: Tube bounds (N, 2) from training. Defaults to 0.1.
        n_state_y: History buffer length.
        seed: Random seed for synthetic data.
        show_progress: Show tqdm progress bar.

    Returns:
        SimulationResult from dbs_bench.simulation.simulate.
    """
    from dbs_bench.simulation.simulate import SimulationRunner
    from dbs_bench.synthetic.data_generator import generate_demo_patient

    if beta_0 is None:
        beta_0 = float(_CFG.beta.default_threshold)

    config = KoopmanMPCConfig(
        prediction_horizon=predictor.horizon,
        beta_0=beta_0,
    )
    ctrl = KoopmanControllerAdapter(predictor, config, w_bounds=w_bounds)
    patient = generate_demo_patient(n_state_y=n_state_y, seed=seed)
    runner = SimulationRunner(patient, dt=dt, beta_0=beta_0)

    return runner.run(
        ctrl,
        duration=duration,
        controller_type="custom",
        show_progress=show_progress,
    )
