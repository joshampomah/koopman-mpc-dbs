"""Quick Koopman MPC demo — train a small model, run simulation.

Completes in under 60 seconds on CPU.

Run:
    python examples/quick_demo.py
"""
from __future__ import annotations

import numpy as np
import torch

from koopman_mpc.models.koopman_model import MultiStepKoopman
from koopman_mpc.training.train_koopman_ols import (
    fit_ols_koopman, prepare_multistep_sequences, load_training_data
)
from koopman_mpc.simulation.simulate import run_koopman_mpc


def main() -> None:
    """Train a small Koopman model and run a short closed-loop demo."""
    print("Generating synthetic training data ...")
    (x_tr, u_tr, y_tr), (x_te, u_te, y_te), _ = load_training_data(
        horizon=5,
        n_augmentations=1,
        modulation_seed=42,
        train_duration=300.0,  # 5 min — fast demo
    )
    print(f"  {x_tr.shape[0]} train / {x_te.shape[0]} test samples")

    print("Training lasso46 model by OLS ...")
    model, w_bounds, metrics = fit_ols_koopman(
        x_tr, u_tr, y_tr, x_te, u_te, y_te,
        model_type="lasso46", horizon=5, verbose=True,
    )

    print("\nRunning Koopman MPC simulation (10 s) ...")
    result = run_koopman_mpc(
        model, duration=10.0, w_bounds=w_bounds, seed=0, show_progress=False
    )

    print("\n--- Simulation Metrics ---")
    for k, v in result.metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")

    print("\nDemo complete.")


if __name__ == "__main__":
    main()
