"""Run Koopman MPC on synthetic beta data.

Requires a trained model. Train one first:
    python -m koopman_mpc.training.train_koopman_ols --save-dir models/koopman_lasso46

Usage:
    python scripts/run_koopman_mpc.py --model models/koopman_lasso46 --duration 60
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from koopman_mpc.models.koopman_model import load_koopman_model
from koopman_mpc.simulation.simulate import run_koopman_mpc


def main() -> None:
    """Load a trained Koopman model and run a closed-loop simulation."""
    p = argparse.ArgumentParser(description="Run Koopman MPC on synthetic data")
    p.add_argument("--model", required=True, help="Path to trained model directory")
    p.add_argument("--duration", type=float, default=60.0, help="Simulation duration (s)")
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-progress", action="store_true")
    args = p.parse_args()

    print(f"Loading model from {args.model}...")
    predictor = load_koopman_model(args.model)

    # Load w_bounds if available
    w_bounds_path = Path(args.model) / "w_bounds.npy"
    w_bounds = np.load(w_bounds_path) if w_bounds_path.exists() else None

    print(f"Running Koopman MPC (duration={args.duration}s, horizon={predictor.horizon})...")
    result = run_koopman_mpc(
        predictor,
        duration=args.duration,
        dt=args.dt,
        w_bounds=w_bounds,
        seed=args.seed,
        show_progress=not args.no_progress,
    )

    print("\n--- Results ---")
    for k, v in result.metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
