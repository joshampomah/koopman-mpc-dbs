"""Tests for OLS training pipeline."""
import numpy as np
import pytest


def test_compute_lasso46_features():
    """Lasso46 feature computation returns (N, 46)."""
    from koopman_mpc.training.train_koopman_ols import compute_lasso46_features
    rng = np.random.default_rng(0)
    z = rng.standard_normal((20, 30)).astype(np.float32)
    feats = compute_lasso46_features(z)
    assert feats.shape == (20, 46)


def test_prepare_multistep_sequences():
    """Sequence extraction produces correct shapes."""
    from koopman_mpc.training.train_koopman_ols import prepare_multistep_sequences
    rng = np.random.default_rng(0)
    T = 500
    beta = rng.standard_normal(T).astype(np.float32)
    stim = rng.random(T).astype(np.float32) * 0.03
    seqs = prepare_multistep_sequences(beta, stim, n_state_y=15, n_state_u=15, horizon=7)
    n_expected = T - 30 - 7 + 1
    assert seqs["x"].shape == (n_expected, 30)
    assert seqs["u"].shape == (n_expected, 7)
    assert seqs["y"].shape == (n_expected, 7)


def test_fit_ols_koopman_arx():
    """OLS ARX model fits on small synthetic dataset."""
    from koopman_mpc.training.train_koopman_ols import (
        fit_ols_koopman, prepare_multistep_sequences
    )
    from koopman_mpc.synthetic.data_generator import generate_modulated_beta

    beta, stim = generate_modulated_beta(n_steps=800, seed=42)
    seqs = prepare_multistep_sequences(beta, stim, horizon=3)

    n = len(seqs["x"])
    split = int(0.8 * n)
    model, w_bounds, metrics = fit_ols_koopman(
        seqs["x"][:split], seqs["u"][:split], seqs["y"][:split],
        seqs["x"][split:], seqs["u"][split:], seqs["y"][split:],
        model_type="arx", horizon=3, verbose=False,
    )
    assert w_bounds.shape == (3, 2)
    assert len(metrics["test_mae"]) == 3
    assert all(np.isfinite(m) for m in metrics["test_mae"])
