"""Tests for OLS training pipeline."""
import json

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
    n_expected = T - 15 - 7
    assert seqs["x"].shape == (n_expected, 30)
    assert seqs["u"].shape == (n_expected, 7)
    assert seqs["y"].shape == (n_expected, 7)


def _write_patient_folder(root, name, beta, stim):
    patient_dir = root / name
    patient_dir.mkdir(parents=True)
    np.savetxt(patient_dir / "beta_causal_RMS.csv", beta, delimiter=",")
    np.savetxt(patient_dir / "stimulation.csv", stim, delimiter=",")
    return patient_dir


def test_load_csv_training_data_with_selected_patients(tmp_path):
    """CSV loader accepts 4YP-style selected patient roots."""
    from koopman_mpc.training.train_koopman_ols import load_csv_training_data

    beta = np.linspace(1.0, 2.0, 80, dtype=np.float32)
    stim = np.linspace(0.0, 0.03, 80, dtype=np.float32)
    _write_patient_folder(tmp_path, "patient_training", beta, np.repeat(stim, 4))
    _write_patient_folder(tmp_path, "patient_refinement", beta, stim)
    (tmp_path / "selected_patients.json").write_text(
        json.dumps(
            {
                "patients": [
                    {"directory": "patient_training", "role": "training"},
                    {"directory": "patient_refinement", "role": "refinement"},
                ]
            }
        )
    )

    (x_tr, u_tr, y_tr), (x_te, u_te, y_te), label = load_csv_training_data(
        tmp_path,
        horizon=3,
        patient_role="training",
        test_fraction=0.25,
    )

    assert label.endswith(":training")
    assert x_tr.shape[1] == 30
    assert u_tr.shape[1] == 3
    assert y_tr.shape[1] == 3
    assert len(x_tr) + len(x_te) == 62
    assert np.isclose(u_tr[0, 0], stim[15])
    assert u_te.shape == y_te.shape


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
