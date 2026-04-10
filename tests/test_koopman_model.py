"""Tests for Koopman model architecture."""
import numpy as np
import pytest
import torch

from koopman_mpc.models.koopman_model import (
    MultiStepKoopman,
    Lasso46Encoder,
    create_encoder,
    load_koopman_model,
)


def test_multistep_koopman_forward():
    """MultiStepKoopman forward pass returns correct shapes."""
    model = MultiStepKoopman(n_state=30, n_input=1, d_lift=64, horizon=5)
    x = torch.randn(4, 30)
    u = torch.randn(4, 5)
    preds = model(x, u)
    assert len(preds) == 5
    for p in preds:
        assert p.shape == (4, 1)


def test_multistep_koopman_forward_k():
    """forward_k returns (batch, 1) for each k."""
    model = MultiStepKoopman(n_state=30, n_input=1, d_lift=64, horizon=3)
    x = torch.randn(8, 30)
    u = torch.randn(8, 3)
    for k in range(1, 4):
        pred = model.forward_k(x, u, k)
        assert pred.shape == (8, 1)


def test_get_qp_matrices():
    """get_qp_matrices returns (e, F) with correct shapes."""
    model = MultiStepKoopman(n_state=30, n_input=1, d_lift=40, horizon=5)
    z = np.random.randn(30).astype(np.float32)
    e, F = model.get_qp_matrices(z)
    assert e.shape == (5,)
    assert F.shape == (5, 5)
    # F should be lower-triangular (F[k,j] = 0 for j > k)
    for k in range(5):
        for j in range(k + 1, 5):
            assert F[k, j] == 0.0, f"F[{k},{j}] should be 0"


def test_lift():
    """lift() returns (d_lift,) array."""
    model = MultiStepKoopman(n_state=30, n_input=1, d_lift=64, horizon=3)
    z = np.random.randn(30).astype(np.float32)
    psi = model.lift(z)
    assert psi.shape == (64,)


def test_lasso46_encoder():
    """Lasso46Encoder produces correct output dimension."""
    enc = Lasso46Encoder(n_state=30, d_extra=46, n_state_y=15)
    x = torch.randn(8, 30)
    out = enc(x)
    assert out.shape == (8, 76)


def test_create_encoder_variants():
    """create_encoder handles all supported encoder types."""
    for enc_type in ["mlp", "lasso46", "edmd", "edmd7", "principled", "robust"]:
        enc = create_encoder(enc_type, n_state=30, d_extra=34, n_state_y=15)
        x = torch.randn(4, 30)
        out = enc(x)
        assert out.shape == (4, 64), f"Failed for {enc_type}"


def test_save_load_round_trip(tmp_path):
    """Model round-trips through save/load."""
    model = MultiStepKoopman(n_state=30, n_input=1, d_lift=64, horizon=3,
                              encoder_type="lasso46", n_state_y=15)
    import json

    # Save
    torch.save(model.state_dict(), tmp_path / "koopman_model.pt")
    config = {
        "n_state": 30, "n_input": 1, "d_lift": 64, "hidden": 128,
        "horizon": 3, "n_encoder_layers": 2, "encoder_type": "lasso46",
        "n_state_y": 15,
    }
    with open(tmp_path / "config.json", "w") as f:
        json.dump(config, f)

    loaded = load_koopman_model(tmp_path)
    assert loaded.horizon == 3
    assert loaded.d_lift == 64

    # Same outputs
    z = np.random.randn(30).astype(np.float32)
    e1, F1 = model.get_qp_matrices(z)
    e2, F2 = loaded.get_qp_matrices(z)
    np.testing.assert_allclose(e1, e2, atol=1e-5)
    np.testing.assert_allclose(F1, F2, atol=1e-5)
