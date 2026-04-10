"""Tests for DenseQPBuilder (requires piqp)."""
import numpy as np
import pytest


def test_dense_qp_builder_import():
    """DenseQPBuilder imports without error."""
    from koopman_mpc.controllers.dense_qp_builder import DenseQPBuilder
    assert DenseQPBuilder is not None


def test_dense_qp_basic(monkeypatch):
    """DenseQPBuilder.solve returns DenseQPResult with correct shape."""
    pytest.importorskip("piqp")
    from koopman_mpc.controllers.dense_qp_builder import DenseQPBuilder

    N = 5
    qp = DenseQPBuilder(N=N, Q=1.0, R=0.01, beta_0=2.3,
                         u_min=0.0, u_max=0.03, delta_u_max=0.01)

    rng = np.random.default_rng(0)
    y_nom = rng.uniform(2.0, 2.5, size=N)
    F = np.tril(rng.standard_normal((N, N)) * 0.1)
    u_nom = np.full(N, 0.01)
    s_max = np.full(N, 0.1)

    result = qp.solve(y_nom, F, u_nom, u_prev=0.01, s_max=s_max)
    assert result.u_optimal.shape == (N,)
    assert np.all(result.u_optimal >= -1e-6)   # u >= 0
    assert np.all(result.u_optimal <= 0.03 + 1e-6)  # u <= u_max


def test_scp_result_dataclass():
    """SCPResult dataclass works as expected."""
    from koopman_mpc.controllers.scp_result import SCPResult

    r = SCPResult(
        u_optimal=np.array([0.01, 0.01]),
        y_nominal=np.array([2.2, 2.1]),
        u_nominal=np.array([0.01, 0.01]),
        J_optimal=0.5,
        n_iterations=1,
        converged=True,
    )
    assert r.n_iterations == 1
    assert r.converged is True
    assert r.status == "SUCCESS"
