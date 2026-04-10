# Canonical owner: koopman-mpc-dbs
"""Multi-step Koopman operator model for MPC.

Learns a lifting function psi(z) that maps the state into a space where
dynamics are approximately linear:
    psi(z_{t+k}) ≈ A_k @ psi(z_t) + B_k @ u[0:k]
    y_{t+k} = C @ psi(z_{t+k})

Since psi(z_t) is independent of future u, the prediction is affine in u:
    y_k = C @ A_k @ psi(z_0) + C @ B_k @ u[0:k]
        = e_k + F_k @ u[0:k]

This means MPC is a standard QP — no SCP needed.

Uses K independent (A_k, B_k) pairs (multi-step Koopman), mirroring
the DCNN's K independent networks to avoid error compounding.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class KoopmanEncoder(nn.Module):
    """Lifting function psi: R^n_state -> R^d_lift.

    Uses a skip connection: psi(z) = [z, phi(z)] where phi is a learnable
    MLP. This guarantees injectivity and allows C to trivially extract y.
    """

    def __init__(self, n_state: int = 30, d_extra: int = 34, hidden: int = 128,
                 n_encoder_layers: int = 2):
        super().__init__()
        self.n_state = n_state
        self.d_extra = d_extra
        self.d_lift = n_state + d_extra

        layers = [nn.Linear(n_state, hidden), nn.ReLU()]
        for _ in range(n_encoder_layers - 1):
            layers.extend([nn.Linear(hidden, hidden), nn.ReLU()])
        layers.append(nn.Linear(hidden, d_extra))
        self.phi = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Lift state z to psi(z) = [z, phi(z)]."""
        return torch.cat([z, self.phi(z)], dim=-1)


class Conv1DEncoder(nn.Module):
    """1D convolutional encoder that treats beta and stim as temporal signals.

    Processes the two 15-step histories with 1D convolutions to capture
    local temporal patterns (trends, oscillations) before lifting.
    """

    def __init__(self, n_state: int = 30, d_extra: int = 34, n_state_y: int = 15,
                 n_channels: int = 32, kernel_size: int = 3):
        super().__init__()
        self.n_state = n_state
        self.d_extra = d_extra
        self.d_lift = n_state + d_extra
        self.n_state_y = n_state_y
        self.n_state_u = n_state - n_state_y

        # Conv1d expects (batch, channels, length)
        # Two input channels: beta history and stim history
        self.conv = nn.Sequential(
            nn.Conv1d(2, n_channels, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Conv1d(n_channels, n_channels, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
        )
        # Flatten conv output and project to d_extra
        self.proj = nn.Sequential(
            nn.Linear(n_channels * n_state_y, 128),
            nn.ReLU(),
            nn.Linear(128, d_extra),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        y_hist = z[:, :self.n_state_y]  # (batch, 15)
        u_hist = z[:, self.n_state_y:]  # (batch, 15)
        # Pad u_hist to same length if needed
        if u_hist.shape[1] < y_hist.shape[1]:
            u_hist = torch.nn.functional.pad(u_hist, (0, y_hist.shape[1] - u_hist.shape[1]))
        x = torch.stack([y_hist, u_hist[:, :self.n_state_y]], dim=1)  # (batch, 2, 15)
        h = self.conv(x)  # (batch, n_channels, 15)
        h = h.flatten(1)  # (batch, n_channels * 15)
        phi = self.proj(h)  # (batch, d_extra)
        return torch.cat([z, phi], dim=-1)


class SplitEncoder(nn.Module):
    """Separate MLP encoders for beta and stim histories.

    Each signal gets its own feature extractor before concatenation,
    allowing specialised representation for each modality.
    """

    def __init__(self, n_state: int = 30, d_extra: int = 34, n_state_y: int = 15,
                 hidden: int = 64):
        super().__init__()
        self.n_state = n_state
        self.d_extra = d_extra
        self.d_lift = n_state + d_extra
        self.n_state_y = n_state_y
        self.n_state_u = n_state - n_state_y

        d_beta = d_extra * 2 // 3  # 2/3 of features from beta
        d_stim = d_extra - d_beta  # 1/3 from stim

        self.beta_enc = nn.Sequential(
            nn.Linear(n_state_y, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, d_beta),
        )
        self.stim_enc = nn.Sequential(
            nn.Linear(n_state - n_state_y, hidden), nn.ReLU(),
            nn.Linear(hidden, d_stim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        y_hist = z[:, :self.n_state_y]
        u_hist = z[:, self.n_state_y:]
        phi_y = self.beta_enc(y_hist)
        phi_u = self.stim_enc(u_hist)
        return torch.cat([z, phi_y, phi_u], dim=-1)


class HandcraftedEncoder(nn.Module):
    """Handcrafted features based on known beta/stim dynamics.

    Features: raw state + temporal derivatives + moving averages +
    cross-correlations. No learned parameters except a small
    projection to match d_lift.
    """

    def __init__(self, n_state: int = 30, d_extra: int = 34, n_state_y: int = 15):
        super().__init__()
        self.n_state = n_state
        self.d_extra = d_extra
        self.d_lift = n_state + d_extra
        self.n_state_y = n_state_y

        # Handcrafted features from beta history:
        # - 1st differences (14)
        # - 2nd differences (13)
        # - 3-step moving average (13)
        # - mean, std, min, max, trend slope (5)
        # - stim: mean, max, sum (3)
        # Total: ~48 features
        n_handcrafted = 48
        # Small projection to d_extra (mostly linear, preserves interpretability)
        self.proj = nn.Sequential(
            nn.Linear(n_handcrafted, d_extra),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        batch = z.shape[0]
        y = z[:, :self.n_state_y]  # (batch, 15)
        u = z[:, self.n_state_y:]  # (batch, 15)

        feats = []
        # 1st differences
        dy = y[:, 1:] - y[:, :-1]  # (batch, 14)
        feats.append(dy)
        # 2nd differences
        ddy = dy[:, 1:] - dy[:, :-1]  # (batch, 13)
        feats.append(ddy)
        # 3-step moving average
        ma3 = (y[:, :-2] + y[:, 1:-1] + y[:, 2:]) / 3.0  # (batch, 13)
        feats.append(ma3)
        # Summary stats
        feats.append(y.mean(dim=1, keepdim=True))
        feats.append(y.std(dim=1, keepdim=True))
        feats.append(y.min(dim=1, keepdim=True).values)
        feats.append(y.max(dim=1, keepdim=True).values)
        # Linear trend slope (simple regression coefficient)
        t_axis = torch.arange(self.n_state_y, dtype=z.dtype, device=z.device)
        t_mean = t_axis.mean()
        slope = ((t_axis - t_mean).unsqueeze(0) * (y - y.mean(dim=1, keepdim=True))).sum(dim=1, keepdim=True) / ((t_axis - t_mean) ** 2).sum()
        feats.append(slope)
        # Stim features
        feats.append(u.mean(dim=1, keepdim=True))
        feats.append(u.max(dim=1, keepdim=True).values)
        feats.append(u.sum(dim=1, keepdim=True))

        h = torch.cat(feats, dim=1)  # (batch, 48)
        phi = self.proj(h)
        return torch.cat([z, phi], dim=-1)


class EDMDEncoder(nn.Module):
    """LASSO-selected features from EDMD analysis.

    Features selected by data-driven process:
    1. Run LASSO regression: y_{t+1} = w @ features(z_t) + b
    2. Features with nonzero coefficients are predictive of next-step dynamics
    3. Use these as the Koopman lifting dictionary

    Selected features (by |coefficient|):
    - 14 first differences dy_0..dy_13 (rate of change dominates prediction)
    - 1 second difference ddy_11 (acceleration at recent step)
    - 3 quadratic: y14², y13², y14*y13 (mild nonlinearity)
    - 1 cross-term: y14*u14 (state-dependent stim sensitivity)
    - 3 summary: y_mean, y_std, u_sum

    Total: 22 derived features + 30 raw state (skip connection) = 52 lifted dims.
    Only the small projection layer is learned (22 -> d_extra mapping).
    """

    def __init__(self, n_state: int = 30, d_extra: int = 34, n_state_y: int = 15):
        super().__init__()
        self.n_state = n_state
        self.d_extra = d_extra
        self.d_lift = n_state + d_extra
        self.n_state_y = n_state_y

        # 14 dy + 1 ddy + 3 quad + 1 cross + 3 summary = 22 features
        n_edmd = 22
        self.proj = nn.Sequential(
            nn.Linear(n_edmd, d_extra),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        y = z[:, :self.n_state_y]   # (batch, 15)
        u = z[:, self.n_state_y:]   # (batch, 15)

        feats = []
        # 14 first differences (dy_0 .. dy_13)
        dy = y[:, 1:] - y[:, :-1]  # (batch, 14)
        feats.append(dy)

        # 1 second difference at step 11
        ddy_11 = (y[:, 13] - 2 * y[:, 12] + y[:, 11]).unsqueeze(1)  # (batch, 1)
        feats.append(ddy_11)

        # 3 quadratic features
        feats.append((y[:, 14] ** 2).unsqueeze(1))       # y14²
        feats.append((y[:, 13] ** 2).unsqueeze(1))       # y13²
        feats.append((y[:, 14] * y[:, 13]).unsqueeze(1))  # y14*y13

        # 1 cross-term: y14 * u14 (state-dependent stim sensitivity)
        feats.append((y[:, 14] * u[:, 14]).unsqueeze(1))

        # 3 summary stats
        feats.append(y.mean(dim=1, keepdim=True))    # y_mean
        feats.append(y.std(dim=1, keepdim=True))     # y_std
        feats.append(u.sum(dim=1, keepdim=True))     # u_sum

        h = torch.cat(feats, dim=1)  # (batch, 22)
        phi = self.proj(h)
        return torch.cat([z, phi], dim=-1)


class EDMD7Encoder(nn.Module):
    """7 LASSO-selected features from expanded dictionary analysis.

    Selected by LassoCV from 95 candidates — the minimal set that
    maximises linear prediction accuracy:
    - dy_13, dy_11: most recent first differences (rate of change)
    - ddy_12, ddy_10, ddy_8: second differences (acceleration at 3 lags)
    - y14*u14: state-dependent stim sensitivity
    - y_range: max-min of beta history (volatility)
    """

    def __init__(self, n_state: int = 30, d_extra: int = 34, n_state_y: int = 15):
        super().__init__()
        self.n_state = n_state
        self.d_extra = d_extra
        self.d_lift = n_state + d_extra
        self.n_state_y = n_state_y
        n_feats = 7
        self.proj = nn.Linear(n_feats, d_extra)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        y = z[:, :self.n_state_y]
        u = z[:, self.n_state_y:]
        feats = [
            (y[:, 14] - y[:, 13]).unsqueeze(1),                          # dy_13
            (y[:, 12] - y[:, 11]).unsqueeze(1),                          # dy_11
            (y[:, 14] * u[:, 14]).unsqueeze(1),                          # y14*u14
            (y[:, 14] - 2*y[:, 13] + y[:, 12]).unsqueeze(1),            # ddy_12
            (y.max(dim=1).values - y.min(dim=1).values).unsqueeze(1),   # y_range
            (y[:, 12] - 2*y[:, 11] + y[:, 10]).unsqueeze(1),            # ddy_10
            (y[:, 10] - 2*y[:, 9] + y[:, 8]).unsqueeze(1),              # ddy_8
        ]
        h = torch.cat(feats, dim=1)
        phi = self.proj(h)
        return torch.cat([z, phi], dim=-1)


class PrincipledEncoder(nn.Module):
    """Features selected by deconfounding score: predictive of y residual
    AND orthogonal to u_future.

    Selected by: score = corr(f, AR_residual) * (1 - corr(f, u_future))

    Features (8 total = 1 intercept + 7 nonlinear):
    - intercept (constant 1)
    - |dy_12|: abs first difference at lag 12 (beta volatility)
    - y_var: variance of beta history (volatility)
    - y_range: max-min of beta history (volatility)
    - y_std: std of beta history (volatility)
    - y14*u14: newest y × newest u (state-dependent sensitivity)
    - y13*u14: second-newest y × newest u
    - u_std: std of stim history (stim variability)
    """

    def __init__(self, n_state: int = 30, d_extra: int = 34, n_state_y: int = 15):
        super().__init__()
        self.n_state = n_state
        self.d_extra = d_extra
        self.d_lift = n_state + d_extra
        self.n_state_y = n_state_y
        n_feats = 8  # intercept + 7 nonlinear
        self.proj = nn.Linear(n_feats, d_extra)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        y = z[:, :self.n_state_y]
        u = z[:, self.n_state_y:]
        feats = [
            torch.ones(z.shape[0], 1, device=z.device, dtype=z.dtype),  # intercept
            torch.abs(y[:, 13] - y[:, 12]).unsqueeze(1),                 # |dy_12|
            y.var(dim=1, keepdim=True),                                   # y_var
            (y.max(dim=1).values - y.min(dim=1).values).unsqueeze(1),   # y_range
            y.std(dim=1, keepdim=True),                                   # y_std
            (y[:, 14] * u[:, 14]).unsqueeze(1),                          # y14*u14
            (y[:, 13] * u[:, 14]).unsqueeze(1),                          # y13*u14
            u.std(dim=1, keepdim=True),                                   # u_std
        ]
        h = torch.cat(feats, dim=1)
        phi = self.proj(h)
        return torch.cat([z, phi], dim=-1)


class Lasso46Encoder(nn.Module):
    """46 features selected by LASSO on prediction error from wide dictionary.

    Combines linear features (signed dy, y_mean, u_sum) for pseudoinverse
    conditioning with nonlinear features (|dy|, |ddy|, quadratics, cross-terms)
    for function space expansion. Selected by LassoCV on 1-step prediction.
    """

    def __init__(self, n_state: int = 30, d_extra: int = 34, n_state_y: int = 15):
        super().__init__()
        self.n_state = n_state
        self.d_extra = d_extra
        self.d_lift = n_state + d_extra
        self.n_state_y = n_state_y
        self.proj = nn.Linear(46, d_extra)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        y = z[:, :self.n_state_y]
        u = z[:, self.n_state_y:]
        feats = []
        # 14 signed first differences (linear)
        for i in range(14):
            feats.append((y[:, i+1] - y[:, i]).unsqueeze(1))
        # y_mean, u_sum (linear)
        feats.append(y.mean(dim=1, keepdim=True))
        feats.append(u.sum(dim=1, keepdim=True))
        # 7 abs first differences (nonlinear)
        for i in [0, 1, 7, 10, 11, 12, 13]:
            feats.append(torch.abs(y[:, i+1] - y[:, i]).unsqueeze(1))
        # 9 abs second differences (nonlinear)
        for i in [1, 2, 4, 6, 8, 9, 10, 11, 12]:
            feats.append(torch.abs(y[:, i+2] - 2*y[:, i+1] + y[:, i]).unsqueeze(1))
        # 12 quadratic y products (nonlinear)
        for i, j in [(10,12),(10,13),(10,14),(11,11),(11,13),(11,14),
                      (12,12),(12,13),(12,14),(13,13),(13,14),(14,14)]:
            feats.append((y[:, i] * y[:, j]).unsqueeze(1))
        # 1 cross-term (nonlinear)
        feats.append((y[:, 12] * u[:, 14]).unsqueeze(1))
        # y_range (nonlinear)
        feats.append((y.max(dim=1).values - y.min(dim=1).values).unsqueeze(1))
        h = torch.cat(feats, dim=1)
        return torch.cat([z, self.proj(h)], dim=-1)


class RobustEncoder(nn.Module):
    """Features selected by multi-step LASSO with robust consistency filtering.

    Selection process:
    1. Build nonlinear candidate dictionary (abs differences, quadratics, cross-terms)
    2. For each horizon k=1..7: fit AR on raw state, LASSO on residual
    3. Keep features selected in >= 4/7 steps (robust across horizons)

    47 features: abs differences, abs second differences, y quadratics,
    y*u cross-terms, volatility stats. Plus intercept = 48 total.
    """

    def __init__(self, n_state: int = 30, d_extra: int = 34, n_state_y: int = 15):
        super().__init__()
        self.n_state = n_state
        self.d_extra = d_extra
        self.d_lift = n_state + d_extra
        self.n_state_y = n_state_y
        # Identity projection (features are used directly)
        self.proj = nn.Linear(48, d_extra)

    def _compute_features(self, y: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        feats = [torch.ones(y.shape[0], 1, device=y.device, dtype=y.dtype)]  # intercept

        # Abs first differences (selected lags)
        for i in [0, 1, 2, 4, 5, 8, 10, 11, 12, 13]:
            feats.append(torch.abs(y[:, i+1] - y[:, i]).unsqueeze(1))

        # Abs second differences (selected lags)
        for i in [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
            feats.append(torch.abs(y[:, i+2] - 2*y[:, i+1] + y[:, i]).unsqueeze(1))

        # Quadratic y products
        for i, j in [(10,10),(10,11),(10,12),(10,13),(10,14),
                      (11,11),(11,12),(11,13),(11,14),
                      (12,12),(12,13),(12,14),
                      (13,13),(13,14),(14,14)]:
            feats.append((y[:, i] * y[:, j]).unsqueeze(1))

        # Cross y*u terms
        for i, j in [(14,14),(14,13),(10,14),(10,10),(10,11),(10,12)]:
            feats.append((y[:, i] * u[:, j]).unsqueeze(1))

        # Summary stats
        feats.append(y.var(dim=1, keepdim=True))
        feats.append(y.std(dim=1, keepdim=True))
        feats.append((y.max(dim=1).values - y.min(dim=1).values).unsqueeze(1))
        feats.append(u.std(dim=1, keepdim=True))

        return torch.cat(feats, dim=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        y = z[:, :self.n_state_y]
        u = z[:, self.n_state_y:]
        h = self._compute_features(y, u)
        phi = self.proj(h)
        return torch.cat([z, phi], dim=-1)


def create_encoder(encoder_type: str, n_state: int = 30, d_extra: int = 34,
                   hidden: int = 128, n_encoder_layers: int = 2,
                   n_state_y: int = 15) -> nn.Module:
    """Factory for encoder variants."""
    if encoder_type == "mlp":
        return KoopmanEncoder(n_state, d_extra, hidden, n_encoder_layers)
    elif encoder_type == "conv1d":
        return Conv1DEncoder(n_state, d_extra, n_state_y)
    elif encoder_type == "split":
        return SplitEncoder(n_state, d_extra, n_state_y, hidden=hidden // 2)
    elif encoder_type == "handcrafted":
        return HandcraftedEncoder(n_state, d_extra, n_state_y)
    elif encoder_type == "edmd":
        return EDMDEncoder(n_state, d_extra, n_state_y)
    elif encoder_type == "edmd7":
        return EDMD7Encoder(n_state, d_extra, n_state_y)
    elif encoder_type == "principled":
        return PrincipledEncoder(n_state, d_extra, n_state_y)
    elif encoder_type == "robust":
        return RobustEncoder(n_state, d_extra, n_state_y)
    elif encoder_type == "lasso46":
        return Lasso46Encoder(n_state, d_extra, n_state_y)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


class MultiStepKoopman(nn.Module):
    """K independent Koopman operators for multi-step prediction.

    Same interface as MultiStepDCNN: forward_k(x, u, k) and forward(x, u).

    Attributes:
        encoder: Lifting function psi.
        A: K matrices, each (d_lift, d_lift).
        B: K matrices, B_k has shape (d_lift, k).
        C: Output projection (1, d_lift).
        horizon: Number of prediction steps K.
        n_state: Original state dimension.
        d_lift: Lifted state dimension.
    """

    def __init__(
        self,
        n_state: int = 30,
        n_input: int = 1,
        d_lift: int = 64,
        hidden: int = 128,
        horizon: int = 5,
        n_encoder_layers: int = 2,
        encoder_type: str = "mlp",
        n_state_y: int = 15,
    ):
        super().__init__()
        self.n_state = n_state
        self.n_input = n_input
        self.d_lift = d_lift
        self.n_hidden = hidden
        self.n_layers = n_encoder_layers
        self.horizon = horizon
        self.encoder_type = encoder_type

        d_extra = d_lift - n_state
        self.encoder = create_encoder(
            encoder_type, n_state, d_extra, hidden, n_encoder_layers, n_state_y,
        )

        # K independent linear operators
        self.A = nn.ParameterList([
            nn.Parameter(torch.eye(d_lift) + 0.01 * torch.randn(d_lift, d_lift))
            for _ in range(horizon)
        ])
        self.B = nn.ParameterList([
            nn.Parameter(0.01 * torch.randn(d_lift, (k + 1) * n_input))
            for k in range(horizon)
        ])

        # Output projection: initialized to pick out y from psi
        # psi = [z, phi(z)] where z = [y_past(n_state_y), u_past(n_state_u)]
        # Newest y is at index n_state_y - 1 = 14 (oldest-to-newest)
        self.C = nn.Parameter(torch.zeros(1, d_lift))
        # Will be initialized properly in init_output_projection()

    def init_output_projection(self, n_state_y: int = 15):
        """Initialize C to extract newest y from psi = [z, phi(z)]."""
        with torch.no_grad():
            self.C.zero_()
            self.C[0, n_state_y - 1] = 1.0

    def forward_k(self, x: torch.Tensor, u: torch.Tensor, k: int) -> torch.Tensor:
        """Predict k steps ahead.

        Args:
            x: State tensor (batch, n_state).
            u: Control sequence (batch, horizon).
            k: Prediction step (1 to horizon).

        Returns:
            Prediction (batch, 1).
        """
        psi = self.encoder(x)  # (batch, d_lift)
        u_k = u[:, :k]  # (batch, k)
        psi_k = psi @ self.A[k - 1].T + u_k @ self.B[k - 1].T  # (batch, d_lift)
        return (psi_k @ self.C.T)  # (batch, 1)

    def forward(
        self, x: torch.Tensor, u: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        """Predict all steps 1 to horizon."""
        psi = self.encoder(x)
        preds = []
        for k in range(1, self.horizon + 1):
            u_k = u[:, :k]
            psi_k = psi @ self.A[k - 1].T + u_k @ self.B[k - 1].T
            preds.append(psi_k @ self.C.T)
        return tuple(preds)

    def get_qp_matrices(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extract E, F matrices for QP: y_k = e_k + F_k @ u[0:k].

        Args:
            z: Current state, shape (n_state,).

        Returns:
            e: Constant terms, shape (N,). e_k = C @ A_k @ psi(z).
            F: Jacobian matrix, shape (N, N), lower-triangular.
                F[k, j] = C @ B_k[:, j] for j < k, else 0.
        """
        N = self.horizon
        with torch.no_grad():
            z_t = torch.tensor(z, dtype=torch.float32).unsqueeze(0)
            psi = self.encoder(z_t).squeeze(0)  # (d_lift,)

            e = np.zeros(N)
            F = np.zeros((N, N))

            C = self.C.squeeze(0).numpy()  # (d_lift,)

            for k in range(N):
                A_k = self.A[k].numpy()  # (d_lift, d_lift)
                B_k = self.B[k].numpy()  # (d_lift, k+1)

                e[k] = C @ A_k @ psi.numpy()

                # F[k, j] = C @ B_k[:, j] for j <= k
                for j in range(k + 1):
                    F[k, j] = C @ B_k[:, j]

        return e, F

    def lift(self, z: np.ndarray) -> np.ndarray:
        """Lift state to Koopman space."""
        with torch.no_grad():
            z_t = torch.tensor(z, dtype=torch.float32).unsqueeze(0)
            return self.encoder(z_t).squeeze(0).numpy()


def load_koopman_model(model_dir: str | Path) -> MultiStepKoopman:
    """Load a trained Koopman model.

    Args:
        model_dir: Directory containing koopman_model.pt and config.json.

    Returns:
        MultiStepKoopman ready for inference.
    """
    model_dir = Path(model_dir)

    with open(model_dir / "config.json") as f:
        config = json.load(f)

    model = MultiStepKoopman(
        n_state=config.get("n_state", 30),
        n_input=config.get("n_input", 1),
        d_lift=config.get("d_lift", 64),
        hidden=config.get("hidden", 128),
        horizon=config.get("horizon", 5),
        n_encoder_layers=config.get("n_encoder_layers", 2),
        encoder_type=config.get("encoder_type", "mlp"),
        n_state_y=config.get("n_state_y", 15),
    )
    model.load_state_dict(
        torch.load(model_dir / "koopman_model.pt", map_location="cpu", weights_only=True)
    )
    model.eval()

    model.n_state_y = config.get("n_state_y", 15)
    model.n_state_u = config.get("n_state_u", 15)

    return model
