#!/usr/bin/env python3
"""Train Koopman lasso46 and Multi-step ARX models by OLS on synthetic data.

Both models use the same MultiStepKoopman interface so they slot directly
into the KoopmanController.

Koopman lasso46:
    psi(z) = [z, features_46(z)]  (76-dim)
    y_{t+k} = C @ A_k @ psi(z_t) + C @ B_k @ u[0:k]
    C = e_14, A_k/B_k fitted by OLS on [psi, u] -> y

Multi-step ARX (+ intercept):
    psi(z) = [z, 1]  (31-dim)
    Same prediction equation, C = e_30, OLS-fitted.

Usage:
    python train_koopman_ols.py
    python train_koopman_ols.py --model lasso46 --horizon 7
    python train_koopman_ols.py --model arx --save-dir ./my_model
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from koopman_mpc.models.koopman_model import MultiStepKoopman
from koopman_mpc.synthetic.data_generator import (
    _build_stim_matrices,
    generate_modulated_beta,
    generate_synthetic_stimulation,
)


# ── Sequence extraction ───────────────────────────────────────────────

def prepare_multistep_sequences(
    beta: np.ndarray,
    stim: np.ndarray,
    n_state_y: int = 15,
    n_state_u: int = 15,
    horizon: int = 7,
) -> dict:
    """Extract overlapping (x, u, y) sequence windows.

    Args:
        beta: Log-space beta signal, shape (T,).
        stim: Stimulation signal, shape (T,).
        n_state_y: Output history length.
        n_state_u: Input history length.
        horizon: Number of prediction steps.

    Returns:
        Dict with 'x' (N, 30), 'u' (N, horizon), 'y' (N, horizon).
    """
    n_history = max(n_state_y, n_state_u)
    T = len(beta)
    n_seq = T - n_history - horizon
    if n_seq <= 0:
        raise ValueError(
            f"Signal too short ({T}) for history={n_history}, horizon={horizon}"
        )

    x = np.empty((n_seq, n_state_y + n_state_u), dtype=np.float32)
    u_fut = np.empty((n_seq, horizon), dtype=np.float32)
    y_fut = np.empty((n_seq, horizon), dtype=np.float32)

    y_start = n_history - n_state_y + 1
    u_start = n_history - n_state_u
    for i in range(n_state_y):
        x[:, i] = beta[y_start + i:y_start + i + n_seq]
    for i in range(n_state_u):
        x[:, n_state_y + i] = stim[u_start + i:u_start + i + n_seq]
    for k in range(horizon):
        u_fut[:, k] = stim[n_history + k:n_history + k + n_seq]
        y_fut[:, k] = beta[n_history + k + 1:n_history + k + 1 + n_seq]

    return {
        "x": x,
        "u": u_fut,
        "y": y_fut,
    }


# ── Data loading ──────────────────────────────────────────────────────

def _read_csv_signal(path: Path) -> np.ndarray:
    """Read a one-column CSV/text signal as float32."""
    try:
        arr = np.loadtxt(path, delimiter=",")
    except ValueError:
        arr = np.loadtxt(path)
    return np.asarray(arr, dtype=np.float32).reshape(-1)


def _align_beta_and_stim(beta: np.ndarray, stim: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Align beta and stimulation, downsampling obvious integer-rate mismatches."""
    if len(beta) == len(stim):
        return beta, stim

    if len(stim) > len(beta):
        ratio = round(len(stim) / len(beta))
        if ratio > 1 and abs(len(stim) / len(beta) - ratio) < 0.05:
            stim = stim[::ratio]
    elif len(beta) > len(stim):
        ratio = round(len(beta) / len(stim))
        if ratio > 1 and abs(len(beta) / len(stim) - ratio) < 0.05:
            beta = beta[::ratio]

    n = min(len(beta), len(stim))
    return beta[:n], stim[:n]


def _patient_dirs(data_dir: Path, patient_role: str = "all") -> list[Path]:
    """Find CSV patient folders, or treat data_dir itself as one folder."""
    if (data_dir / "beta_causal_RMS.csv").exists() and (data_dir / "stimulation.csv").exists():
        return [data_dir]

    selected_path = data_dir / "selected_patients.json"
    if selected_path.exists():
        with open(selected_path) as f:
            selected = json.load(f)
        dirs = []
        for item in selected.get("patients", []):
            if patient_role != "all" and item.get("role") != patient_role:
                continue
            p = data_dir / item["directory"]
            if (p / "beta_causal_RMS.csv").exists() and (p / "stimulation.csv").exists():
                dirs.append(p)
        if dirs:
            return dirs

    return sorted(
        p for p in data_dir.glob("patient_*")
        if (p / "beta_causal_RMS.csv").exists() and (p / "stimulation.csv").exists()
    )


def _apply_synthetic_stimulation(
    beta_log: np.ndarray,
    seed: int,
    dt: float = 0.02,
) -> Tuple[np.ndarray, np.ndarray]:
    """Overlay synthetic PRBS stimulation on an autonomous log-beta trace."""
    stim = generate_synthetic_stimulation(len(beta_log), seed=seed)
    Ad, Bd, Cd = _build_stim_matrices(dt=dt)
    x_eta = np.zeros(2, dtype=np.float64)
    eta = np.empty(len(beta_log), dtype=np.float32)
    for i, u_i in enumerate(stim):
        eta[i] = float((Cd @ x_eta)[0])
        x_eta = Ad @ x_eta + Bd.flatten() * float(u_i)
    return (beta_log - eta).astype(np.float32), stim.astype(np.float32)


def _split_arrays(
    x_all: np.ndarray,
    u_all: np.ndarray,
    y_all: np.ndarray,
    test_fraction: float,
    label: str,
) -> Tuple[Tuple, Tuple, str]:
    """Chronologically split stacked arrays into train/test groups."""
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}")
    if len(x_all) < 2:
        raise ValueError("Need at least two samples for train/test split")

    split = int(round(len(x_all) * (1.0 - test_fraction)))
    split = min(max(split, 1), len(x_all) - 1)
    return (
        (x_all[:split], u_all[:split], y_all[:split]),
        (x_all[split:], u_all[split:], y_all[split:]),
        label,
    )


def load_npz_training_data(
    data_dir: Path,
    test_fraction: float = 0.2,
) -> Tuple[Tuple, Tuple, str]:
    """Load pre-windowed training arrays from one or more `.npz` files.

    Each file must contain `x`, `u`, and `y` arrays. `x` stores
    [y_history, u_history] oldest-to-newest; `u` and `y` store future
    control/output sequences.
    """
    npz_files = sorted(data_dir.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {data_dir}")

    xs, us, ys = [], [], []
    for path in npz_files:
        with np.load(path) as arrays:
            missing = {"x", "u", "y"} - set(arrays.files)
            if missing:
                raise ValueError(f"{path} is missing required arrays: {sorted(missing)}")
            x = np.asarray(arrays["x"], dtype=np.float32)
            u = np.asarray(arrays["u"], dtype=np.float32)
            y = np.asarray(arrays["y"], dtype=np.float32)

        if x.ndim != 2 or u.ndim != 2 or y.ndim != 2:
            raise ValueError(f"{path} arrays must all be 2-D")
        if len(x) != len(u) or len(x) != len(y):
            raise ValueError(f"{path} x/u/y arrays must have the same row count")
        xs.append(x)
        us.append(u)
        ys.append(y)

    x_all = np.vstack(xs)
    u_all = np.vstack(us)
    y_all = np.vstack(ys)
    label = f"custom:{data_dir.name}"
    return _split_arrays(x_all, u_all, y_all, test_fraction, label)


def load_csv_training_data(
    data_dir: Path,
    horizon: int = 7,
    input_space: str = "linear",
    patient_role: str = "all",
    synthetic_stim: bool = False,
    max_samples_per_patient: Optional[int] = None,
    test_fraction: float = 0.2,
    seed: int = 42,
    dt: float = 0.02,
) -> Tuple[Tuple, Tuple, str]:
    """Load Mark/4YP-style beta_causal_RMS.csv + stimulation.csv folders."""
    dirs = _patient_dirs(data_dir, patient_role=patient_role)
    if not dirs:
        raise FileNotFoundError(
            f"No patient CSV folders or .npz windows found in {data_dir}"
        )

    xs, us, ys = [], [], []
    for idx, patient_dir in enumerate(dirs):
        beta = _read_csv_signal(patient_dir / "beta_causal_RMS.csv")
        stim = _read_csv_signal(patient_dir / "stimulation.csv")
        beta, stim = _align_beta_and_stim(beta, stim)

        if max_samples_per_patient is not None and len(beta) > max_samples_per_patient:
            start = (len(beta) - max_samples_per_patient) // 2
            beta = beta[start:start + max_samples_per_patient]
            stim = stim[start:start + max_samples_per_patient]

        if input_space == "linear":
            beta = np.log(np.maximum(beta, 1e-10)).astype(np.float32)
        elif input_space != "log":
            raise ValueError(f"input_space must be 'linear' or 'log', got {input_space}")

        if synthetic_stim:
            beta, stim = _apply_synthetic_stimulation(
                beta,
                seed=seed + idx * 100000,
                dt=dt,
            )

        seqs = prepare_multistep_sequences(beta, stim, horizon=horizon)
        xs.append(seqs["x"])
        us.append(seqs["u"])
        ys.append(seqs["y"])

    x_all = np.vstack(xs)
    u_all = np.vstack(us)
    y_all = np.vstack(ys)
    label = f"csv:{data_dir.name}:{patient_role}"
    return _split_arrays(x_all, u_all, y_all, test_fraction, label)


def load_training_data(
    horizon: int = 7,
    n_augmentations: int = 1,
    modulation_seed: int = 42,
    train_duration: float = 3600.0,  # 1 hour of synthetic data
    test_duration: float = 600.0,    # 10 min test
    dt: float = 0.02,
    data_dir: Optional[Path] = None,
    test_fraction: float = 0.2,
    input_space: str = "linear",
    patient_role: str = "all",
    synthetic_stim: bool = False,
    max_samples_per_patient: Optional[int] = None,
) -> Tuple[Tuple, Tuple, str]:
    """Load custom data or generate synthetic training/test data.

    Uses the AR + PRBS stimulation model from the bench repo.

    Returns:
        (x_train, u_train, y_train), (x_test, u_test, y_test), dataset label
    """
    if data_dir is not None:
        if any(data_dir.glob("*.npz")):
            return load_npz_training_data(data_dir, test_fraction=test_fraction)
        return load_csv_training_data(
            data_dir=data_dir,
            horizon=horizon,
            input_space=input_space,
            patient_role=patient_role,
            synthetic_stim=synthetic_stim,
            max_samples_per_patient=max_samples_per_patient,
            test_fraction=test_fraction,
            seed=modulation_seed,
            dt=dt,
        )

    n_train = int(round(train_duration / dt))
    n_test = int(round(test_duration / dt))

    all_x, all_u, all_y = [], [], []

    for aug in range(n_augmentations):
        seed = modulation_seed + aug * 100000
        beta, stim = generate_modulated_beta(n_train, seed=seed, dt=dt)
        seqs = prepare_multistep_sequences(
            beta.astype(np.float32), stim.astype(np.float32),
            n_state_y=15, n_state_u=15, horizon=horizon,
        )
        all_x.append(seqs["x"])
        all_u.append(seqs["u"])
        all_y.append(seqs["y"])

    x_train = np.concatenate(all_x, axis=0)
    u_train = np.concatenate(all_u, axis=0)
    y_train = np.concatenate(all_y, axis=0)

    # Test data with a different seed
    beta_te, stim_te = generate_modulated_beta(n_test, seed=modulation_seed + 999, dt=dt)
    test_seqs = prepare_multistep_sequences(
        beta_te.astype(np.float32), stim_te.astype(np.float32),
        n_state_y=15, n_state_u=15, horizon=horizon,
    )

    return (x_train, u_train, y_train), (test_seqs["x"], test_seqs["u"], test_seqs["y"]), "synthetic"


# ── Feature computation ──────────────────────────────────────────────

def compute_lasso46_features(z: np.ndarray) -> np.ndarray:
    """Compute 46 analytical features matching Lasso46Encoder.forward().

    Args:
        z: State array, shape (N, 30). First 15 = y history, last 15 = u history.

    Returns:
        features: shape (N, 46).
    """
    y = z[:, :15]
    u = z[:, 15:]
    feats = []
    # 14 signed first differences
    for i in range(14):
        feats.append(y[:, i + 1] - y[:, i])
    # y_mean, u_sum
    feats.append(y.mean(axis=1))
    feats.append(u.sum(axis=1))
    # 7 abs first differences
    for i in [0, 1, 7, 10, 11, 12, 13]:
        feats.append(np.abs(y[:, i + 1] - y[:, i]))
    # 9 abs second differences
    for i in [1, 2, 4, 6, 8, 9, 10, 11, 12]:
        feats.append(np.abs(y[:, i + 2] - 2 * y[:, i + 1] + y[:, i]))
    # 12 quadratic y products
    for i, j in [
        (10, 12), (10, 13), (10, 14), (11, 11), (11, 13), (11, 14),
        (12, 12), (12, 13), (12, 14), (13, 13), (13, 14), (14, 14),
    ]:
        feats.append(y[:, i] * y[:, j])
    # 1 cross-term
    feats.append(y[:, 12] * u[:, 14])
    # y_range
    feats.append(y.max(axis=1) - y.min(axis=1))
    return np.column_stack(feats)


# ── OLS fitting ──────────────────────────────────────────────────────

def fit_ols_koopman(
    x_train: np.ndarray,
    u_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    u_test: np.ndarray,
    y_test: np.ndarray,
    model_type: str = "lasso46",
    horizon: int = 7,
    verbose: bool = True,
) -> Tuple[MultiStepKoopman, np.ndarray, dict]:
    """Fit a Koopman or ARX model by OLS.

    Args:
        model_type: "lasso46" (76-dim lifted) or "arx" (31-dim with intercept).

    Returns:
        (model, w_bounds, metrics)
    """
    if model_type == "lasso46":
        features_tr = compute_lasso46_features(x_train)
        features_te = compute_lasso46_features(x_test)
        psi_tr = np.hstack([x_train, features_tr])  # (N, 76)
        psi_te = np.hstack([x_test, features_te])
        d_lift = 76
        c_idx = 14  # C picks newest y (index 14 = y_past[-1])
        encoder_type = "lasso46"
    elif model_type == "arx":
        psi_tr = np.hstack([x_train, np.ones((len(x_train), 1))])  # (N, 31)
        psi_te = np.hstack([x_test, np.ones((len(x_test), 1))])
        d_lift = 31
        c_idx = 30  # C picks intercept slot (last element)
        encoder_type = "mlp"
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model = MultiStepKoopman(
        n_state=30, n_input=1, d_lift=d_lift, hidden=32,
        horizon=horizon, n_encoder_layers=1,
        encoder_type=encoder_type, n_state_y=15,
    )

    with torch.no_grad():
        # Set encoder to identity/passthrough (bypass learned part)
        if model_type == "lasso46":
            model.encoder.proj.weight.copy_(torch.eye(46))
            model.encoder.proj.bias.zero_()
        elif model_type == "arx":
            # phi outputs d_extra=1; set to constant 1 (intercept)
            for param in model.encoder.phi.parameters():
                param.data.zero_()
            list(model.encoder.phi.parameters())[-1].data[0] = 1.0

        # C vector: picks out the newest y from psi
        model.C.zero_()
        model.C[0, c_idx] = 1.0

        # OLS per horizon step k
        for k in range(1, horizon + 1):
            # Regressor: [psi(z), u_0, ..., u_{k-1}]
            X = np.hstack([psi_tr, u_train[:, :k]])
            W = np.linalg.lstsq(X, y_train[:, k - 1], rcond=None)[0]

            # Pack into A_k, B_k so that C @ A_k @ psi + C @ B_k @ u = W @ X
            A_k = torch.zeros(d_lift, d_lift)
            A_k[c_idx, :] = torch.tensor(W[:d_lift], dtype=torch.float32)
            B_k = torch.zeros(d_lift, k)
            B_k[c_idx, :] = torch.tensor(W[d_lift:], dtype=torch.float32)

            model.A[k - 1].copy_(A_k)
            model.B[k - 1].copy_(B_k)

    model.eval()

    # Compute w_bounds (80th percentile of |residual|) and test MAE
    w_bounds = np.zeros((horizon, 2))
    test_maes = []
    with torch.no_grad():
        for k in range(1, horizon + 1):
            pred = model.forward_k(
                torch.tensor(x_test, dtype=torch.float32),
                torch.tensor(u_test[:, :k], dtype=torch.float32), k,
            ).numpy().flatten()
            resid = y_test[:, k - 1] - pred
            p80 = float(np.percentile(np.abs(resid), 80))
            w_bounds[k - 1] = [-p80, p80]
            test_maes.append(float(np.mean(np.abs(resid))))

    if verbose:
        print(f"  Test MAE per step:")
        for k, mae in enumerate(test_maes):
            print(f"    k={k+1}: {mae:.6f}")

    return model, w_bounds, {"test_mae": test_maes}


# ── Save ──────────────────────────────────────────────────────────────

def save_model(
    model: MultiStepKoopman,
    w_bounds: np.ndarray,
    save_dir: Path,
    model_type: str,
    n_augmentations: int,
    patient: str,
) -> None:
    """Save model weights, config.json, and w_bounds.npy."""
    save_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), save_dir / "koopman_model.pt")

    config = {
        "n_state": 30,
        "n_input": 1,
        "d_lift": model.d_lift,
        "hidden": model.n_hidden,
        "horizon": model.horizon,
        "n_state_y": 15,
        "n_state_u": 15,
        "log_space": True,
        "encoder_type": model.encoder_type,
        "n_encoder_layers": 1,
        "training_method": "least_squares",
        "n_augmentations": n_augmentations,
        "patient": patient,
    }
    with open(save_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    np.save(save_dir / "w_bounds.npy", w_bounds)
    print(f"  Saved model to {save_dir}/")


# ── Main ──────────────────────────────────────────────────────────────

def main() -> None:
    """Train and save one or more Koopman/ARX models from CLI arguments."""
    parser = argparse.ArgumentParser(description="Train Koopman/ARX model by OLS")
    parser.add_argument(
        "--model", choices=["lasso46", "arx", "both"], default="both",
        help="Which model(s) to train (default: both)",
    )
    parser.add_argument("--horizon", type=int, default=7)
    parser.add_argument("--n-augmentations", type=int, default=1)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--data-dir", type=Path, default=None,
        help=(
            "Processed patient root/folder with beta_causal_RMS.csv and "
            "stimulation.csv, or directory of cached .npz x/u/y windows"
        ),
    )
    parser.add_argument(
        "--test-fraction", type=float, default=0.2,
        help="Fraction of custom rows held out for testing",
    )
    parser.add_argument(
        "--input-space",
        choices=["linear", "log"],
        default="linear",
        help="Space used by beta_causal_RMS.csv values before training conversion.",
    )
    parser.add_argument(
        "--patient-role",
        choices=["all", "training", "refinement"],
        default="all",
        help="Role filter when data-dir contains selected_patients.json.",
    )
    parser.add_argument(
        "--synthetic-stim",
        action="store_true",
        help="Overlay synthetic PRBS stimulation on resting-state/autonomous beta.",
    )
    parser.add_argument(
        "--max-samples-per-patient",
        type=int,
        default=None,
        help="Optional centered crop length per patient folder before windowing.",
    )
    parser.add_argument(
        "--train-duration", type=float, default=3600.0,
        help="Synthetic training data duration in seconds (default: 3600)",
    )
    args = parser.parse_args()

    models_to_train = ["lasso46", "arx"] if args.model == "both" else [args.model]

    if args.data_dir is not None:
        print(f"Loading custom data from {args.data_dir} (horizon={args.horizon})...")
    else:
        print(f"Generating synthetic data (horizon={args.horizon}, n_aug={args.n_augmentations})...")
    (x_tr, u_tr, y_tr), (x_te, u_te, y_te), patient = load_training_data(
        horizon=args.horizon,
        n_augmentations=args.n_augmentations,
        modulation_seed=args.seed,
        train_duration=args.train_duration,
        data_dir=args.data_dir,
        test_fraction=args.test_fraction,
        input_space=args.input_space,
        patient_role=args.patient_role,
        synthetic_stim=args.synthetic_stim,
        max_samples_per_patient=args.max_samples_per_patient,
    )
    print(f"  Train: {x_tr.shape[0]} samples, Test: {x_te.shape[0]} samples")

    for model_type in models_to_train:
        t0 = time.time()
        default_name = "koopman_lasso46" if model_type == "lasso46" else "multistep_arx"
        save_dir = Path(args.save_dir) if args.save_dir else Path("models") / default_name

        print(f"\nTraining {model_type} ...")
        model, w_bounds, metrics = fit_ols_koopman(
            x_tr, u_tr, y_tr, x_te, u_te, y_te,
            model_type=model_type, horizon=args.horizon,
        )
        save_model(model, w_bounds, save_dir, model_type, args.n_augmentations, patient)
        print(f"  Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
