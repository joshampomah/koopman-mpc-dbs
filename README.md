# koopman-mpc-dbs

Koopman operator MPC for closed-loop deep brain stimulation (DBS).

This repository contains the Koopman-model controller path from the project. It
lifts the recent beta/stimulation history into a feature space where future
outputs are affine in future control inputs, so each MPC step is a single dense
QP rather than an SCP loop.

This code is a research prototype. It is not a medical device and must not be
used for clinical decision-making or patient treatment. See
[DISCLAIMER.md](DISCLAIMER.md).

## Repository Set

The project is split by responsibility:

| Repository | Purpose |
|---|---|
| [closed-loop-dbs-bench](https://github.com/joshampomah/closed-loop-dbs-bench) | Shared benchmark, synthetic DBS plant, metrics, plotting utilities, bang-bang/PI/linear baselines |
| [dcnn-tube-mpc-dbs](https://github.com/joshampomah/dcnn-tube-mpc-dbs) | DC neural network tube MPC method: predictor, SCP controller, uncertainty bounds, synthetic training/demo code |
| [koopman-mpc-dbs](https://github.com/joshampomah/koopman-mpc-dbs) | Koopman MPC method: lifted-linear predictor, dense QP builder, OLS training/demo code |
| [embedded-stable-neuron-mpc](https://github.com/joshampomah/embedded-stable-neuron-mpc) | C++/STM32 implementation of the stable-neuron and Koopman QP solvers, plus the final report PDF |

Use `closed-loop-dbs-bench` for a common comparison harness. Use this repo for
the Koopman model, training path, and controller implementation.

## What Is In This Repo

- `src/koopman_mpc/models/`: Koopman model definitions, feature encoders, model
  save/load helpers, and QP matrix extraction.
- `src/koopman_mpc/training/`: OLS/Lasso-style synthetic training path.
- `src/koopman_mpc/controllers/`: Koopman MPC controller, dense QP builder, and
  a shared result dataclass.
- `src/koopman_mpc/simulation/`: integration helpers for running Koopman MPC
  with `closed-loop-dbs-bench`.
- `src/koopman_mpc/synthetic/`: public-safe synthetic data generation and schema
  definitions.
- `src/koopman_mpc/evaluation/` and `src/koopman_mpc/analysis/`: shared metric
  and plotting helpers.
- `scripts/run_koopman_mpc.py`: command-line synthetic run for a saved Koopman
  model.
- `examples/quick_demo.py`: end-to-end synthetic demo that trains a small model
  and runs closed-loop simulation.
- `tests/`: lightweight public-safe tests for model, training, and QP utilities.

## What Is Not In This Repo

- No patient recordings.
- No patient-trained model checkpoints.
- No private experiment archive or report-writing material.
- No STM32 firmware. Embedded deployment lives in `embedded-stable-neuron-mpc`.

The included demos train or use synthetic/public-safe data. Any real-data
integration should be done through a private loader that conforms to the public
schema.

## Installation

Requires Python 3.10-3.12.

For the full demo path, install the benchmark repo alongside this repo because
the simulation helper uses `dbs_bench`:

```bash
pip install -e ../closed-loop-dbs-bench
pip install -e ".[dev]"
```

## Quick Start

Run the synthetic end-to-end demo:

```bash
python examples/quick_demo.py
```

Train a model explicitly:

```bash
python -m koopman_mpc.training.train_koopman_ols --horizon 7 --model lasso46
```

Run a saved model:

```bash
python scripts/run_koopman_mpc.py --model models/koopman_lasso46 --duration 60
```

Models are saved under `models/` by the training script.

## Using Your Own Data

This repo does not include patient recordings, but the training script can read
the same processed patient folders used by the DCNN repo:

```text
private_data/processed/aperiodic/
├── patient_001/
│   ├── beta_causal_RMS.csv
│   ├── stimulation.csv
│   └── metadata.json
├── patient_002/
│   └── ...
└── selected_patients.json
```

In the 4YP workflow, those folders were produced from raw `.mat` files from the
Cambium/MRC BNDU dataset [STN local field potential recordings from awake
patients with Parkinson's, ON and OFF meds, and during 130 Hz
DBS](https://data.mrc.ox.ac.uk/stn-lfp-on-off-and-dbs). Registered/logged-in
users can download or request access to the raw data from that page. The
medication-state `.mat` files contain a `SmrData` struct with `Fs`, `WvData`,
and `WvTits`; the MATLAB processing selected the STN channel, extracted a
causal 13-30 Hz beta RMS envelope, resampled to 50 Hz, and wrote
`beta_causal_RMS.csv`. For resting-state recordings, `stimulation.csv` is the
same length and contains zeros.

Train from the processed root with:

```bash
python -m koopman_mpc.training.train_koopman_ols \
  --data-dir ../private_data/processed/aperiodic \
  --input-space linear \
  --patient-role training \
  --synthetic-stim \
  --horizon 7 \
  --model lasso46 \
  --save-dir models/koopman_custom
```

`--synthetic-stim` overlays PRBS stimulation on autonomous/resting-state beta.
Omit it if `stimulation.csv` already contains applied stimulation.
`--input-space linear` is correct for the processed `beta_causal_RMS.csv`; use
`--input-space log` only if your CSV has already been log-transformed.

You can also point `--data-dir` at a single folder containing
`beta_causal_RMS.csv` and `stimulation.csv`, rather than a root containing
many `patient_*` folders.

Cached `.npz` files with `x`, `u`, and `y` arrays are still supported as an
optional private preprocessing format. The loader stacks all rows and holds out
`--test-fraction` for test metrics. For proper patient/session-level
validation, do the split in a private script and call `fit_ols_koopman(...)`
directly.

Runtime controller histories are newest-first; training windows are
oldest-to-newest. The benchmark repo has a longer
[DATA.md](https://github.com/joshampomah/closed-loop-dbs-bench/blob/master/DATA.md)
covering the raw `.mat` processing route and replay format.

## Main Programming Interface

Load a model and construct the controller:

```python
import numpy as np

from koopman_mpc.controllers.koopman_controller import (
    KoopmanController,
    KoopmanMPCConfig,
)
from koopman_mpc.models.koopman_model import load_koopman_model

predictor = load_koopman_model("models/koopman_lasso46")
w_bounds = np.load("models/koopman_lasso46/w_bounds.npy")

cfg = KoopmanMPCConfig(prediction_horizon=predictor.horizon, beta_0=2.3)
ctrl = KoopmanController(predictor, cfg, W_bounds=w_bounds)
u, info = ctrl.compute_control(y_history, u_history, u_prev)
```

`y_history` and `u_history` are newest-first arrays. `compute_control` returns
the first control action and an `SCPResult`-compatible diagnostics object.

## Using With The Benchmark Repo

Use the adapter in `koopman_mpc.simulation.simulate`:

```python
from dbs_bench.simulation.simulate import SimulationRunner
from dbs_bench.synthetic.data_generator import generate_demo_patient
from koopman_mpc.models.koopman_model import load_koopman_model
from koopman_mpc.simulation.simulate import KoopmanControllerAdapter

predictor = load_koopman_model("models/koopman_lasso46")
ctrl = KoopmanControllerAdapter(predictor)

patient = generate_demo_patient(n_state_y=15)
runner = SimulationRunner(patient, dt=0.02, beta_0=2.3)
result = runner.run(ctrl, duration=60.0, controller_type="koopman-mpc")
print(result.metrics)
```

The convenience function `run_koopman_mpc(...)` wraps this setup for synthetic
demos.

## Method Summary

The Koopman model predicts:

```text
y_{t+k} = C A_k psi(z_t) + C B_k u_{0:k}
        = e_k(z_t) + F_k u_{0:k}
```

For a fixed current state `z_t`, `e_k` and `F_k` are known, so the MPC problem
is a standard QP. The `lasso46` encoder uses 46 hand-crafted analytical
features: signed differences, absolute differences, second differences,
quadratic products, and a cross-term.

## Tests

```bash
pytest tests/ -v
```

The GitHub Actions workflow runs the tests on Python 3.10, 3.11, and 3.12.

## Citation

```bibtex
@software{ampomah2025koopmandbs,
  author = {Ampomah, Joshua},
  title  = {koopman-mpc-dbs},
  year   = {2025},
}
```

See [CITATION.cff](CITATION.cff) for full citation metadata.

## License

MIT. See [LICENSE](LICENSE).
