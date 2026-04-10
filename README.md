# koopman-mpc-dbs

Koopman operator MPC for closed-loop deep brain stimulation (DBS).

The Koopman model lifts the state to a space where dynamics are **affine in future control** — enabling a single QP per timestep (no iterative SCP). Trained by OLS in seconds; no GPU required.

> **Disclaimer**: This is a research prototype and is **not a medical device**. See [DISCLAIMER.md](DISCLAIMER.md).

---

## Quick start

```bash
pip install -e ".[dev]"
python examples/quick_demo.py
```

## Training

```bash
# Train lasso46 model (recommended) and multi-step ARX
python -m koopman_mpc.training.train_koopman_ols --horizon 7

# Lasso46 only
python -m koopman_mpc.training.train_koopman_ols --model lasso46
```

Models are saved to `models/koopman_lasso46/` and `models/multistep_arx/`.

## Running MPC

```bash
python scripts/run_koopman_mpc.py --model models/koopman_lasso46 --duration 60
```

## Using with the bench repo

```python
from koopman_mpc.models.koopman_model import load_koopman_model
from koopman_mpc.simulation.simulate import run_koopman_mpc

predictor = load_koopman_model("models/koopman_lasso46")
result = run_koopman_mpc(predictor, duration=60.0)
print(result.metrics)
```

Or plug directly into the bench `SimulationRunner` via `KoopmanControllerAdapter`:

```python
from koopman_mpc.simulation.simulate import KoopmanControllerAdapter
from dbs_bench.simulation.simulate import SimulationRunner
from dbs_bench.synthetic.data_generator import generate_demo_patient

ctrl = KoopmanControllerAdapter(predictor)
patient = generate_demo_patient()
runner = SimulationRunner(patient, dt=0.02, beta_0=2.3)
result = runner.run(ctrl, duration=60.0, controller_type="custom")
```

## How it works

The Koopman model makes multi-step predictions:
```
y_{t+k} = C @ A_k @ psi(z_t) + C @ B_k @ u[0:k]
         = e_k(z_t)  +  F_k @ u[0:k]
```

Because `e_k` depends only on the current state `z_t` (not future `u`), the
MPC cost is a standard QP — no SCP iterations, no linearisation error.

The `lasso46` encoder uses 46 hand-crafted analytical features selected by
LassoCV: signed differences, absolute differences, second differences,
quadratic products, and a cross-term. All fitted by OLS (< 1 second).

## Tests

```bash
pytest tests/ -v
```

## Citation

```bibtex
@software{ampomah2025koopmandbs,
  author = {Ampomah, Joshua},
  title  = {koopman-mpc-dbs},
  year   = {2025},
}
```

## License

MIT — see [LICENSE](LICENSE).
