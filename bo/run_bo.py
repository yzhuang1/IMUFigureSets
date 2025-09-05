"""
Stub for Bayesian Optimization loop.
Replace suggest() and observe() with your favorite BO library (Ax, Optuna, BoTorch, scikit-optimize...).
The contract is:
- suggest() -> dict of hyperparameters
- objective(hparams) -> returns scalar or dict of metrics (you can convert multi-objective to scalar)
- observe(hparams, value) -> record the result
"""

import random

_search_space = {
    "lr": (1e-4, 3e-3),
    "epochs": (3, 10),
    "hidden": (32, 256),
}

def suggest():
    return {
        "lr": 10 ** random.uniform(-4, -2.5),
        "epochs": random.randint(3, 8),
        "hidden": random.choice([32, 64, 128, 256]),
    }

def observe(hparams, value):
    # no-op stub
    pass
