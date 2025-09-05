from typing import Dict, Any, Tuple
import numpy as np
from adapters.unified_adapter import to_torch_dataset, pad_collate
from torch.utils.data import DataLoader
import torch

from models.model_picker import pick_model
from train import train_one_model
from evaluation.evaluate import evaluate_model

def objective_for_dataset(X, y, kind: str, label_map=None, device="cpu", hparams: Dict[str, Any]=None) -> Tuple[float, Dict[str, Any]]:
    hparams = hparams or {"lr": 1e-3, "epochs": 3, "hidden": 64}
    ds, collate_fn, label_map2 = to_torch_dataset(X, y, kind=kind, standardize=True)
    if label_map is None:
        label_map = label_map2
    loader = DataLoader(ds, batch_size=64, shuffle=True, collate_fn=collate_fn)

    # infer shapes
    if kind == "tabular":
        input_shape = (X.shape[1],)
    elif kind == "image":
        N, C, H, W = X.shape if X.ndim==4 else (X.shape[0], 1, X.shape[1], X.shape[2])
        input_shape = (C, H, W)
    else:  # timeseries
        if isinstance(X, np.ndarray) and X.ndim==3:
            input_shape = (X.shape[1], X.shape[2])  # (T, C)
        else:
            input_shape = None

    num_classes = len(label_map) if label_map else int(np.unique(y).size)

    model = pick_model(kind=kind, input_shape=input_shape, num_classes=num_classes, hidden=hparams.get("hidden", 64))
    model.to(device)

    model = train_one_model(model, loader, device=device, epochs=hparams.get("epochs", 3), lr=hparams.get("lr", 1e-3))
    metrics = evaluate_model(model, loader, device=device)
    value = metrics.get("macro_f1") or 0.0
    return float(value), metrics
