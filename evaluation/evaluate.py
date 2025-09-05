from typing import Iterable, Dict, Any
import torch
from torch import nn
from sklearn.metrics import accuracy_score, f1_score

@torch.no_grad()
def evaluate_model(model: nn.Module, loader: Iterable, device: str="cpu") -> Dict[str, Any]:
    model.eval()
    ys, ps = [], []
    
    for batch in loader:
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            X, y, lengths = batch
        elif isinstance(batch, (list, tuple)) and len(batch) == 2:
            X, y = batch
            lengths = None
        else:
            X = batch
            y = None
            lengths = None
            
        X = X.to(device)
        if y is None:
            continue
        y = y.cpu().numpy()

        logits = model(X) if lengths is None else model(X, lengths=lengths)
        pred = logits.argmax(dim=-1).cpu().numpy()

        ys.append(y)
        ps.append(pred)
        
    if not ys:
        return {"acc": None, "macro_f1": None}
        
    import numpy as np
    ys = np.concatenate(ys, axis=0)
    ps = np.concatenate(ps, axis=0)
    return {
        "acc": float(accuracy_score(ys, ps)),
        "macro_f1": float(f1_score(ys, ps, average="macro"))
    }
