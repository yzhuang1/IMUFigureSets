from adapters.unified_adapter import to_torch_dataset
from torch.utils.data import DataLoader
import numpy as np
import torch

from train import train_one_model
from evaluation.evaluate import evaluate_model
from models.model_picker import pick_model

def demo_run():
    # Example: Tabular dataset
    X = np.random.randn(256, 20).astype("float32")
    y = np.random.choice(["A","B","C"], size=256)
    ds, collate_fn, label_map = to_torch_dataset(X, y, kind="tabular", standardize=True)
    loader = DataLoader(ds, batch_size=64, shuffle=True, collate_fn=collate_fn)
    
    model = pick_model(kind="tabular", input_shape=(20,), num_classes=len(label_map) if label_map else 3)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    model = train_one_model(model, loader, device=device, epochs=3)
    metrics = evaluate_model(model, loader, device=device)
    print("Demo metrics:", metrics)

if __name__ == "__main__":
    demo_run()
