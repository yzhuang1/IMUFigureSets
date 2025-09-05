from typing import Iterable
import torch
from torch import nn
from torch.optim import Adam
from tqdm import tqdm

def train_one_model(model: nn.Module, loader: Iterable, device: str="cpu", epochs: int=5, lr: float=1e-3) -> nn.Module:
    model.train()
    opt = Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    
    for _ in range(epochs):
        for batch in tqdm(loader, desc="training", leave=False):
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
            y = y.to(device)

            logits = model(X) if lengths is None else model(X, lengths=lengths)
            loss = loss_fn(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model
