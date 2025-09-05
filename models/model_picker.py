from typing import Optional, Tuple
from torch import nn
from .tabular_mlp import TabMLP
from .image_cnn import SmallCNN
from .tiny_cnn_1d import TinyCNN1D

def pick_model(kind: str, input_shape: Optional[Tuple]=None, num_classes: int=2, hidden: int=64) -> nn.Module:
    kind = kind.lower()
    if kind == "tabular":
        in_dim = input_shape[0]
        return TabMLP(in_dim, num_classes=num_classes, hidden=hidden)
    elif kind == "image":
        C, H, W = input_shape
        return SmallCNN(in_channels=C, num_classes=num_classes, hidden=hidden)
    elif kind == "timeseries":
        # Expect [B,T,C] as input for TinyCNN1D
        feat_dim = input_shape[-1] if input_shape else 6
        return TinyCNN1D(in_channels=feat_dim, num_classes=num_classes, hidden=hidden)
    else:
        raise ValueError(f"unknown kind: {kind}")
