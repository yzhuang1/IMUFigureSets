from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset
try:
    import pandas as pd
except Exception:
    pd = None  # optional

def _to_float_tensor(x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.float()
    return torch.from_numpy(np.asarray(x)).float()

def _encode_labels(y: Optional[Union[List[Any], np.ndarray, torch.Tensor]]) -> Tuple[Optional[torch.Tensor], Dict[Any, int]]:
    if y is None:
        return None, {}
    if isinstance(y, torch.Tensor):
        if y.dtype == torch.long or y.dtype == torch.int64:
            return y.long(), {}
        y = y.cpu().numpy()
    y = np.asarray(y)
    classes, inv = np.unique(y, return_inverse=True)
    mapping = {cls: int(i) for i, cls in enumerate(classes)}
    y_enc = torch.from_numpy(inv.astype(np.int64))
    return y_enc, mapping

class RaggedTimeSeriesDataset(Dataset):
    def __init__(
        self,
        sequences: List[Union[np.ndarray, torch.Tensor]],
        labels: Optional[Union[List[Any], np.ndarray, torch.Tensor]] = None,
        layout: str = "TC",
        standardize: bool = False,
    ):
        self.X = []
        for s in sequences:
            t = _to_float_tensor(s)
            if layout.upper() == "CT":
                t = t.permute(1, 0)
            if standardize:
                mean = t.mean(dim=0, keepdim=True)
                std = t.std(dim=0, keepdim=True).clamp_min(1e-6)
                t = (t - mean) / std
            self.X.append(t)
        self.y, self.label_map = _encode_labels(labels)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is None:
            return self.X[idx]
        return self.X[idx], self.y[idx]

def pad_collate(
    batch: List[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]],
    pad_value: float = 0.0,
    time_first: bool = False
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    has_label = isinstance(batch[0], (tuple, list))
    if has_label:
        xs, ys = zip(*batch)
        ys = torch.stack(ys, dim=0).long()
    else:
        xs = batch
        ys = None

    lengths = torch.tensor([x.shape[0] for x in xs], dtype=torch.long)
    max_len = int(lengths.max().item())
    feat_dim = xs[0].shape[1]
    X_pad = xs[0].new_full((len(xs), max_len, feat_dim), pad_value)
    for i, x in enumerate(xs):
        T = x.shape[0]
        X_pad[i, :T, :] = x

    if time_first:
        X_pad = X_pad.permute(0, 2, 1)
    return X_pad, ys, lengths

class UnifiedDataset(Dataset):
    def __init__(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Optional[Union[List[Any], np.ndarray, torch.Tensor]] = None,
        layout: Optional[str] = None,
        standardize: bool = False,
        image_mode: bool = False
    ):
        Xt = _to_float_tensor(X)
        if layout is not None:
            if layout.upper() == "NCT":
                Xt = Xt.permute(0, 2, 1)
        if standardize and not image_mode:
            mean = Xt.mean(dim=0, keepdim=True)
            std = Xt.std(dim=0, keepdim=True).clamp_min(1e-6)
            Xt = (Xt - mean) / std

        self.X = Xt
        self.y, self.label_map = _encode_labels(y)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if self.y is None:
            return self.X[idx]
        return self.X[idx], self.y[idx]

_ADAPTERS: Dict[str, Callable[..., Tuple[Dataset, Optional[Callable], Dict[Any, int]]]] = {}

def register_adapter(name: str, fn: Callable[..., Tuple[Dataset, Optional[Callable], Dict[Any, int]]]) -> None:
    _ADAPTERS[name.lower()] = fn

def _adapt_timeseries(
    data: Union[np.ndarray, torch.Tensor, List[Union[np.ndarray, torch.Tensor]]],
    labels: Optional[Union[List[Any], np.ndarray, torch.Tensor]] = None,
    layout: str = "NTC",
    standardize: bool = False,
    **kwargs
) -> Tuple[Dataset, Optional[Callable], Dict[Any, int]]:
    if isinstance(data, list):
        ds = RaggedTimeSeriesDataset(
            sequences=data, labels=labels,
            layout="TC" if layout.upper() in ("NTC", "TC") else "CT",
            standardize=standardize
        )
        return ds, pad_collate, ds.label_map
    else:
        X = _to_float_tensor(data)
        if layout.upper() == "NCT":
            X = X.permute(0, 2, 1)
        ds = UnifiedDataset(X, labels, layout="NTC", standardize=standardize, image_mode=False)
        return ds, None, ds.label_map

def _adapt_tabular(
    data: Union[np.ndarray, torch.Tensor, "pd.DataFrame"],
    labels: Optional[Union[List[Any], np.ndarray, torch.Tensor]] = None,
    standardize: bool = False,
    **kwargs
) -> Tuple[Dataset, Optional[Callable], Dict[Any, int]]:
    if pd is not None and isinstance(data, pd.DataFrame):
        X = data.values
    else:
        X = data
    X = _to_float_tensor(X)
    if standardize:
        mean = X.mean(dim=0, keepdim=True)
        std = X.std(dim=0, keepdim=True).clamp_min(1e-6)
        X = (X - mean) / std
    y, label_map = _encode_labels(labels)
    if y is None:
        ds = TensorDataset(X)
    else:
        ds = TensorDataset(X, y)
    return ds, None, label_map

def _adapt_image(
    data: Union[np.ndarray, torch.Tensor],
    labels: Optional[Union[List[Any], np.ndarray, torch.Tensor]] = None,
    channel_first: bool = True,
    **kwargs
) -> Tuple[Dataset, Optional[Callable], Dict[Any, int]]:
    X = _to_float_tensor(data)
    if X.ndim == 3:
        X = X.unsqueeze(1)
    elif X.ndim == 4 and not channel_first:
        X = X.permute(0, 3, 1, 2)
    ds = UnifiedDataset(X, labels, image_mode=True)
    return ds, None, ds.label_map

register_adapter("timeseries", _adapt_timeseries)
register_adapter("tabular", _adapt_tabular)
register_adapter("image", _adapt_image)

def to_torch_dataset(
    data: Any,
    labels: Optional[Union[List[Any], np.ndarray, torch.Tensor]] = None,
    kind: Optional[str] = None,
    **kwargs
) -> Tuple[Dataset, Optional[Callable], Dict[Any, int]]:
    if kind is not None:
        fn = _ADAPTERS.get(kind.lower())
        if fn is None:
            raise ValueError(f"Unknown kind='{kind}'. Registered: {list(_ADAPTERS.keys())}")
        return fn(data, labels=labels, **kwargs)

    if pd is not None and isinstance(data, pd.DataFrame):
        return _ADAPTERS["tabular"](data, labels=labels, **kwargs)

    if isinstance(data, list):
        return _ADAPTERS["timeseries"](data, labels=labels, **kwargs)

    if isinstance(data, (np.ndarray, torch.Tensor)):
        ndim = data.ndim
        if ndim == 2:
            return _ADAPTERS["tabular"](data, labels=labels, **kwargs)
        elif ndim == 3:
            return _ADAPTERS["timeseries"](data, labels=labels, layout=kwargs.pop("layout", "NTC"), **kwargs)
        elif ndim == 4:
            return _ADAPTERS["image"](data, labels=labels, **kwargs)

    raise ValueError("Cannot infer data kind. Please pass kind= 'timeseries' | 'tabular' | 'image'.")
