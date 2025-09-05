"""
Universal Data Converter
Automatically converts various data formats to PyTorch tensor format
Supports automatic data type detection and feature extraction
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset
import json
import logging

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from PIL import Image
except ImportError:
    Image = None

logger = logging.getLogger(__name__)

class DataProfile:
    """Data feature description class"""
    def __init__(self):
        self.data_type: str = "unknown"
        self.shape: Tuple[int, ...] = ()
        self.dtype: str = "unknown"
        self.feature_count: int = 0
        self.sample_count: int = 0
        self.is_sequence: bool = False
        self.is_image: bool = False
        self.is_tabular: bool = False
        self.has_labels: bool = False
        self.label_count: int = 0
        self.sequence_lengths: Optional[List[int]] = None
        self.channels: Optional[int] = None
        self.height: Optional[int] = None
        self.width: Optional[int] = None
        self.metadata: Dict[str, Any] = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for AI model selection"""
        return {
            "data_type": self.data_type,
            "shape": self.shape,
            "dtype": str(self.dtype),
            "feature_count": self.feature_count,
            "sample_count": self.sample_count,
            "is_sequence": self.is_sequence,
            "is_image": self.is_image,
            "is_tabular": self.is_tabular,
            "has_labels": self.has_labels,
            "label_count": self.label_count,
            "sequence_lengths": self.sequence_lengths,
            "channels": self.channels,
            "height": self.height,
            "width": self.width,
            "metadata": self.metadata
        }

    def __str__(self) -> str:
        return f"DataProfile(type={self.data_type}, shape={self.shape}, samples={self.sample_count}, features={self.feature_count})"

def _to_float_tensor(x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    """Convert to float32 tensor"""
    if isinstance(x, torch.Tensor):
        return x.float()
    return torch.from_numpy(np.asarray(x, dtype=np.float32)).float()

def _encode_labels(y: Optional[Union[List[Any], np.ndarray, torch.Tensor]]) -> Tuple[Optional[torch.Tensor], Dict[Any, int]]:
    """Encode labels as integers"""
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

def analyze_data_profile(data: Any, labels: Optional[Any] = None) -> DataProfile:
    """Analyze data characteristics and generate data profile"""
    profile = DataProfile()
    
    # Process labels
    if labels is not None:
        profile.has_labels = True
        if isinstance(labels, (list, np.ndarray, torch.Tensor)):
            profile.label_count = len(np.unique(labels))
        else:
            profile.label_count = 1
    
    # Process data
    if isinstance(data, list):
        # List data - might be sequence data
        profile.data_type = "sequence_list"
        profile.sample_count = len(data)
        profile.is_sequence = True
        
        if len(data) > 0:
            first_item = data[0]
            if isinstance(first_item, (np.ndarray, torch.Tensor)):
                profile.feature_count = first_item.shape[-1] if len(first_item.shape) > 1 else 1
                profile.sequence_lengths = [len(item) for item in data]
                profile.shape = (len(data), max(profile.sequence_lengths), profile.feature_count)
            else:
                profile.feature_count = 1
                profile.shape = (len(data), 1)
    
    elif isinstance(data, np.ndarray):
        profile.data_type = "numpy_array"
        profile.shape = data.shape
        profile.sample_count = data.shape[0]
        profile.dtype = str(data.dtype)
        
        if data.ndim == 2:
            # 2D array - tabular data
            profile.is_tabular = True
            profile.feature_count = data.shape[1]
        elif data.ndim == 3:
            # 3D array - might be time series or image
            if data.shape[1] == 3 or data.shape[2] == 3:
                # Might be image data (H, W, C) or (C, H, W)
                profile.is_image = True
                profile.channels = 3
                profile.height = data.shape[0] if data.shape[2] == 3 else data.shape[1]
                profile.width = data.shape[1] if data.shape[2] == 3 else data.shape[2]
            else:
                # Time series data (N, T, C)
                profile.is_sequence = True
                profile.feature_count = data.shape[2]
        elif data.ndim == 4:
            # 4D array - image batch (N, C, H, W) or (N, H, W, C)
            profile.is_image = True
            profile.sample_count = data.shape[0]
            profile.channels = data.shape[1] if data.shape[1] <= 4 else data.shape[3]
            profile.height = data.shape[2] if data.shape[1] <= 4 else data.shape[1]
            profile.width = data.shape[3] if data.shape[1] <= 4 else data.shape[2]
    
    elif isinstance(data, torch.Tensor):
        profile.data_type = "torch_tensor"
        profile.shape = data.shape
        profile.sample_count = data.shape[0]
        profile.dtype = str(data.dtype)
        
        if data.ndim == 2:
            profile.is_tabular = True
            profile.feature_count = data.shape[1]
        elif data.ndim == 3:
            if data.shape[1] == 3 or data.shape[2] == 3:
                profile.is_image = True
                profile.channels = 3
                profile.height = data.shape[0] if data.shape[2] == 3 else data.shape[1]
                profile.width = data.shape[1] if data.shape[2] == 3 else data.shape[2]
            else:
                profile.is_sequence = True
                profile.feature_count = data.shape[2]
        elif data.ndim == 4:
            profile.is_image = True
            profile.sample_count = data.shape[0]
            profile.channels = data.shape[1] if data.shape[1] <= 4 else data.shape[3]
            profile.height = data.shape[2] if data.shape[1] <= 4 else data.shape[1]
            profile.width = data.shape[3] if data.shape[1] <= 4 else data.shape[2]
    
    elif pd is not None and isinstance(data, pd.DataFrame):
        profile.data_type = "pandas_dataframe"
        profile.is_tabular = True
        profile.sample_count = len(data)
        profile.feature_count = len(data.columns)
        profile.shape = (len(data), len(data.columns))
        profile.metadata["columns"] = list(data.columns)
        profile.metadata["dtypes"] = {col: str(dtype) for col, dtype in data.dtypes.items()}
    
    else:
        # Try to convert to numpy array
        try:
            data_array = np.asarray(data)
            return analyze_data_profile(data_array, labels)
        except Exception as e:
            logger.warning(f"Cannot analyze data type: {type(data)}, error: {e}")
            profile.data_type = "unknown"
    
    return profile

class UniversalDataset(Dataset):
    """Universal dataset class supporting various data formats"""
    
    def __init__(
        self,
        data: Any,
        labels: Optional[Any] = None,
        profile: Optional[DataProfile] = None,
        standardize: bool = False,
        **kwargs
    ):
        self.profile = profile or analyze_data_profile(data, labels)
        self.standardize = standardize
        
        # Convert data
        self.X = self._convert_data(data)
        self.y, self.label_map = _encode_labels(labels)
        
        # Standardize
        if standardize and not self.profile.is_image:
            self._standardize_data()
    
    def _convert_data(self, data: Any) -> torch.Tensor:
        """Convert data to tensor format"""
        if isinstance(data, torch.Tensor):
            return _to_float_tensor(data)
        elif isinstance(data, np.ndarray):
            return _to_float_tensor(data)
        elif isinstance(data, list):
            # Handle sequence data
            if self.profile.is_sequence:
                return self._convert_sequence_data(data)
            else:
                return _to_float_tensor(np.array(data))
        elif pd is not None and isinstance(data, pd.DataFrame):
            return _to_float_tensor(data.values)
        else:
            # Try to convert to numpy then tensor
            return _to_float_tensor(np.asarray(data))
    
    def _convert_sequence_data(self, sequences: List) -> torch.Tensor:
        """Convert sequence data"""
        # For sequence data, we return a special marker
        # Actual data will be handled in collate_fn
        self._sequences = [_to_float_tensor(seq) for seq in sequences]
        return torch.zeros(len(sequences))  # Placeholder
    
    def _standardize_data(self):
        """Standardize data"""
        if hasattr(self, '_sequences'):
            # Sequence data standardization
            for i, seq in enumerate(self._sequences):
                mean = seq.mean(dim=0, keepdim=True)
                std = seq.std(dim=0, keepdim=True).clamp_min(1e-6)
                self._sequences[i] = (seq - mean) / std
        else:
            # Regular data standardization
            mean = self.X.mean(dim=0, keepdim=True)
            std = self.X.std(dim=0, keepdim=True).clamp_min(1e-6)
            self.X = (self.X - mean) / std
    
    def __len__(self):
        return self.sample_count
    
    def __getitem__(self, idx):
        if hasattr(self, '_sequences'):
            # Sequence data
            if self.y is None:
                return self._sequences[idx]
            return self._sequences[idx], self.y[idx]
        else:
            # Regular data
            if self.y is None:
                return self.X[idx]
            return self.X[idx], self.y[idx]
    
    @property
    def sample_count(self):
        return len(self._sequences) if hasattr(self, '_sequences') else self.X.shape[0]

def universal_collate_fn(batch: List) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Universal collate function handling various data formats"""
    if not batch:
        return torch.empty(0), None, None
    
    # Check if there are labels
    has_label = isinstance(batch[0], (tuple, list)) and len(batch[0]) == 2
    
    if has_label:
        xs, ys = zip(*batch)
        ys = torch.stack(ys, dim=0).long()
    else:
        xs = batch
        ys = None
    
    # Check if it's sequence data
    if isinstance(xs[0], torch.Tensor) and xs[0].dim() > 1:
        # Regular tensor data
        X = torch.stack(xs, dim=0)
        return X, ys, None
    else:
        # Sequence data, needs padding
        lengths = torch.tensor([x.shape[0] for x in xs], dtype=torch.long)
        max_len = int(lengths.max().item())
        feat_dim = xs[0].shape[1] if len(xs[0].shape) > 1 else 1
        
        X_pad = xs[0].new_full((len(xs), max_len, feat_dim), 0.0)
        for i, x in enumerate(xs):
            T = x.shape[0]
            X_pad[i, :T, :] = x
        
        return X_pad, ys, lengths

class UniversalConverter:
    """Universal data converter"""
    
    def __init__(self):
        self.converters = {}
        self._register_default_converters()
    
    def _register_default_converters(self):
        """Register default converters"""
        self.converters["numpy_array"] = self._convert_numpy
        self.converters["torch_tensor"] = self._convert_torch
        self.converters["pandas_dataframe"] = self._convert_pandas
        self.converters["sequence_list"] = self._convert_sequence_list
    
    def register_converter(self, data_type: str, converter_func: Callable):
        """Register custom converter"""
        self.converters[data_type] = converter_func
    
    def _convert_numpy(self, data: np.ndarray, labels: Optional[Any] = None, **kwargs) -> Tuple[Dataset, Optional[Callable], DataProfile]:
        """Convert numpy array"""
        profile = analyze_data_profile(data, labels)
        dataset = UniversalDataset(data, labels, profile, **kwargs)
        collate_fn = universal_collate_fn if profile.is_sequence else None
        return dataset, collate_fn, profile
    
    def _convert_torch(self, data: torch.Tensor, labels: Optional[Any] = None, **kwargs) -> Tuple[Dataset, Optional[Callable], DataProfile]:
        """Convert torch tensor"""
        profile = analyze_data_profile(data, labels)
        dataset = UniversalDataset(data, labels, profile, **kwargs)
        collate_fn = universal_collate_fn if profile.is_sequence else None
        return dataset, collate_fn, profile
    
    def _convert_pandas(self, data: pd.DataFrame, labels: Optional[Any] = None, **kwargs) -> Tuple[Dataset, Optional[Callable], DataProfile]:
        """Convert pandas DataFrame"""
        profile = analyze_data_profile(data, labels)
        dataset = UniversalDataset(data, labels, profile, **kwargs)
        return dataset, None, profile
    
    def _convert_sequence_list(self, data: List, labels: Optional[Any] = None, **kwargs) -> Tuple[Dataset, Optional[Callable], DataProfile]:
        """Convert sequence list"""
        profile = analyze_data_profile(data, labels)
        dataset = UniversalDataset(data, labels, profile, **kwargs)
        return dataset, universal_collate_fn, profile
    
    def convert(
        self, 
        data: Any, 
        labels: Optional[Any] = None, 
        data_type: Optional[str] = None,
        **kwargs
    ) -> Tuple[Dataset, Optional[Callable], DataProfile]:
        """
        Universal data conversion method
        
        Args:
            data: Input data
            labels: Label data
            data_type: Specify data type, if None then auto-detect
            **kwargs: Other parameters
        
        Returns:
            dataset: PyTorch dataset
            collate_fn: Collate function for data loading
            profile: Data feature profile
        """
        if data_type is None:
            profile = analyze_data_profile(data, labels)
            data_type = profile.data_type
        else:
            profile = analyze_data_profile(data, labels)
        
        converter = self.converters.get(data_type)
        if converter is None:
            # Try universal conversion
            return self._convert_numpy(data, labels, **kwargs)
        
        return converter(data, labels, **kwargs)

# Global converter instance
universal_converter = UniversalConverter()

def convert_to_torch_dataset(
    data: Any,
    labels: Optional[Any] = None,
    data_type: Optional[str] = None,
    **kwargs
) -> Tuple[Dataset, Optional[Callable], DataProfile]:
    """
    Convenience function: Convert any data to PyTorch dataset
    
    Args:
        data: Input data
        labels: Label data
        data_type: Specify data type, if None then auto-detect
        **kwargs: Other parameters
    
    Returns:
        dataset: PyTorch dataset
        collate_fn: Collate function for data loading
        profile: Data feature profile
    """
    return universal_converter.convert(data, labels, data_type, **kwargs)
