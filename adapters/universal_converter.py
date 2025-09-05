"""
Universal Data Converter
将各种数据格式自动转换为PyTorch tensor格式
支持自动检测数据类型和特征提取
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
    """数据特征描述类"""
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
        """转换为字典格式，用于AI模型选择"""
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
    """转换为float32 tensor"""
    if isinstance(x, torch.Tensor):
        return x.float()
    return torch.from_numpy(np.asarray(x, dtype=np.float32)).float()

def _encode_labels(y: Optional[Union[List[Any], np.ndarray, torch.Tensor]]) -> Tuple[Optional[torch.Tensor], Dict[Any, int]]:
    """编码标签为整数"""
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
    """分析数据特征，生成数据档案"""
    profile = DataProfile()
    
    # 处理标签
    if labels is not None:
        profile.has_labels = True
        if isinstance(labels, (list, np.ndarray, torch.Tensor)):
            profile.label_count = len(np.unique(labels))
        else:
            profile.label_count = 1
    
    # 处理数据
    if isinstance(data, list):
        # 列表数据 - 可能是序列数据
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
            # 2D数组 - 表格数据
            profile.is_tabular = True
            profile.feature_count = data.shape[1]
        elif data.ndim == 3:
            # 3D数组 - 可能是时间序列或图像
            if data.shape[1] == 3 or data.shape[2] == 3:
                # 可能是图像数据 (H, W, C) 或 (C, H, W)
                profile.is_image = True
                profile.channels = 3
                profile.height = data.shape[0] if data.shape[2] == 3 else data.shape[1]
                profile.width = data.shape[1] if data.shape[2] == 3 else data.shape[2]
            else:
                # 时间序列数据 (N, T, C)
                profile.is_sequence = True
                profile.feature_count = data.shape[2]
        elif data.ndim == 4:
            # 4D数组 - 图像批次 (N, C, H, W) 或 (N, H, W, C)
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
        # 尝试转换为numpy数组
        try:
            data_array = np.asarray(data)
            return analyze_data_profile(data_array, labels)
        except Exception as e:
            logger.warning(f"无法分析数据类型: {type(data)}, 错误: {e}")
            profile.data_type = "unknown"
    
    return profile

class UniversalDataset(Dataset):
    """通用数据集类，支持各种数据格式"""
    
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
        
        # 转换数据
        self.X = self._convert_data(data)
        self.y, self.label_map = _encode_labels(labels)
        
        # 标准化
        if standardize and not self.profile.is_image:
            self._standardize_data()
    
    def _convert_data(self, data: Any) -> torch.Tensor:
        """将数据转换为tensor格式"""
        if isinstance(data, torch.Tensor):
            return _to_float_tensor(data)
        elif isinstance(data, np.ndarray):
            return _to_float_tensor(data)
        elif isinstance(data, list):
            # 处理序列数据
            if self.profile.is_sequence:
                return self._convert_sequence_data(data)
            else:
                return _to_float_tensor(np.array(data))
        elif pd is not None and isinstance(data, pd.DataFrame):
            return _to_float_tensor(data.values)
        else:
            # 尝试转换为numpy然后tensor
            return _to_float_tensor(np.asarray(data))
    
    def _convert_sequence_data(self, sequences: List) -> torch.Tensor:
        """转换序列数据"""
        # 对于序列数据，我们返回一个特殊的标记
        # 实际的数据会在collate_fn中处理
        self._sequences = [_to_float_tensor(seq) for seq in sequences]
        return torch.zeros(len(sequences))  # 占位符
    
    def _standardize_data(self):
        """标准化数据"""
        if hasattr(self, '_sequences'):
            # 序列数据标准化
            for i, seq in enumerate(self._sequences):
                mean = seq.mean(dim=0, keepdim=True)
                std = seq.std(dim=0, keepdim=True).clamp_min(1e-6)
                self._sequences[i] = (seq - mean) / std
        else:
            # 普通数据标准化
            mean = self.X.mean(dim=0, keepdim=True)
            std = self.X.std(dim=0, keepdim=True).clamp_min(1e-6)
            self.X = (self.X - mean) / std
    
    def __len__(self):
        return self.sample_count
    
    def __getitem__(self, idx):
        if hasattr(self, '_sequences'):
            # 序列数据
            if self.y is None:
                return self._sequences[idx]
            return self._sequences[idx], self.y[idx]
        else:
            # 普通数据
            if self.y is None:
                return self.X[idx]
            return self.X[idx], self.y[idx]
    
    @property
    def sample_count(self):
        return len(self._sequences) if hasattr(self, '_sequences') else self.X.shape[0]

def universal_collate_fn(batch: List) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """通用collate函数，处理各种数据格式"""
    if not batch:
        return torch.empty(0), None, None
    
    # 检查是否有标签
    has_label = isinstance(batch[0], (tuple, list)) and len(batch[0]) == 2
    
    if has_label:
        xs, ys = zip(*batch)
        ys = torch.stack(ys, dim=0).long()
    else:
        xs = batch
        ys = None
    
    # 检查是否是序列数据
    if isinstance(xs[0], torch.Tensor) and xs[0].dim() > 1:
        # 普通tensor数据
        X = torch.stack(xs, dim=0)
        return X, ys, None
    else:
        # 序列数据，需要padding
        lengths = torch.tensor([x.shape[0] for x in xs], dtype=torch.long)
        max_len = int(lengths.max().item())
        feat_dim = xs[0].shape[1] if len(xs[0].shape) > 1 else 1
        
        X_pad = xs[0].new_full((len(xs), max_len, feat_dim), 0.0)
        for i, x in enumerate(xs):
            T = x.shape[0]
            X_pad[i, :T, :] = x
        
        return X_pad, ys, lengths

class UniversalConverter:
    """通用数据转换器"""
    
    def __init__(self):
        self.converters = {}
        self._register_default_converters()
    
    def _register_default_converters(self):
        """注册默认转换器"""
        self.converters["numpy_array"] = self._convert_numpy
        self.converters["torch_tensor"] = self._convert_torch
        self.converters["pandas_dataframe"] = self._convert_pandas
        self.converters["sequence_list"] = self._convert_sequence_list
    
    def register_converter(self, data_type: str, converter_func: Callable):
        """注册自定义转换器"""
        self.converters[data_type] = converter_func
    
    def _convert_numpy(self, data: np.ndarray, labels: Optional[Any] = None, **kwargs) -> Tuple[Dataset, Optional[Callable], DataProfile]:
        """转换numpy数组"""
        profile = analyze_data_profile(data, labels)
        dataset = UniversalDataset(data, labels, profile, **kwargs)
        collate_fn = universal_collate_fn if profile.is_sequence else None
        return dataset, collate_fn, profile
    
    def _convert_torch(self, data: torch.Tensor, labels: Optional[Any] = None, **kwargs) -> Tuple[Dataset, Optional[Callable], DataProfile]:
        """转换torch tensor"""
        profile = analyze_data_profile(data, labels)
        dataset = UniversalDataset(data, labels, profile, **kwargs)
        collate_fn = universal_collate_fn if profile.is_sequence else None
        return dataset, collate_fn, profile
    
    def _convert_pandas(self, data: pd.DataFrame, labels: Optional[Any] = None, **kwargs) -> Tuple[Dataset, Optional[Callable], DataProfile]:
        """转换pandas DataFrame"""
        profile = analyze_data_profile(data, labels)
        dataset = UniversalDataset(data, labels, profile, **kwargs)
        return dataset, None, profile
    
    def _convert_sequence_list(self, data: List, labels: Optional[Any] = None, **kwargs) -> Tuple[Dataset, Optional[Callable], DataProfile]:
        """转换序列列表"""
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
        通用数据转换方法
        
        Args:
            data: 输入数据
            labels: 标签数据
            data_type: 指定数据类型，如果为None则自动检测
            **kwargs: 其他参数
        
        Returns:
            dataset: PyTorch数据集
            collate_fn: 数据加载时的collate函数
            profile: 数据特征档案
        """
        if data_type is None:
            profile = analyze_data_profile(data, labels)
            data_type = profile.data_type
        else:
            profile = analyze_data_profile(data, labels)
        
        converter = self.converters.get(data_type)
        if converter is None:
            # 尝试通用转换
            return self._convert_numpy(data, labels, **kwargs)
        
        return converter(data, labels, **kwargs)

# 全局转换器实例
universal_converter = UniversalConverter()

def convert_to_torch_dataset(
    data: Any,
    labels: Optional[Any] = None,
    data_type: Optional[str] = None,
    **kwargs
) -> Tuple[Dataset, Optional[Callable], DataProfile]:
    """
    便捷函数：将任意数据转换为PyTorch数据集
    
    Args:
        data: 输入数据
        labels: 标签数据
        data_type: 指定数据类型，如果为None则自动检测
        **kwargs: 其他参数
    
    Returns:
        dataset: PyTorch数据集
        collate_fn: 数据加载时的collate函数
        profile: 数据特征档案
    """
    return universal_converter.convert(data, labels, data_type, **kwargs)
