"""
Class Balancing Module
Handles imbalanced datasets with techniques like weighted loss, sampling, and SMOTE
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.utils.class_weight import compute_class_weight
from typing import Dict, Any, Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)

def compute_class_weights(y: np.ndarray, method: str = 'balanced') -> Dict[int, float]:
    """
    Compute class weights for imbalanced dataset
    
    Args:
        y: Label array
        method: 'balanced' or 'balanced_subsample'
        
    Returns:
        Dictionary mapping class indices to weights
    """
    unique_classes = np.unique(y)
    weights = compute_class_weight(method, classes=unique_classes, y=y)
    weight_dict = {int(cls): float(weight) for cls, weight in zip(unique_classes, weights)}
    
    logger.info(f"Computed class weights using {method}:")
    for cls, weight in weight_dict.items():
        count = np.sum(y == cls)
        logger.info(f"  Class {cls}: weight={weight:.3f}, count={count}")
    
    return weight_dict

def create_weighted_loss_function(class_weights: Dict[int, float], device: str = 'cpu') -> nn.CrossEntropyLoss:
    """
    Create weighted CrossEntropyLoss for imbalanced classes
    
    Args:
        class_weights: Dictionary mapping class indices to weights
        device: Device to place weights tensor
        
    Returns:
        Weighted CrossEntropyLoss function
    """
    # Convert to tensor in correct order
    max_class = max(class_weights.keys())
    weight_tensor = torch.zeros(max_class + 1)
    
    for cls_idx, weight in class_weights.items():
        weight_tensor[cls_idx] = weight
    
    weight_tensor = weight_tensor.to(device)
    
    logger.info(f"Created weighted loss function with weights: {weight_tensor}")
    return nn.CrossEntropyLoss(weight=weight_tensor)

def create_balanced_sampler(y: np.ndarray) -> WeightedRandomSampler:
    """
    Create weighted random sampler for balanced batch sampling
    
    Args:
        y: Label array
        
    Returns:
        WeightedRandomSampler for balanced sampling
    """
    # Compute class weights
    class_weights = compute_class_weights(y, method='balanced')
    
    # Create sample weights
    sample_weights = np.array([class_weights[int(label)] for label in y])
    sample_weights = torch.from_numpy(sample_weights).double()
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    logger.info(f"Created balanced sampler for {len(y)} samples")
    return sampler

class BalancedDataLoader:
    """DataLoader wrapper with automatic class balancing"""
    
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        balancing_method: str = 'weighted_loss',
        **kwargs
    ):
        """
        Args:
            dataset: PyTorch dataset
            batch_size: Batch size
            balancing_method: 'weighted_loss', 'weighted_sampling', or 'both'
            **kwargs: Additional DataLoader arguments
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.balancing_method = balancing_method
        
        # Extract labels from dataset
        if hasattr(dataset, 'y') and dataset.y is not None:
            if isinstance(dataset.y, torch.Tensor):
                self.labels = dataset.y.cpu().numpy()
            else:
                self.labels = np.array(dataset.y)
        else:
            # Try to extract labels by iterating through dataset
            labels = []
            for i in range(len(dataset)):
                item = dataset[i]
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    labels.append(item[1].item() if isinstance(item[1], torch.Tensor) else item[1])
                else:
                    raise ValueError("Cannot extract labels from dataset for balancing")
            self.labels = np.array(labels)
        
        # Compute class weights
        self.class_weights = compute_class_weights(self.labels)
        
        # Set up sampler if needed
        sampler = None
        if balancing_method in ['weighted_sampling', 'both']:
            sampler = create_balanced_sampler(self.labels)
            kwargs['shuffle'] = False  # Can't shuffle with custom sampler
        
        # Create DataLoader
        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            **kwargs
        )
        
        logger.info(f"Created BalancedDataLoader with method: {balancing_method}")
    
    def get_weighted_loss_function(self, device: str = 'cpu') -> nn.CrossEntropyLoss:
        """Get weighted loss function for this dataset"""
        return create_weighted_loss_function(self.class_weights, device)
    
    def __iter__(self):
        return iter(self.loader)
    
    def __len__(self):
        return len(self.loader)

def apply_focal_loss(alpha: Optional[torch.Tensor] = None, gamma: float = 2.0) -> nn.Module:
    """
    Create Focal Loss for handling extreme class imbalance
    
    Args:
        alpha: Class weights tensor
        gamma: Focusing parameter (higher = more focus on hard examples)
        
    Returns:
        Focal Loss module
    """
    
    class FocalLoss(nn.Module):
        def __init__(self, alpha=None, gamma=2.0):
            super(FocalLoss, self).__init__()
            self.alpha = alpha
            self.gamma = gamma
            self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        
        def forward(self, inputs, targets):
            ce_loss = self.ce_loss(inputs, targets)
            pt = torch.exp(-ce_loss)
            focal_loss = (1 - pt) ** self.gamma * ce_loss
            
            if self.alpha is not None:
                alpha_t = self.alpha.gather(0, targets)
                focal_loss = alpha_t * focal_loss
            
            return focal_loss.mean()
    
    return FocalLoss(alpha, gamma)

def get_balanced_batch_metrics(y_batch: torch.Tensor) -> Dict[str, float]:
    """
    Compute batch-level class distribution metrics
    
    Args:
        y_batch: Batch labels tensor
        
    Returns:
        Dictionary with class distribution statistics
    """
    y_np = y_batch.cpu().numpy()
    unique, counts = np.unique(y_np, return_counts=True)
    
    total = len(y_np)
    metrics = {
        'batch_size': total,
        'num_classes_present': len(unique),
        'class_distribution': {int(cls): int(count) for cls, count in zip(unique, counts)},
        'class_proportions': {int(cls): float(count/total) for cls, count in zip(unique, counts)},
        'imbalance_ratio': float(max(counts) / min(counts)) if len(counts) > 1 else 1.0
    }
    
    return metrics

def recommend_balancing_strategy(y: np.ndarray) -> Dict[str, Any]:
    """
    Analyze class distribution and recommend balancing strategy
    
    Args:
        y: Label array
        
    Returns:
        Dictionary with analysis and recommendations
    """
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)
    
    # Calculate imbalance metrics
    min_count = min(counts)
    max_count = max(counts)
    imbalance_ratio = max_count / min_count
    
    # Class proportions
    proportions = counts / total
    min_proportion = min(proportions)
    
    # Recommendations based on imbalance severity
    if imbalance_ratio <= 3:
        strategy = "mild_imbalance"
        recommendations = ["Standard training should work", "Consider class_weight='balanced'"]
    elif imbalance_ratio <= 10:
        strategy = "moderate_imbalance"
        recommendations = ["Use weighted loss function", "Consider weighted sampling"]
    elif imbalance_ratio <= 100:
        strategy = "severe_imbalance"
        recommendations = ["Use weighted loss + weighted sampling", "Consider Focal Loss", "Evaluate with balanced metrics"]
    else:
        strategy = "extreme_imbalance"
        recommendations = ["Use Focal Loss with alpha weights", "Stratified sampling essential", "Consider data augmentation for minority classes"]
    
    analysis = {
        'total_samples': total,
        'num_classes': len(unique),
        'class_counts': {int(cls): int(count) for cls, count in zip(unique, counts)},
        'class_proportions': {int(cls): float(prop) for cls, prop in zip(unique, proportions)},
        'imbalance_ratio': float(imbalance_ratio),
        'min_class_proportion': float(min_proportion),
        'strategy': strategy,
        'recommendations': recommendations
    }
    
    logger.info(f"Class imbalance analysis:")
    logger.info(f"  Strategy: {strategy}")
    logger.info(f"  Imbalance ratio: {imbalance_ratio:.2f}")
    logger.info(f"  Recommendations: {', '.join(recommendations)}")
    
    return analysis