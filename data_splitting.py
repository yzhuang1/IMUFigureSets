"""
Centralized Data Splitting Module
Handles consistent train/validation/test splits across all pipeline components
Prevents data leakage and ensures reproducible splits
"""

import numpy as np
import torch
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from typing import Dict, Tuple, Optional, Any, List
import logging
from dataclasses import dataclass
from class_balancing import recommend_balancing_strategy

logger = logging.getLogger(__name__)

@dataclass
class DataSplits:
    """Container for consistent data splits"""
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray
    class_weights: Dict[int, float]
    balancing_strategy: Dict[str, Any]
    standardization_stats: Optional[Dict[str, torch.Tensor]] = None

class CentralizedDataSplitter:
    """Centralized data splitting with consistent splits across all components"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.splits = None
        
    def create_splits(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        test_size: float = 0.2,
        val_size: float = 0.2,
        stratify: bool = True,
        compute_class_weights: bool = True
    ) -> DataSplits:
        """
        Create consistent train/val/test splits with proper stratification
        
        Args:
            X: Feature data
            y: Labels
            test_size: Proportion for test set (default 0.2 = 20%)
            val_size: Proportion for validation set from remaining data (default 0.2 = 20%)
            stratify: Whether to stratify splits by class labels
            compute_class_weights: Whether to compute class weights for imbalanced data
        
        Returns:
            DataSplits object containing all splits and metadata
        """
        logger.info(f"Creating centralized data splits with test_size={test_size}, val_size={val_size}")
        logger.info(f"Input data shape: X={X.shape}, y={y.shape}")
        logger.info(f"Class distribution: {np.bincount(y)}")
        
        # First split: separate test set
        stratify_first = y if stratify else None
        X_temp, X_test, y_temp, y_test, idx_temp, idx_test = train_test_split(
            X, y, np.arange(len(X)),
            test_size=test_size,
            random_state=self.random_state,
            stratify=stratify_first
        )
        
        # Second split: train/validation from remaining data
        stratify_second = y_temp if stratify else None
        X_train, X_val, y_train, y_val, idx_train_temp, idx_val_temp = train_test_split(
            X_temp, y_temp, np.arange(len(X_temp)),
            test_size=val_size,
            random_state=self.random_state,
            stratify=stratify_second
        )
        
        # Map back to original indices
        train_indices = idx_temp[idx_train_temp]
        val_indices = idx_temp[idx_val_temp]
        test_indices = idx_test
        
        # Compute class weights for imbalanced data
        class_weights = {}
        if compute_class_weights:
            unique_classes = np.unique(y_train)
            weights = compute_class_weight(
                'balanced',
                classes=unique_classes,
                y=y_train
            )
            class_weights = dict(zip(unique_classes, weights))
            logger.info(f"Computed class weights: {class_weights}")
        
        # Analyze class imbalance and recommend strategy
        balancing_strategy = recommend_balancing_strategy(y_train)
        
        # Log split information
        logger.info(f"Final splits - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        logger.info(f"Train class distribution: {np.bincount(y_train)}")
        logger.info(f"Val class distribution: {np.bincount(y_val)}")
        logger.info(f"Test class distribution: {np.bincount(y_test)}")
        logger.info(f"Recommended balancing strategy: {balancing_strategy['strategy']}")
        
        self.splits = DataSplits(
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            train_indices=train_indices,
            val_indices=val_indices,
            test_indices=test_indices,
            class_weights=class_weights,
            balancing_strategy=balancing_strategy
        )
        
        return self.splits
    
    def get_bo_subset(self, max_samples: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get a subset of training data for Bayesian Optimization
        Maintains class distribution and uses consistent random state
        
        Args:
            max_samples: Maximum number of samples for BO (if None, uses config.bo_sample_num)
            
        Returns:
            Tuple of (X_bo, y_bo) subset
        """
        if self.splits is None:
            raise ValueError("Must call create_splits() first")

        if max_samples is None:
            from config import config
            max_samples = config.bo_sample_num

        X_train, y_train = self.splits.X_train, self.splits.y_train
        
        if len(X_train) <= max_samples:
            logger.info(f"Using all {len(X_train)} training samples for BO")
            return X_train, y_train
        
        # Stratified sampling for BO subset
        _, X_bo, _, y_bo = train_test_split(
            X_train, y_train,
            test_size=max_samples,
            random_state=self.random_state,
            stratify=y_train
        )
        
        logger.info(f"Created BO subset: {len(X_bo)} samples")
        logger.info(f"BO subset class distribution: {np.bincount(y_bo)}")
        
        return X_bo, y_bo
    
    def compute_standardization_stats(self) -> Dict[str, torch.Tensor]:
        """
        Compute standardization statistics from training data only
        
        Returns:
            Dictionary with 'mean' and 'std' tensors
        """
        if self.splits is None:
            raise ValueError("Must call create_splits() first")
        
        X_train_tensor = torch.from_numpy(self.splits.X_train.astype(np.float32))
        
        # For 3D ECG data (samples, time, features), compute stats across samples and time
        if X_train_tensor.dim() == 3:
            # Reshape to (samples*time, features) for stats computation
            X_flat = X_train_tensor.view(-1, X_train_tensor.size(-1))
            mean = X_flat.mean(dim=0, keepdim=True)
            std = X_flat.std(dim=0, keepdim=True).clamp_min(1e-6)
        else:
            mean = X_train_tensor.mean(dim=0, keepdim=True)
            std = X_train_tensor.std(dim=0, keepdim=True).clamp_min(1e-6)
        
        stats = {'mean': mean, 'std': std}
        self.splits.standardization_stats = stats
        
        logger.info(f"Computed standardization stats - mean shape: {mean.shape}, std shape: {std.shape}")
        
        return stats
    
    def get_splits(self) -> DataSplits:
        """Get the current data splits"""
        if self.splits is None:
            raise ValueError("Must call create_splits() first")
        return self.splits
    
    def get_cross_validation_folds(self, n_folds: int = 5) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create stratified k-fold cross-validation splits from training data
        
        Args:
            n_folds: Number of CV folds
            
        Returns:
            List of (train_idx, val_idx) tuples for each fold
        """
        if self.splits is None:
            raise ValueError("Must call create_splits() first")
        
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
        
        folds = []
        for train_idx, val_idx in skf.split(self.splits.X_train, self.splits.y_train):
            folds.append((train_idx, val_idx))
        
        logger.info(f"Created {n_folds}-fold cross-validation splits")
        return folds

# Global splitter instance for consistency across components
global_splitter = CentralizedDataSplitter()

def create_consistent_splits(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42
) -> DataSplits:
    """
    Convenience function to create consistent splits using global splitter
    
    Args:
        X: Feature data
        y: Labels  
        test_size: Test set proportion
        val_size: Validation set proportion
        random_state: Random seed for reproducibility
        
    Returns:
        DataSplits object
    """
    global global_splitter
    if global_splitter.random_state != random_state:
        global_splitter = CentralizedDataSplitter(random_state)
    
    return global_splitter.create_splits(
        X, y, 
        test_size=test_size, 
        val_size=val_size
    )

def get_bo_subset(max_samples: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """Get BO subset using global splitter"""
    if max_samples is None:
        from config import config
        max_samples = config.bo_sample_num
    return global_splitter.get_bo_subset(max_samples)

def get_current_splits() -> DataSplits:
    """Get current splits from global splitter"""
    return global_splitter.get_splits()

def compute_standardization_stats() -> Dict[str, torch.Tensor]:
    """Compute standardization stats using global splitter"""
    return global_splitter.compute_standardization_stats()