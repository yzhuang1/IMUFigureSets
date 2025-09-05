"""
AI-Enhanced Objective Function
Integrates universal data converter and AI model selector
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader
import logging

from adapters.universal_converter import convert_to_torch_dataset
from models.ai_model_selector import select_model_for_data
from models.dynamic_model_registry import build_model_from_recommendation
from train import train_one_model
from evaluation.evaluate import evaluate_model

logger = logging.getLogger(__name__)

class AIEnhancedObjective:
    """AI-enhanced objective function class"""
    
    def __init__(self, data, labels=None, device="cpu", **kwargs):
        """
        Initialize objective function
        
        Args:
            data: Input data
            labels: Label data
            device: Device
            **kwargs: Other parameters
        """
        self.data = data
        self.labels = labels
        self.device = device
        self.kwargs = kwargs
        
        # Preprocess data
        self._preprocess_data()
        
        # Get AI recommendation
        self._get_ai_recommendation()
    
    def _preprocess_data(self):
        """Preprocess data"""
        logger.info("Preprocessing data...")
        
        # Convert data
        self.dataset, self.collate_fn, self.data_profile = convert_to_torch_dataset(
            self.data, self.labels, **self.kwargs
        )
        
        logger.info(f"Data preprocessing completed: {self.data_profile}")
    
    def _get_ai_recommendation(self):
        """Get AI model recommendation"""
        logger.info("Getting AI model recommendation...")
        
        self.recommendation = select_model_for_data(self.data_profile.to_dict())
        
        logger.info(f"AI recommended model: {self.recommendation.model_name}")
        logger.info(f"Recommendation reason: {self.recommendation.reasoning}")
    
    def _determine_input_shape(self) -> tuple:
        """Determine input shape"""
        if self.data_profile.is_sequence:
            return (self.data_profile.feature_count,)
        elif self.data_profile.is_image:
            if (self.data_profile.channels and 
                self.data_profile.height and 
                self.data_profile.width):
                return (self.data_profile.channels, 
                       self.data_profile.height, 
                       self.data_profile.width)
            else:
                return (3, 32, 32)  # Default image size
        elif self.data_profile.is_tabular:
            return (self.data_profile.feature_count,)
        else:
            return (self.data_profile.feature_count,)
    
    def _create_model(self, hparams: Dict[str, Any]) -> torch.nn.Module:
        """Create model"""
        # Determine input shape and number of classes
        input_shape = self._determine_input_shape()
        num_classes = self.data_profile.label_count if self.data_profile.has_labels else 2
        
        # Merge hyperparameters
        model_params = self.recommendation.hyperparameters.copy()
        model_params.update(hparams)
        
        # Build model
        model = build_model_from_recommendation(
            self.recommendation, input_shape, num_classes
        )
        
        return model
    
    def __call__(self, hparams: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """
        Objective function call
        
        Args:
            hparams: Hyperparameters
        
        Returns:
            Tuple[float, Dict]: (Objective value, detailed metrics)
        """
        try:
            # Create model
            model = self._create_model(hparams)
            model.to(self.device)
            
            # Create data loader
            batch_size = hparams.get('batch_size', 64)
            loader = DataLoader(
                self.dataset, 
                batch_size=batch_size, 
                shuffle=True, 
                collate_fn=self.collate_fn
            )
            
            # Train model
            epochs = hparams.get('epochs', 3)
            lr = hparams.get('lr', 1e-3)
            
            trained_model = train_one_model(
                model, loader, device=self.device, epochs=epochs, lr=lr
            )
            
            # Evaluate model
            metrics = evaluate_model(trained_model, loader, device=self.device)
            
            # Return objective value (using macro_f1 as main metric)
            objective_value = metrics.get("macro_f1", 0.0)
            
            # Add extra information
            metrics.update({
                "model_name": self.recommendation.model_name,
                "confidence": self.recommendation.confidence,
                "data_type": self.data_profile.data_type,
                "sample_count": self.data_profile.sample_count,
                "feature_count": self.data_profile.feature_count
            })
            
            logger.info(f"Objective function evaluation completed: {objective_value:.4f}")
            
            return float(objective_value), metrics
        
        except Exception as e:
            logger.error(f"Objective function evaluation failed: {e}")
            return 0.0, {"error": str(e)}

def create_ai_enhanced_objective(data, labels=None, device="cpu", **kwargs):
    """
    Create AI-enhanced objective function
    
    Args:
        data: Input data
        labels: Label data
        device: Device
        **kwargs: Other parameters
    
    Returns:
        AIEnhancedObjective: Objective function instance
    """
    return AIEnhancedObjective(data, labels, device, **kwargs)

def objective_for_dataset_ai_enhanced(
    data, 
    labels=None, 
    device="cpu", 
    hparams: Optional[Dict[str, Any]] = None
) -> Tuple[float, Dict[str, Any]]:
    """
    Convenience function: Create AI-enhanced objective function for dataset and evaluate
    
    Args:
        data: Input data
        labels: Label data
        device: Device
        hparams: Hyperparameters
    
    Returns:
        Tuple[float, Dict]: (Objective value, detailed metrics)
    """
    hparams = hparams or {"lr": 1e-3, "epochs": 3, "hidden": 64}
    
    objective = create_ai_enhanced_objective(data, labels, device)
    return objective(hparams)

# Backward compatible alias
objective_for_dataset = objective_for_dataset_ai_enhanced
