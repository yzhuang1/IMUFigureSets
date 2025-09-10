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
from models.ai_template_selector import select_template_for_data
from models.model_templates import create_model_from_template
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
        
        # Get AI template selection
        self._get_ai_template_selection()
    
    def _preprocess_data(self):
        """Preprocess data"""
        logger.info("Preprocessing data...")
        
        # Convert data
        self.dataset, self.collate_fn, self.data_profile = convert_to_torch_dataset(
            self.data, self.labels, **self.kwargs
        )
        
        logger.info(f"Data preprocessing completed: {self.data_profile}")
    
    def _get_ai_template_selection(self):
        """Get AI template selection"""
        logger.info("Getting AI template selection...")
        
        # Determine input shape and num classes for template selection
        input_shape = self._determine_input_shape()
        num_classes = self.data_profile.label_count if self.data_profile.has_labels else 2
        
        self.template_rec = select_template_for_data(
            self.data_profile.to_dict(),
            input_shape,
            num_classes
        )
        
        logger.info(f"AI selected template: {self.template_rec.template_name} -> {self.template_rec.model_name}")
        logger.info(f"Selection reason: {self.template_rec.reasoning}")
    
    def _determine_input_shape(self) -> tuple:
        """Determine input shape"""
        if self.data_profile.is_sequence:
            # For sequence data like ECG (samples, seq_len, features), use (seq_len, features)
            if len(self.data_profile.shape) == 3:
                return self.data_profile.shape[1:]  # (seq_len, features)
            else:
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
        """Create model using template with properly mapped parameters"""
        # Start with template's base configuration
        config = self.template_rec.config.copy()
        
        # Separate training params from model params
        training_params = ['lr', 'batch_size', 'epochs']
        
        # Update config with BO hyperparameters that are model params
        for param, value in hparams.items():
            if param in training_params:
                continue  # Skip training params
            
            # Direct mapping for matching parameter names
            if param in config:
                config[param] = value
            # Handle special cases for parameter name mapping
            elif param == 'hidden' and 'hidden_size' in config:
                config['hidden_size'] = value
            elif self.template_rec.template_name == "MLP" and param == 'hidden_size':
                # For MLP, map hidden_size to hidden_sizes list
                if 'hidden_sizes' in config:
                    config['hidden_sizes'] = [value, value // 2, value // 4]
        
        # Log the final configuration for debugging
        logger.info(f"Creating {self.template_rec.template_name} with config: {config}")
        
        model = create_model_from_template(self.template_rec.template_name, config)
        
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
            
            # Split dataset into train/validation (80/20 split) with FIXED seed
            from torch.utils.data import random_split
            train_size = int(0.8 * len(self.dataset))
            val_size = len(self.dataset) - train_size
            generator = torch.Generator().manual_seed(42)  # Fixed seed for reproducible splits
            train_dataset, val_dataset = random_split(self.dataset, [train_size, val_size], generator=generator)
            
            # Create separate data loaders
            batch_size = hparams.get('batch_size', 64)
            train_loader = DataLoader(
                train_dataset, 
                batch_size=batch_size, 
                shuffle=True, 
                collate_fn=self.collate_fn
            )
            val_loader = DataLoader(
                val_dataset, 
                batch_size=batch_size, 
                shuffle=False, 
                collate_fn=self.collate_fn
            )
            
            # Train model on training set
            epochs = hparams.get('epochs', 3)
            lr = hparams.get('lr', 1e-3)
            
            trained_model = train_one_model(
                model, train_loader, device=self.device, epochs=epochs, lr=lr
            )
            
            # Evaluate model on validation set (unseen data)
            metrics = evaluate_model(trained_model, val_loader, device=self.device)
            
            # Return objective value (using macro_f1 as main metric)
            objective_value = metrics.get("macro_f1", 0.0)
            
            # Add extra information
            metrics.update({
                "template_name": self.template_rec.template_name,
                "model_name": self.template_rec.model_name,
                "confidence": self.template_rec.confidence,
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
