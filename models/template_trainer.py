"""
Template Trainer
Handles training of template-based models with proper metrics
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
from typing import Dict, Any, Tuple
import numpy as np

logger = logging.getLogger(__name__)

class TemplateTrainer:
    """Trains template-based models"""
    
    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model = model.to(device)
        self.device = device
        
    def train_model(self, train_loader: DataLoader, 
                   lr: float = 0.001, epochs: int = 10, **kwargs) -> Tuple[nn.Module, Dict[str, float]]:
        """
        Train the model and return trained model + metrics
        
        Args:
            train_loader: Training data loader
            lr: Learning rate
            epochs: Number of epochs
            **kwargs: Additional training parameters
            
        Returns:
            Tuple[nn.Module, Dict[str, float]]: (trained_model, metrics)
        """
        
        # Setup optimizer and loss
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        self.model.train()
        
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        all_predictions = []
        all_labels = []
        
        logger.info(f"Starting training: {epochs} epochs, lr={lr}")
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            
            for batch_idx, (data, targets) in enumerate(train_loader):
                data, targets = data.to(self.device), targets.to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(data)
                loss = criterion(outputs, targets)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Statistics
                epoch_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                epoch_total += targets.size(0)
                epoch_correct += (predicted == targets).sum().item()
                
                # Collect for final metrics
                if epoch == epochs - 1:  # Last epoch
                    all_predictions.extend(predicted.cpu().numpy())
                    all_labels.extend(targets.cpu().numpy())
            
            epoch_acc = epoch_correct / epoch_total
            avg_loss = epoch_loss / len(train_loader)
            
            if epoch % max(1, epochs // 5) == 0 or epoch == epochs - 1:
                logger.info(f"Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Acc={epoch_acc:.4f}")
            
            # Final epoch statistics
            if epoch == epochs - 1:
                total_loss = avg_loss
                correct_predictions = epoch_correct
                total_samples = epoch_total
        
        # Calculate final metrics
        final_accuracy = correct_predictions / total_samples
        final_f1 = self._calculate_f1_score(all_labels, all_predictions)
        
        metrics = {
            'acc': final_accuracy,
            'macro_f1': final_f1,
            'loss': total_loss
        }
        
        logger.info(f"Training completed. Final metrics: {metrics}")
        
        return self.model, metrics
    
    def _calculate_f1_score(self, y_true, y_pred) -> float:
        """Calculate macro F1 score"""
        try:
            from sklearn.metrics import f1_score
            return f1_score(y_true, y_pred, average='macro', zero_division=0)
        except ImportError:
            # Simple F1 calculation if sklearn not available
            return self._simple_f1_score(y_true, y_pred)
    
    def _simple_f1_score(self, y_true, y_pred) -> float:
        """Simple F1 score calculation without sklearn"""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        unique_labels = np.union1d(y_true, y_pred)
        f1_scores = []
        
        for label in unique_labels:
            tp = np.sum((y_true == label) & (y_pred == label))
            fp = np.sum((y_true != label) & (y_pred == label))
            fn = np.sum((y_true == label) & (y_pred != label))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            f1_scores.append(f1)
        
        return np.mean(f1_scores) if f1_scores else 0.0

def train_template_model(model: nn.Module, train_loader: DataLoader, device: str, **kwargs) -> Tuple[nn.Module, Dict[str, float]]:
    """
    Convenience function to train a template model
    
    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        device: Device for training
        **kwargs: Training parameters (lr, epochs, etc.)
        
    Returns:
        Tuple[nn.Module, Dict[str, float]]: (trained_model, metrics)
    """
    trainer = TemplateTrainer(model, device)
    return trainer.train_model(train_loader, **kwargs)