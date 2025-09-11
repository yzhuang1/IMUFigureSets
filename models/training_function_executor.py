"""
Training Function Executor
Loads and executes training functions from JSON files
"""

import json
import logging
import torch
from typing import Dict, Any, Tuple, Optional, List
from pathlib import Path
import importlib.util
import tempfile
import os
import time

logger = logging.getLogger(__name__)

class TrainingFunctionExecutor:
    """Executes training functions loaded from JSON files"""
    
    def __init__(self):
        self.code_storage_dir = Path("generated_training_functions")
        self.code_storage_dir.mkdir(exist_ok=True)
        # Auto-detect best available device
        self.default_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.default_device == 'cuda':
            logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("Using CPU for training")
    
    def load_training_function(self, filepath: str) -> Dict[str, Any]:
        """Load training function data from JSON file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"Loaded training function: {data.get('model_name', 'Unknown')}")
            logger.info(f"Reasoning: {data.get('reasoning', 'No reasoning provided')}")
            
            return data
        except Exception as e:
            logger.error(f"Failed to load training function from {filepath}: {e}")
            raise
    
    def execute_training_function(
        self, 
        training_data: Dict[str, Any],
        X_train: torch.Tensor,
        y_train: torch.Tensor, 
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        device: str = None,
        **hyperparams
    ) -> Tuple[torch.nn.Module, Dict[str, Any]]:
        """Execute training function with given data and hyperparameters"""
        
        try:
            # Use GPU if available, otherwise fall back to provided device or CPU
            if device is None:
                device = self.default_device
            
            logger.info(f"Using device: {device}")
            
            # Move tensors to device
            X_train = X_train.to(device)
            y_train = y_train.to(device)
            X_val = X_val.to(device)
            y_val = y_val.to(device)
            
            # Extract training code
            training_code = training_data['training_code']
            model_name = training_data['model_name']
            
            logger.info(f"Executing training function: {model_name}")
            logger.info(f"Hyperparameters: {hyperparams}")
            
            # Create namespace for code execution
            namespace = {}
            
            # Execute the function definition
            exec(training_code, namespace)
            
            # Get the train_model function
            if 'train_model' not in namespace:
                raise ValueError("train_model function not found in training code")
            
            train_model = namespace['train_model']
            
            # Merge default hyperparameters with provided ones
            default_hyperparams = training_data.get('hyperparameters', {})
            final_hyperparams = {**default_hyperparams, **hyperparams}
            
            # Execute training
            logger.info(f"Starting training with hyperparameters: {final_hyperparams}")
            
            model, metrics = train_model(
                X_train, y_train, X_val, y_val,
                device=device,
                **final_hyperparams
            )
            
            # Add metadata to metrics
            metrics.update({
                'model_name': model_name,
                'training_function_source': 'ai_generated',
                'hyperparameters_used': final_hyperparams
            })
            
            logger.info(f"Training completed successfully: {metrics}")
            
            return model, metrics
            
        except Exception as e:
            logger.error(f"Training execution failed: {e}")
            logger.error(f"Training code: {training_code[:200]}...")
            raise
    
    def validate_training_function(self, training_data: Dict[str, Any]) -> bool:
        """Validate that training function data is complete and correct"""
        try:
            required_fields = ['model_name', 'training_code', 'hyperparameters', 'bo_parameters']
            
            for field in required_fields:
                if field not in training_data:
                    logger.error(f"Missing required field: {field}")
                    return False
            
            # Check that training code compiles
            training_code = training_data['training_code']
            try:
                compile(training_code, '<string>', 'exec')
            except SyntaxError as e:
                logger.error(f"Syntax error in training code: {e}")
                return False
            
            # Check that hyperparameters and bo_parameters are valid
            hyperparams = training_data['hyperparameters']
            bo_params = training_data['bo_parameters']
            
            if not isinstance(hyperparams, dict):
                logger.error("hyperparameters must be a dictionary")
                return False
            
            if not isinstance(bo_params, list):
                logger.error("bo_parameters must be a list")
                return False
            
            logger.info("Training function validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Training function validation failed: {e}")
            return False
    
    def test_training_function(
        self, 
        training_data: Dict[str, Any],
        X_sample: torch.Tensor,
        y_sample: torch.Tensor
    ) -> bool:
        """Test training function with small sample data"""
        try:
            # Create minimal test data
            test_size = min(20, len(X_sample))
            X_test = X_sample[:test_size]
            y_test = y_sample[:test_size]
            
            # Split into train/val
            split_idx = test_size // 2
            X_train_test = X_test[:split_idx]
            y_train_test = y_test[:split_idx]
            X_val_test = X_test[split_idx:]
            y_val_test = y_test[split_idx:]
            
            # Test with minimal hyperparameters
            test_hyperparams = {
                'epochs': 1,
                'batch_size': min(4, len(X_train_test))
            }
            
            # Add other required hyperparameters with minimal values
            default_hyperparams = training_data.get('hyperparameters', {})
            for key, value in default_hyperparams.items():
                if key not in test_hyperparams:
                    if isinstance(value, (int, float)) and key != 'lr':
                        test_hyperparams[key] = min(value, 32) if key == 'hidden_size' else value
                    else:
                        test_hyperparams[key] = value
            
            # Execute training function
            model, metrics = self.execute_training_function(
                training_data,
                X_train_test, y_train_test,
                X_val_test, y_val_test,
                device=self.default_device,
                **test_hyperparams
            )
            
            # Validate outputs
            if not hasattr(model, 'eval'):
                logger.error("Returned object is not a PyTorch model")
                return False
            
            if not isinstance(metrics, dict):
                logger.error("Metrics should be a dictionary")
                return False
            
            logger.info("Training function test passed")
            return True
            
        except Exception as e:
            logger.error(f"Training function test failed: {e}")
            return False
    
    def save_training_function(self, training_data: Dict[str, Any], filename: Optional[str] = None) -> str:
        """Save training function data to JSON file"""
        
        if filename is None:
            import time
            timestamp = int(time.time())
            model_name = training_data.get('model_name', 'unknown')
            filename = f"training_function_{model_name}_{timestamp}.json"
        
        filepath = self.code_storage_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Training function saved to: {filepath}")
        return str(filepath)
    
    def list_available_training_functions(self) -> List[Dict[str, Any]]:
        """List all available training functions"""
        functions = []
        
        for json_file in self.code_storage_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                functions.append({
                    'filepath': str(json_file),
                    'model_name': data.get('model_name', 'Unknown'),
                    'reasoning': data.get('reasoning', 'No reasoning'),
                    'confidence': data.get('confidence', 0.0),
                    'timestamp': data.get('timestamp', 0),
                    'hyperparameters': data.get('hyperparameters', {}),
                    'bo_parameters': data.get('bo_parameters', [])
                })
                
            except Exception as e:
                logger.warning(f"Failed to read {json_file}: {e}")
                continue
        
        # Sort by timestamp (newest first)
        functions.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return functions

class BO_TrainingObjective:
    """Objective function for Bayesian Optimization using generated training functions"""
    
    def __init__(
        self, 
        training_data: Dict[str, Any],
        X_subset: torch.Tensor,
        y_subset: torch.Tensor,
        device: str = None
    ):
        self.training_data = training_data
        self.X_subset = X_subset
        self.y_subset = y_subset
        # Auto-detect device if not specified
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.executor = TrainingFunctionExecutor()
        
        # Split data for train/val
        from sklearn.model_selection import train_test_split
        X_array = X_subset.cpu().numpy()
        y_array = y_subset.cpu().numpy()
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_array, y_array, test_size=0.2, random_state=42, stratify=y_array
        )
        
        # Create tensors on the correct device
        self.X_train = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        self.y_train = torch.tensor(y_train, dtype=torch.long, device=self.device)
        self.X_val = torch.tensor(X_val, dtype=torch.float32, device=self.device)
        self.y_val = torch.tensor(y_val, dtype=torch.long, device=self.device)
    
    def __call__(self, hparams: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Execute training function with given hyperparameters"""
        t0 = time.time()
        
        try:
            # Convert numpy types to Python types
            processed_hparams = {}
            for key, value in hparams.items():
                if hasattr(value, 'item'):
                    value = value.item()
                
                # Type conversion
                if key in ['epochs', 'batch_size', 'hidden_size', 'num_layers']:
                    value = int(value)
                elif key in ['lr', 'dropout']:
                    value = float(value)
                
                processed_hparams[key] = value
            
            # Execute training
            model, metrics = self.executor.execute_training_function(
                self.training_data,
                self.X_train, self.y_train,
                self.X_val, self.y_val,
                device=self.device,
                **processed_hparams
            )
            
            # Return objective value (F1 score or accuracy)
            objective_value = metrics.get('macro_f1', metrics.get('val_accuracy', 0.0))
            
            objective_time = time.time() - t0
            logger.info(f"[PROFILE] objective(train+eval) took {objective_time:.3f}s")
            
            return float(objective_value), metrics
            
        except Exception as e:
            logger.error(f"BO training objective failed: {e}")
            objective_time = time.time() - t0
            logger.info(f"[PROFILE] objective(train+eval) FAILED took {objective_time:.3f}s")
            return 0.0, {"error": str(e)}

# Global instance
training_executor = TrainingFunctionExecutor()

def execute_training_from_json(
    filepath: str,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    device: str = 'cpu',
    **hyperparams
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """
    Convenience function: Load and execute training function from JSON file
    """
    training_data = training_executor.load_training_function(filepath)
    return training_executor.execute_training_function(
        training_data, X_train, y_train, X_val, y_val, device, **hyperparams
    )