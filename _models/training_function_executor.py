"""
Training Function Executor
Loads and executes training functions from JSON files
"""

import json
import logging
import torch
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from pathlib import Path
import importlib.util
import tempfile
import os
import time
from data_splitting import get_bo_subset, get_current_splits
from config import config

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
    
    def _convert_numpy_types(self, hyperparams: Dict[str, Any]) -> Dict[str, Any]:
        """Convert NumPy types to native Python types for PyTorch compatibility"""
        converted = {}
        for key, value in hyperparams.items():
            if isinstance(value, np.integer):
                converted[key] = int(value)
            elif isinstance(value, np.floating):
                converted[key] = float(value)
            elif isinstance(value, np.ndarray):
                converted[key] = value.tolist()
            elif isinstance(value, np.bool_):
                converted[key] = bool(value)
            else:
                converted[key] = value
        
        logger.debug(f"Converted hyperparameters: {converted}")
        return converted

    def _process_training_code(self, training_code: str) -> str:
        """Process training code to handle JSON escape sequences"""
        if isinstance(training_code, str):
            try:
                # Replace common JSON escape sequences that might cause issues
                training_code = training_code.replace('\\n', '\n')
                training_code = training_code.replace('\\t', '\t')
                training_code = training_code.replace('\\r', '\r')
                training_code = training_code.replace('\\"', '"')
                training_code = training_code.replace('\\\\', '\\')
            except Exception as e:
                logger.warning(f"Failed to process escape sequences: {e}")
        return training_code

    def _validate_hyperparameters(self, hyperparams: Dict[str, Any], model_name: str = None) -> Dict[str, Any]:
        """Validate hyperparameters (lightweight validation since BO handles constraints)"""
        validated = hyperparams.copy()
        
        # Ensure positive values for common parameters (basic safety checks)
        positive_params = ['lr', 'batch_size', 'epochs', 'hidden_size', 'embed_dim', 'd_model', 'num_heads']
        for param in positive_params:
            if param in validated and validated[param] <= 0:
                if param == 'lr':
                    validated[param] = 1e-3
                elif param == 'batch_size':
                    validated[param] = 32
                elif param == 'epochs':
                    validated[param] = 10
                elif param == 'num_heads':
                    validated[param] = 1
                else:
                    validated[param] = 128
                logger.warning(f"Fixed {param}: negative/zero -> {validated[param]}")
        
        # Ensure reasonable num_heads values (basic bounds)
        if 'num_heads' in validated:
            num_heads = int(validated['num_heads'])
            if num_heads > 16:
                validated['num_heads'] = 16
                logger.warning(f"Fixed num_heads: {num_heads} -> 16 (maximum reasonable value)")
        
        # Note: Constraint enforcement (embed_dim % num_heads == 0) is now handled at BO level
        logger.debug("Hyperparameter validation complete (constraint enforcement handled by BO)")
        
        return validated
    
    def validate_training_function(self, training_data: Dict[str, Any]) -> bool:
        """Validate training function data and model architecture"""
        try:
            # Check required fields
            required_fields = ['model_name', 'training_code', 'bo_config']
            for field in required_fields:
                if field not in training_data:
                    logger.error(f"Missing required field: {field}")
                    return False

            # Extract hyperparameters from bo_config
            bo_config = training_data['bo_config']
            hyperparams = {param: config["default"] for param, config in bo_config.items()}

            # Validate hyperparameters
            validated_hyperparams = self._validate_hyperparameters(
                hyperparams,
                training_data.get('model_name', 'Unknown')
            )
            
            # Test model instantiation
            return self._validate_model_instantiation(
                training_data['training_code'], 
                validated_hyperparams
            )
            
        except Exception as e:
            logger.error(f"Training function validation failed: {e}")
            return False
    
    def _validate_model_instantiation(self, training_code: str, hyperparams: Dict[str, Any]) -> bool:
        """Test model instantiation with given hyperparameters to catch architecture errors early"""
        try:
            # Create a test environment to instantiate the model
            test_globals = {
                'torch': torch,
                'nn': torch.nn,
                'F': torch.nn.functional,
                'math': __import__('math'),
                'numpy': np,
                'np': np
            }
            
            # Execute the training code to get the model classes
            exec(training_code, test_globals)
            
            # Try to find and instantiate the model with test parameters
            test_hyperparams = hyperparams.copy()
            test_hyperparams.update({
                'device': 'cpu',  # Use CPU for testing
                'lr': 1e-3,
                'epochs': 1,
                'batch_size': 2,
            })
            
            # Create dummy data for testing
            dummy_X_train = torch.randn(2, 10, 2)  # Small batch for testing
            dummy_y_train = torch.randint(0, 5, (2,))
            dummy_X_val = torch.randn(2, 10, 2)
            dummy_y_val = torch.randint(0, 5, (2,))
            
            # Try to execute just the model instantiation part
            if 'train_model' in test_globals:
                train_func = test_globals['train_model']
                # This will fail if there are parameter incompatibilities
                try:
                    # Just test parameter validation, don't run full training
                    model, _ = train_func(dummy_X_train, dummy_y_train, dummy_X_val, dummy_y_val, 'cpu', **test_hyperparams)
                    logger.info("Model architecture validation passed")
                    return True
                except Exception as e:
                    logger.warning(f"Model architecture validation failed: {e}")
                    return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Model validation failed: {e}")
            return False
    
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
            logger.info(f"Hyperparameters: {dict(hyperparams) if hyperparams else 'None'}")
            
            # Handle JSON escape sequences in training code
            training_code = self._process_training_code(training_code)

            # Create namespace for code execution
            namespace = {}

            # Execute the function definition
            exec(training_code, namespace)
            
            # Get the train_model function
            if 'train_model' not in namespace:
                raise ValueError("train_model function not found in training code")
            
            train_model = namespace['train_model']
            
            # Extract default hyperparameters from bo_config and merge with provided ones
            bo_config = training_data.get('bo_config', {})
            default_hyperparams = {param: config["default"] for param, config in bo_config.items()}
            final_hyperparams = {**default_hyperparams, **hyperparams}
            
            # Convert NumPy types to native Python types (required for PyTorch)
            final_hyperparams = self._convert_numpy_types(final_hyperparams)
            
            # Execute training
            logger.info(f"Starting training with hyperparameters: {dict(final_hyperparams) if final_hyperparams else 'None'}")
            
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
            
            logger.info(f"Training completed successfully: {dict(metrics) if metrics else 'None'}")
            
            return model, metrics
            
        except Exception as e:
            logger.error(f"Training execution failed: {e}")
            logger.error(f"Training code: {training_code[:200]}...")
            raise
    
    def validate_training_function(self, training_data: Dict[str, Any]) -> bool:
        """Validate that training function data is complete and correct"""
        try:
            required_fields = ['model_name', 'training_code', 'bo_config']

            for field in required_fields:
                if field not in training_data:
                    logger.error(f"Missing required field: {field}")
                    return False

            # Check that training code compiles
            training_code = training_data['training_code']

            # Handle JSON escape sequences properly
            training_code = self._process_training_code(training_code)

            try:
                compile(training_code, '<string>', 'exec')
            except SyntaxError as e:
                logger.error(f"Syntax error in training code: {e}")
                logger.error(f"First 200 chars of training code: {training_code[:200]}")
                return False

            # Check that bo_config is valid
            bo_config = training_data['bo_config']

            if not isinstance(bo_config, dict):
                logger.error("bo_config must be a dictionary")
                return False

            # Validate bo_config structure
            for param_name, config in bo_config.items():
                if not isinstance(config, dict):
                    logger.error(f"bo_config parameter '{param_name}' must be a dictionary")
                    return False
                if "default" not in config:
                    logger.error(f"bo_config parameter '{param_name}' missing 'default' value")
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
            
            # Add other required hyperparameters with minimal values from bo_config
            bo_config = training_data.get('bo_config', {})
            default_hyperparams = {param: config["default"] for param, config in bo_config.items()}
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
                
                bo_config = data.get('bo_config', {})
                functions.append({
                    'filepath': str(json_file),
                    'model_name': data.get('model_name', 'Unknown'),
                    'confidence': data.get('confidence', 0.0),
                    'timestamp': data.get('timestamp', 0),
                    'bo_config': bo_config,
                    'bo_parameters': list(bo_config.keys())
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
        X_subset: Optional[torch.Tensor] = None,
        y_subset: Optional[torch.Tensor] = None,
        device: str = None
    ):
        self.training_data = training_data
        # Auto-detect device if not specified
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.executor = TrainingFunctionExecutor()
        
        # Use centralized data splits or provided subset
        if X_subset is not None and y_subset is not None:
            # Legacy path for backward compatibility
            logger.warning("Using provided subset instead of centralized splits - this may cause data leakage")
            from sklearn.model_selection import train_test_split
            X_array = X_subset.cpu().numpy()
            y_array = y_subset.cpu().numpy()
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_array, y_array, test_size=0.2, random_state=42, stratify=y_array
            )
        else:
            # Use centralized data splits
            try:
                splits = get_current_splits()
                logger.info("Using centralized data splits for BO objective")
                
                # Use training and validation splits from centralized splitter
                X_train, y_train = splits.X_train, splits.y_train
                X_val, y_val = splits.X_val, splits.y_val
                
            except ValueError:
                # Fallback: use BO subset if centralized splits not available
                logger.warning("Centralized splits not available, using BO subset")
                X_bo, y_bo = get_bo_subset()
                from sklearn.model_selection import train_test_split
                X_train, X_val, y_train, y_val = train_test_split(
                    X_bo, y_bo, test_size=0.2, random_state=42, stratify=y_bo
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
            # Convert numpy types to Python types using centralized method
            processed_hparams = self.executor._convert_numpy_types(hparams)
            
            # Validate and fix hyperparameter incompatibilities
            processed_hparams = self.executor._validate_hyperparameters(
                processed_hparams, 
                model_name=self.training_data.get('model_name', 'Unknown')
            )
            
            # Execute training
            model, metrics = self.executor.execute_training_function(
                self.training_data,
                self.X_train, self.y_train,
                self.X_val, self.y_val,
                device=self.device,
                **processed_hparams
            )
            
            # Return objective value (F1 score or accuracy)
            # Try val_f1 (list), then macro_f1, then val_acc (list), then val_accuracy, default 0.0
            val_f1 = metrics.get('val_f1', [])
            val_acc = metrics.get('val_acc', [])
            
            if val_f1 and isinstance(val_f1, list) and len(val_f1) > 0:
                # Use the last (best) F1 score from training
                objective_value = val_f1[-1]
            elif 'macro_f1' in metrics:
                objective_value = metrics['macro_f1']
            elif val_acc and isinstance(val_acc, list) and len(val_acc) > 0:
                # Use the last (best) accuracy from training
                objective_value = val_acc[-1]
            elif 'val_accuracy' in metrics:
                objective_value = metrics['val_accuracy']
            else:
                objective_value = 0.0
            
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