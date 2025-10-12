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
import sys
import io
from data_splitting import get_bo_subset, get_current_splits
from config import config

logger = logging.getLogger(__name__)

class LoggerWriter:
    """Redirects stdout/stderr to logger"""
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
        self.buffer = []

    def write(self, message):
        if message and message.strip():  # Skip empty lines
            self.logger.log(self.level, message.rstrip())

    def flush(self):
        pass

def calculate_model_storage_size_kb(model: torch.nn.Module,
                                   quantization_bits: int = 32,
                                   quantize_weights: bool = False,
                                   quantize_activations: bool = False) -> float:
    """
    Calculate actual storage size in KB considering quantization.

    Args:
        model: PyTorch model
        quantization_bits: Bits per parameter (8, 16, 32)
        quantize_weights: Whether weights are quantized
        quantize_activations: Whether activations are quantized (affects intermediate storage)

    Returns:
        Storage size in KB
    """
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if quantize_weights:
        # Use specified quantization bits for weights
        bytes_per_param = quantization_bits / 8
    else:
        # Default float32 = 4 bytes per parameter
        bytes_per_param = 4

    # Calculate base model storage
    storage_bytes = total_params * bytes_per_param

    # Add overhead for model structure (buffers, non-parameter data)
    # Typically 5-10% overhead for metadata, buffers, etc.
    overhead_factor = 1.1

    storage_kb = (storage_bytes * overhead_factor) / 1024

    return storage_kb

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
        # JSON loading already handles escape sequences correctly
        # Only process if there are obvious double-escaped sequences
        if isinstance(training_code, str):
            try:
                # Only fix double-escaped sequences if they exist
                if '\\\\n' in training_code:
                    training_code = training_code.replace('\\\\n', '\n')
                if '\\\\t' in training_code:
                    training_code = training_code.replace('\\\\t', '\t')
                if '\\\\r' in training_code:
                    training_code = training_code.replace('\\\\r', '\r')
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

    def _count_model_parameters(self, model) -> int:
        """Count the total number of parameters in a PyTorch model"""
        try:
            if hasattr(model, 'parameters'):
                return sum(p.numel() for p in model.parameters())
            else:
                logger.warning("Model does not have parameters() method")
                return 0
        except Exception as e:
            logger.error(f"Failed to count model parameters: {e}")
            return 0

    
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

            # Keep all data on CPU to save GPU memory
            # AI-generated training functions manage GPU transfer internally via DataLoader
            if X_train.device != torch.device('cpu'):
                X_train = X_train.to('cpu')
            if y_train.device != torch.device('cpu'):
                y_train = y_train.to('cpu')
            # Validation data stays on CPU - will be moved batch-by-batch during evaluation
            if X_val.device != torch.device('cpu'):
                X_val = X_val.to('cpu')
            if y_val.device != torch.device('cpu'):
                y_val = y_val.to('cpu')
            
            # Extract training code
            training_code = training_data['training_code']
            model_name = training_data['model_name']
            
            logger.info(f"Executing training function: {model_name}")
            logger.info(f"Hyperparameters: {dict(hyperparams) if hyperparams else 'None'}")
            
            # Handle JSON escape sequences in training code
            training_code = self._process_training_code(training_code)

            # Create namespace for code execution with common imports
            namespace = {
                'torch': torch,
                'nn': torch.nn,
                'optim': torch.optim,
                'F': torch.nn.functional,
                'np': np,
                'Dict': Dict,
                'Any': Any,
                'Tuple': Tuple,
                'List': List,
                'Optional': Optional,
            }

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

            # Redirect stdout to logger to capture epoch messages
            old_stdout = sys.stdout
            sys.stdout = LoggerWriter(logger, logging.INFO)

            try:
                model, metrics = train_model(
                    X_train, y_train, X_val, y_val,
                    device=device,
                    **final_hyperparams
                )
            finally:
                # Restore original stdout
                sys.stdout = old_stdout
            
            # Add metadata to metrics
            metrics.update({
                'model_name': model_name,
                'training_function_source': 'ai_generated',
                'hyperparameters_used': final_hyperparams
            })
            
            # Validate model storage size considering quantization
            model_param_count = self._count_model_parameters(model)

            # Extract quantization parameters from hyperparameters
            quantization_bits = final_hyperparams.get('quantization_bits', 32)
            quantize_weights = final_hyperparams.get('quantize_weights', False)
            quantize_activations = final_hyperparams.get('quantize_activations', False)

            # Calculate actual storage size in KB
            storage_size_kb = calculate_model_storage_size_kb(
                model, quantization_bits, quantize_weights, quantize_activations
            )

            logger.info(f"Model: {model_param_count:,} parameters, {storage_size_kb:.1f}KB storage")

            # Add model metrics
            if isinstance(metrics, dict):
                metrics['model_parameter_count'] = model_param_count
                metrics['model_storage_size_kb'] = storage_size_kb
                metrics['model_size_validation'] = 'PASS' if storage_size_kb <= 256 else 'FAIL'
                if storage_size_kb > 256:
                    logger.warning(f"Model storage {storage_size_kb:.1f}KB exceeds 256KB limit!")

            logger.info(f"Training completed successfully: {dict(metrics) if metrics else 'None'}")

            return model, metrics
            
        except Exception as e:
            logger.error(f"Training execution failed: {e}")

            # Send complete context to debug GPT immediately
            try:
                from _models.ai_code_generator import ai_code_generator
                corrections = ai_code_generator._debug_json_with_gpt(
                    training_data.get('training_code', ''),
                    str(e),
                    training_data.get('bo_config', {})
                )
                if corrections and corrections != "{}":
                    logger.info(f"GPT suggested corrections: {corrections}")

                    # Check if GPT identified this as a system issue that should stop the pipeline
                    try:
                        import json
                        correction_data = json.loads(corrections)
                        if "system_issue" in correction_data and correction_data["system_issue"] == "STOP_PIPELINE":
                            logger.error("GPT identified this as a system/environment issue that cannot be fixed by code or hyperparameter changes")
                            logger.error("Stopping pipeline as requested by GPT analysis")
                            raise SystemExit("Pipeline stopped: GPT identified unfixable system/environment issue")
                    except json.JSONDecodeError:
                        logger.warning("Could not parse GPT corrections JSON for system issue check")
                    except SystemExit:
                        raise  # Re-raise SystemExit to stop the pipeline
                    except Exception as check_e:
                        logger.warning(f"Error checking for system issue in corrections: {check_e}")

                    # Store corrections for BO to use
                    try:
                        from error_monitor import _global_terminator
                        if _global_terminator:
                            _global_terminator.last_json_corrections = corrections
                            logger.info("Stored corrections for BO process")
                    except Exception as store_e:
                        logger.warning(f"Could not store corrections: {store_e}")
                else:
                    logger.info("GPT could not provide corrections")
            except Exception as debug_e:
                logger.warning(f"Debug GPT call failed: {debug_e}")

            raise

    
    
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
            # Use BO subset for efficient optimization (respects config.bo_sample_num)
            try:
                X_bo, y_bo = get_bo_subset()
                logger.info(f"Using BO subset for optimization: {len(X_bo)} samples (bo_sample_num={config.bo_sample_num})")
                from sklearn.model_selection import train_test_split
                X_train, X_val, y_train, y_val = train_test_split(
                    X_bo, y_bo, test_size=0.2, random_state=42, stratify=y_bo
                )
                logger.info(f"BO splits - Train: {len(X_train)}, Val: {len(X_val)}")

            except (ValueError, Exception) as e:
                # Fallback: use full centralized splits if BO subset fails
                logger.warning(f"BO subset failed ({e}), falling back to full centralized splits")
                splits = get_current_splits()
                X_train, y_train = splits.X_train, splits.y_train
                X_val, y_val = splits.X_val, splits.y_val
        
        # Keep all data on CPU to save GPU memory, move to GPU batch by batch during training
        self.X_train = torch.tensor(X_train, dtype=torch.float32, device='cpu')
        self.y_train = torch.tensor(y_train, dtype=torch.long, device='cpu')
        self.X_val = torch.tensor(X_val, dtype=torch.float32, device='cpu')
        self.y_val = torch.tensor(y_val, dtype=torch.long, device='cpu')

    def _calculate_size_penalty(self, storage_size_kb: float) -> float:
        """Calculate penalty for model storage size to encourage compression"""
        target_storage_kb = 256.0  # 256KB storage limit

        if storage_size_kb <= target_storage_kb:
            # No penalty for models within size limit
            return 0.0
        else:
            # Heavy penalty for oversized models: penalty increases exponentially
            excess_ratio = (storage_size_kb - target_storage_kb) / target_storage_kb
            penalty = min(excess_ratio * 0.5, 0.8)  # Cap penalty at 0.8 to avoid negative objectives
            return penalty

    def __call__(self, hparams: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Execute training function with given hyperparameters"""
        t0 = time.time()
        model = None

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

            # Return objective value with model size penalty
            # Try val_f1 (list), then macro_f1, then val_acc (list), then val_accuracy, default 0.0
            val_f1 = metrics.get('val_f1', [])
            val_acc = metrics.get('val_acc', [])

            if val_f1 and isinstance(val_f1, list) and len(val_f1) > 0:
                # Use the last (best) F1 score from training
                base_objective = val_f1[-1]
            elif 'macro_f1' in metrics:
                base_objective = metrics['macro_f1']
            elif val_acc and isinstance(val_acc, list) and len(val_acc) > 0:
                # Use the last (best) accuracy from training
                base_objective = val_acc[-1]
            elif 'val_accuracy' in metrics:
                base_objective = metrics['val_accuracy']
            else:
                base_objective = 0.0

            # Apply model storage size penalty for compression-aware optimization
            storage_size_kb = metrics.get('model_storage_size_kb', 0.0)
            size_penalty = self._calculate_size_penalty(storage_size_kb)

            # Final objective = performance - size_penalty
            objective_value = base_objective - size_penalty

            model_param_count = metrics.get('model_parameter_count', 0)
            logger.info(f"BO Objective: base={base_objective:.4f}, size_penalty={size_penalty:.4f}, final={objective_value:.4f}")
            logger.info(f"Model: {model_param_count:,} parameters, {storage_size_kb:.1f}KB ({'PASS' if storage_size_kb <= 256 else 'FAIL'} 256KB limit)")

            objective_time = time.time() - t0
            logger.info(f"[PROFILE] objective(train+eval) took {objective_time:.3f}s")

            return float(objective_value), metrics

        except Exception as e:
            logger.error(f"BO training objective failed: {e}")
            objective_time = time.time() - t0
            logger.info(f"[PROFILE] objective(train+eval) FAILED took {objective_time:.3f}s")
            return 0.0, {"error": str(e)}

        finally:
            # Explicit cleanup to prevent memory leaks and state corruption across BO trials
            if model is not None:
                del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.debug("Cleaned up model and GPU cache after BO trial")

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