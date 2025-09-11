"""
Proper Bayesian Optimization implementation using scikit-optimize
Uses Gaussian Process surrogate model with Expected Improvement acquisition function
- suggest() -> dict of hyperparameters based on surrogate model and acquisition function
- objective(hparams) -> returns scalar or dict of metrics (you can convert multi-objective to scalar)
- observe(hparams, value) -> updates the surrogate model with new observations
"""

import numpy as np
from typing import Dict, Any, List, Tuple
import logging

# Try to import scikit-optimize for proper BO, fallback to random if not available
try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.acquisition import gaussian_ei
    from skopt.utils import use_named_args
    from skopt import Optimizer
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    logging.warning("scikit-optimize not available, falling back to random search")

logger = logging.getLogger(__name__)

# Template-aware search spaces
def get_search_space_for_template(template_name: str):
    """Get search space configuration for specific template"""
    
    # Common training parameters
    common_params = [
        Real(1e-4, 3e-3, prior='log-uniform', name='lr'),
        Integer(3, 15, name='epochs'),
        Categorical([16, 32, 64, 128], name='batch_size'),
    ]
    
    # Template-specific architecture parameters
    if template_name == "LSTM":
        return common_params + [
            Integer(32, 256, name='hidden_size'),
            Integer(1, 3, name='num_layers'),
            Real(0.1, 0.5, name='dropout'),
            Categorical([True, False], name='bidirectional')
        ]
    
    elif template_name == "GRU":
        return common_params + [
            Integer(32, 256, name='hidden_size'),
            Integer(1, 3, name='num_layers'),
            Real(0.1, 0.5, name='dropout'),
            Categorical([True, False], name='bidirectional')
        ]
    
    elif template_name == "CNN1D":
        return common_params + [
            Integer(32, 128, name='num_filters'),
            Real(0.1, 0.4, name='dropout'),
            Integer(2, 4, name='pool_size')
        ]
    
    elif template_name == "Transformer":
        return common_params + [
            Categorical([64, 128, 256], name='d_model'),
            Categorical([4, 8, 16], name='nhead'),
            Integer(1, 4, name='num_layers'),
            Real(0.1, 0.3, name='dropout')
        ]
    
    elif template_name == "MLP":
        return common_params + [
            Real(0.1, 0.5, name='dropout'),
            Categorical(['relu', 'tanh'], name='activation')
        ]
    
    elif template_name == "HybridCNNLSTM":
        return common_params + [
            Integer(32, 128, name='cnn_filters'),
            Integer(64, 256, name='lstm_hidden'),
            Real(0.1, 0.4, name='dropout')
        ]
    
    else:
        # Fallback to generic search space
        return common_params + [
            Integer(32, 256, name='hidden_size'),
            Real(0.1, 0.5, name='dropout')
        ]

# Legacy search space for backward compatibility
_search_space_skopt = [
    Real(1e-4, 3e-3, prior='log-uniform', name='lr'),
    Integer(3, 10, name='epochs'), 
    Categorical([32, 64, 128, 256], name='hidden'),
]

_search_space_bounds = {
    "lr": (1e-4, 3e-3),
    "epochs": (3, 10), 
    "hidden": [32, 64, 128, 256],
}

class BayesianOptimizer:
    """Bayesian Optimization implementation with Gaussian Process surrogate model"""
    
    def __init__(self, search_space=None, template_name=None, n_initial_points=8, acquisition_func='EI'):
        """
        Initialize Bayesian Optimizer
        
        Args:
            search_space: Search space definition (if None, uses template_name)
            template_name: Template name to get search space automatically
            n_initial_points: Number of random initial points before using GP
            acquisition_func: Acquisition function ('EI' for Expected Improvement)
        """
        # Use template-aware search space if template_name provided
        if search_space is None and template_name:
            self.search_space = get_search_space_for_template(template_name)
            logger.info(f"Using template-aware search space for {template_name}")
        else:
            self.search_space = search_space or _search_space_skopt
            
        self.template_name = template_name
        self.n_initial_points = n_initial_points
        self.acquisition_func = acquisition_func
        
        # History storage
        self.X_observed: List[List[float]] = []  # Parameter vectors
        self.y_observed: List[float] = []         # Objective values
        self.param_names = [dim.name for dim in self.search_space]
        
        # BO state
        self.n_calls = 0
        self.optimizer = None
        
        if SKOPT_AVAILABLE:
            self.optimizer = Optimizer(
                dimensions=self.search_space,
                base_estimator="GP",  # Gaussian Process
                acq_func="EI",       # Expected Improvement
                n_initial_points=self.n_initial_points,
                random_state=42
            )
            logger.info("Initialized Gaussian Process Bayesian Optimizer")
        else:
            logger.warning("Using random search fallback (install scikit-optimize for proper BO)")
    
    def suggest(self) -> Dict[str, Any]:
        """
        Suggest next hyperparameters using BO acquisition function
        
        Returns:
            Dict: Suggested hyperparameters
        """
        self.n_calls += 1
        
        if SKOPT_AVAILABLE and self.optimizer:
            # Use proper Bayesian Optimization
            if len(self.y_observed) < self.n_initial_points:
                # Initial random exploration
                suggested = self.optimizer.ask()
                logger.info(f"BO Trial {self.n_calls}: Initial random exploration")
            else:
                # BO with GP surrogate and acquisition function
                suggested = self.optimizer.ask()
                logger.info(f"BO Trial {self.n_calls}: Using GP surrogate + Expected Improvement")
            
            # Convert to parameter dict
            hparams = dict(zip(self.param_names, suggested))
            
        else:
            # Fallback to random search
            import random
            hparams = {
                "lr": 10 ** random.uniform(np.log10(1e-4), np.log10(3e-3)),
                "epochs": random.randint(3, 10),
                "hidden": random.choice([32, 64, 128, 256]),
            }
            logger.info(f"BO Trial {self.n_calls}: Random search fallback")
        
        # Store parameter vector for observation later
        self._last_suggested = [hparams[name] for name in self.param_names]
        
        return hparams
    
    def observe(self, hparams: Dict[str, Any], value: float):
        """
        Observe the result of an evaluation and update the surrogate model
        
        Args:
            hparams: Hyperparameters that were evaluated
            value: Objective function value (higher is better)
        """
        # Convert hparams to parameter vector
        param_vector = [hparams[name] for name in self.param_names]
        
        # Store observation
        self.X_observed.append(param_vector)
        self.y_observed.append(value)
        
        if SKOPT_AVAILABLE and self.optimizer:
            # Update GP surrogate model
            self.optimizer.tell(param_vector, -value)  # Minimize negative (since we maximize)
            logger.info(f"Updated GP surrogate model with observation: {value:.4f}")
        
        logger.info(f"Recorded observation #{len(self.y_observed)}: "
                   f"hparams={hparams}, value={value:.4f}")
    
    def get_best_params(self) -> Tuple[Dict[str, Any], float]:
        """
        Get the best parameters observed so far
        
        Returns:
            Tuple[Dict, float]: Best parameters and best value
        """
        if not self.y_observed:
            return {}, -np.inf
        
        best_idx = np.argmax(self.y_observed)
        best_vector = self.X_observed[best_idx]
        best_value = self.y_observed[best_idx]
        
        best_params = dict(zip(self.param_names, best_vector))
        return best_params, best_value
    
    def validate_params(self, hparams: Dict[str, Any], template_name: str = None) -> Dict[str, Any]:
        """
        Validate and map parameters for specific template
        
        Args:
            hparams: Raw hyperparameters from BO
            template_name: Template name for validation
            
        Returns:
            Dict: Validated and mapped parameters
        """
        validated = hparams.copy()
        template = template_name or self.template_name
        
        # Parameter mapping and validation
        if template in ["LSTM", "GRU"]:
            # Ensure integer types for specific parameters
            if 'hidden_size' in validated:
                validated['hidden_size'] = int(validated['hidden_size'])
            if 'num_layers' in validated:
                validated['num_layers'] = int(validated['num_layers'])
            if 'epochs' in validated:
                validated['epochs'] = int(validated['epochs'])
            if 'batch_size' in validated:
                validated['batch_size'] = int(validated['batch_size'])
                
        elif template == "CNN1D":
            if 'num_filters' in validated:
                validated['num_filters'] = int(validated['num_filters'])
            if 'pool_size' in validated:
                validated['pool_size'] = int(validated['pool_size'])
            if 'epochs' in validated:
                validated['epochs'] = int(validated['epochs'])
            if 'batch_size' in validated:
                validated['batch_size'] = int(validated['batch_size'])
                
        elif template == "Transformer":
            if 'd_model' in validated:
                validated['d_model'] = int(validated['d_model'])
            if 'nhead' in validated:
                validated['nhead'] = int(validated['nhead'])
            if 'num_layers' in validated:
                validated['num_layers'] = int(validated['num_layers'])
            if 'epochs' in validated:
                validated['epochs'] = int(validated['epochs'])
            if 'batch_size' in validated:
                validated['batch_size'] = int(validated['batch_size'])
                
        elif template == "HybridCNNLSTM":
            if 'cnn_filters' in validated:
                validated['cnn_filters'] = int(validated['cnn_filters'])
            if 'lstm_hidden' in validated:
                validated['lstm_hidden'] = int(validated['lstm_hidden'])
            if 'epochs' in validated:
                validated['epochs'] = int(validated['epochs'])
            if 'batch_size' in validated:
                validated['batch_size'] = int(validated['batch_size'])
        
        # Common validations
        if 'epochs' in validated:
            validated['epochs'] = int(validated['epochs'])
        if 'batch_size' in validated:
            validated['batch_size'] = int(validated['batch_size'])
            
        return validated
    
    def get_convergence_info(self) -> Dict[str, Any]:
        """
        Get information about optimization convergence
        
        Returns:
            Dict: Convergence metrics
        """
        if len(self.y_observed) < 2:
            return {"status": "insufficient_data"}
        
        recent_improvements = []
        for i in range(1, len(self.y_observed)):
            current_best = max(self.y_observed[:i+1])
            previous_best = max(self.y_observed[:i])
            improvement = current_best - previous_best
            recent_improvements.append(improvement)
        
        # Check last few improvements
        recent_window = min(5, len(recent_improvements))
        recent_avg_improvement = np.mean(recent_improvements[-recent_window:]) if recent_improvements else 0
        
        return {
            "status": "converging" if recent_avg_improvement < 0.001 else "exploring",
            "total_evaluations": len(self.y_observed),
            "best_value": max(self.y_observed) if self.y_observed else -np.inf,
            "recent_avg_improvement": recent_avg_improvement,
            "improvement_history": recent_improvements[-10:]  # Last 10 improvements
        }

# Global optimizer instance
_global_optimizer = BayesianOptimizer()

def suggest(template_name: str = None) -> Dict[str, Any]:
    """
    Suggest next hyperparameters using Bayesian Optimization
    
    Args:
        template_name: Template name for validation
    
    Returns:
        Dict: Suggested and validated hyperparameters
    """
    hparams = _global_optimizer.suggest()
    
    # Validate parameters if template provided
    if template_name:
        hparams = _global_optimizer.validate_params(hparams, template_name)
    
    return hparams

def observe(hparams: Dict[str, Any], value: float):
    """
    Observe evaluation result and update surrogate model
    
    Args:
        hparams: Hyperparameters that were evaluated  
        value: Objective function value (higher is better)
    """
    _global_optimizer.observe(hparams, value)

def reset_optimizer(template_name: str = None):
    """
    Reset the global optimizer (useful for new optimization runs)
    
    Args:
        template_name: Template name for template-aware search space
    """
    global _global_optimizer
    _global_optimizer = BayesianOptimizer(template_name=template_name)
    logger.info(f"Reset Bayesian Optimizer for template: {template_name}")

def get_optimizer_info() -> Dict[str, Any]:
    """Get information about the current optimization state"""
    return _global_optimizer.get_convergence_info()

def validate_hyperparams(hparams: Dict[str, Any], template_name: str) -> Dict[str, Any]:
    """
    Validate hyperparameters for a specific template
    
    Args:
        hparams: Raw hyperparameters
        template_name: Template name
        
    Returns:
        Dict: Validated hyperparameters
    """
    return _global_optimizer.validate_params(hparams, template_name)
