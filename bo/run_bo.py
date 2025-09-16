"""
Proper Bayesian Optimization implementation using scikit-optimize
Uses Random Forest surrogate model with Expected Improvement acquisition function
- suggest() -> dict of hyperparameters based on surrogate model and acquisition function
- objective(hparams) -> returns scalar or dict of metrics (you can convert multi-objective to scalar)
- observe(hparams, value) -> updates the surrogate model with new observations
"""

import numpy as np
from typing import Dict, Any, List, Tuple
import logging
import time

# Try to import scikit-optimize for proper BO, fallback to random if not available
try:
    from skopt.space import Real, Integer, Categorical
    from skopt import Optimizer
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    logging.warning("scikit-optimize not available, falling back to random search")

logger = logging.getLogger(__name__)



class BayesianOptimizer:
    """Bayesian Optimization implementation with Random Forest surrogate model"""
    
    def __init__(self, search_space=None, gpt_search_space=None, n_initial_points=8, acquisition_func='EI'):
        """
        Initialize Bayesian Optimizer
        
        Args:
            search_space: Search space definition (if None, uses gpt_search_space or default)
            gpt_search_space: GPT-generated search space definition
            n_initial_points: Number of random initial points before using GP
            acquisition_func: Acquisition function ('EI' for Expected Improvement)
        """
        # Priority: explicit search_space > gpt_search_space > default
        if search_space:
            self.search_space = search_space
            logger.info("Using explicitly provided search space")
        elif gpt_search_space:
            self.search_space = self._convert_gpt_search_space(gpt_search_space)
            logger.info("Using GPT-generated search space")
        else:
            # Default generic search space if nothing specified
            self.search_space = [
                Real(1e-5, 1e-1, prior='log-uniform', name='lr'),
                Integer(3, 30, name='epochs'),
                Categorical([8, 16, 32, 64, 128, 256], name='batch_size'),
                Integer(16, 512, name='hidden_size'),
                Real(0.0, 0.7, name='dropout')
            ]
            logger.info("Using default search space")
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
                base_estimator="RF",  # Random Forest
                acq_func="EI",       # Expected Improvement
                n_initial_points=self.n_initial_points,
                random_state=42
            )
            logger.info("Initialized Random Forest Bayesian Optimizer")
        else:
            logger.warning("Using random search fallback (install scikit-optimize for proper BO)")
    
    def _convert_gpt_search_space(self, gpt_search_space: Dict[str, Dict[str, Any]]) -> List:
        """
        Convert GPT-generated search space format to scikit-optimize format
        
        Args:
            gpt_search_space: GPT format like {"lr": {"type": "Real", "low": 1e-5, "high": 1e-1, "prior": "log-uniform"}}
            
        Returns:
            List: scikit-optimize search space dimensions
        """
        search_space = []
        
        for param_name, param_config in gpt_search_space.items():
            param_type = param_config.get("type", "Real")
            
            if param_type == "Real":
                low = param_config.get("low", 0.0)
                high = param_config.get("high", 1.0)
                prior = param_config.get("prior", "uniform")
                search_space.append(Real(low, high, prior=prior, name=param_name))
                
            elif param_type == "Integer":
                low = param_config.get("low", 1)
                high = param_config.get("high", 100)
                search_space.append(Integer(low, high, name=param_name))
                
            elif param_type == "Categorical":
                categories = param_config.get("categories", [])
                if categories:
                    search_space.append(Categorical(categories, name=param_name))
                else:
                    logger.warning(f"No categories specified for {param_name}, skipping")
                    
            else:
                logger.warning(f"Unknown parameter type '{param_type}' for {param_name}, defaulting to Real(0, 1)")
                search_space.append(Real(0.0, 1.0, name=param_name))
        
        logger.info(f"Converted GPT search space: {len(search_space)} parameters")
        return search_space
    
    def _apply_parameter_constraints(self, hparams: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply parameter constraints based on which parameters are present
        Only applies constraints when both required parameters exist
        """
        constrained = hparams.copy()
        
        # Constraint 1: embed_dim must be divisible by num_heads
        if 'embed_dim' in constrained and 'num_heads' in constrained:
            embed_dim = int(constrained['embed_dim'])
            num_heads = int(constrained['num_heads'])
            
            if embed_dim % num_heads != 0:
                # Adjust embed_dim to be divisible by num_heads
                new_embed_dim = ((embed_dim // num_heads) + 1) * num_heads
                logger.info(f"BO constraint applied: embed_dim {embed_dim} -> {new_embed_dim} (divisible by num_heads={num_heads})")
                constrained['embed_dim'] = new_embed_dim
        
        # Constraint 2: d_model must be divisible by num_heads (alternative to embed_dim)
        if 'd_model' in constrained and 'num_heads' in constrained:
            d_model = int(constrained['d_model'])
            num_heads = int(constrained['num_heads'])
            
            if d_model % num_heads != 0:
                new_d_model = ((d_model // num_heads) + 1) * num_heads
                logger.info(f"BO constraint applied: d_model {d_model} -> {new_d_model} (divisible by num_heads={num_heads})")
                constrained['d_model'] = new_d_model
        
        # Constraint 3: hidden_size must be divisible by num_heads (for some attention models)
        if 'hidden_size' in constrained and 'num_heads' in constrained and 'embed_dim' not in constrained and 'd_model' not in constrained:
            # Only apply this if embed_dim and d_model are not present (to avoid double-fixing)
            hidden_size = int(constrained['hidden_size'])
            num_heads = int(constrained['num_heads'])
            
            if hidden_size % num_heads != 0:
                new_hidden_size = ((hidden_size // num_heads) + 1) * num_heads
                logger.info(f"BO constraint applied: hidden_size {hidden_size} -> {new_hidden_size} (divisible by num_heads={num_heads})")
                constrained['hidden_size'] = new_hidden_size
        
        # Log which constraints were checked
        constraint_params = []
        if 'embed_dim' in constrained and 'num_heads' in constrained:
            constraint_params.append('embed_dim/num_heads')
        if 'd_model' in constrained and 'num_heads' in constrained:
            constraint_params.append('d_model/num_heads')
        if 'hidden_size' in constrained and 'num_heads' in constrained and 'embed_dim' not in constrained and 'd_model' not in constrained:
            constraint_params.append('hidden_size/num_heads')
        
        if constraint_params:
            logger.debug(f"Applied constraints for: {', '.join(constraint_params)}")
        else:
            logger.debug("No attention model constraints needed for current parameter set")
        
        return constrained
    
    def suggest(self) -> Dict[str, Any]:
        """
        Suggest next hyperparameters using BO acquisition function
        
        Returns:
            Dict: Suggested hyperparameters
        """
        t0 = time.time()
        
        self.n_calls += 1
        
        if SKOPT_AVAILABLE and self.optimizer:
            # Use proper Bayesian Optimization
            if len(self.y_observed) < self.n_initial_points:
                # Initial random exploration
                suggested = self.optimizer.ask()
                logger.info(f"BO Trial {self.n_calls}: Initial random exploration")
            else:
                # BO with RF surrogate and acquisition function
                suggested = self.optimizer.ask()
                logger.info(f"BO Trial {self.n_calls}: Using RF surrogate + Expected Improvement")
            
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
        
        # Apply constraints to ensure parameter compatibility (only if relevant parameters exist)
        hparams = self._apply_parameter_constraints(hparams)
        
        # Store parameter vector for observation later
        self._last_suggested = [hparams[name] for name in self.param_names]
        
        suggest_time = time.time() - t0
        logger.info(f"[PROFILE] suggest() took {suggest_time:.3f}s")
        
        return hparams
    
    def observe(self, hparams: Dict[str, Any], value: float):
        """
        Observe the result of an evaluation and update the surrogate model
        
        Args:
            hparams: Hyperparameters that were evaluated
            value: Objective function value (higher is better)
        """
        t0 = time.time()
        
        # Convert hparams to parameter vector
        param_vector = [hparams[name] for name in self.param_names]
        
        # Store observation
        self.X_observed.append(param_vector)
        self.y_observed.append(value)
        
        if SKOPT_AVAILABLE and self.optimizer:
            # Update RF surrogate model
            self.optimizer.tell(param_vector, -value)  # Minimize negative (since we maximize)
            logger.info(f"Updated RF surrogate model with observation: {value:.4f}")
        
        observe_time = time.time() - t0
        logger.info(f"[PROFILE] observe()->tell took {observe_time:.3f}s")
        
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
    
    def validate_params(self, hparams: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and convert parameter types
        
        Args:
            hparams: Raw hyperparameters from BO
            
        Returns:
            Dict: Validated parameters with correct types
        """
        validated = hparams.copy()
        
        # Convert common integer parameters
        int_params = ['epochs', 'batch_size', 'hidden_size', 'num_layers', 'num_filters', 
                     'pool_size', 'd_model', 'nhead', 'cnn_filters', 'lstm_hidden', 'hidden_units']
        
        for param in int_params:
            if param in validated:
                validated[param] = int(validated[param])
                
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

def suggest() -> Dict[str, Any]:
    """
    Suggest next hyperparameters using Bayesian Optimization
    
    Returns:
        Dict: Suggested and validated hyperparameters
    """
    t0 = time.time()
    hparams = _global_optimizer.suggest()
    
    # Validate parameters
    hparams = _global_optimizer.validate_params(hparams)
    
    suggest_time = time.time() - t0
    logger.info(f"[PROFILE] global suggest() took {suggest_time:.3f}s")
    
    return hparams

def observe(hparams: Dict[str, Any], value: float):
    """
    Observe evaluation result and update surrogate model
    
    Args:
        hparams: Hyperparameters that were evaluated  
        value: Objective function value (higher is better)
    """
    t0 = time.time()
    _global_optimizer.observe(hparams, value)
    observe_time = time.time() - t0
    logger.info(f"[PROFILE] global observe() took {observe_time:.3f}s")

def reset_optimizer(gpt_search_space: Dict[str, Dict[str, Any]] = None):
    """
    Reset the global optimizer (useful for new optimization runs)
    
    Args:
        gpt_search_space: GPT-generated search space definition
    """
    global _global_optimizer
    _global_optimizer = BayesianOptimizer(gpt_search_space=gpt_search_space)
    if gpt_search_space:
        logger.info("Reset Bayesian Optimizer with GPT-generated search space")
    else:
        logger.info("Reset Bayesian Optimizer with default search space")

def get_optimizer_info() -> Dict[str, Any]:
    """Get information about the current optimization state"""
    return _global_optimizer.get_convergence_info()

def validate_hyperparams(hparams: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate hyperparameters
    
    Args:
        hparams: Raw hyperparameters
        
    Returns:
        Dict: Validated hyperparameters
    """
    return _global_optimizer.validate_params(hparams)

def create_optimizer_from_code_recommendation(code_recommendation) -> BayesianOptimizer:
    """
    Create a new BayesianOptimizer from a CodeRecommendation with GPT-suggested search space
    
    Args:
        code_recommendation: CodeRecommendation object with bo_search_space
        
    Returns:
        BayesianOptimizer: New optimizer instance
    """
    return BayesianOptimizer(gpt_search_space=code_recommendation.bo_search_space)

def reset_optimizer_from_code_recommendation(code_recommendation):
    """
    Reset global optimizer from a CodeRecommendation with GPT-suggested search space
    
    Args:
        code_recommendation: CodeRecommendation object with bo_search_space
    """
    global _global_optimizer
    _global_optimizer = create_optimizer_from_code_recommendation(code_recommendation)
    logger.info(f"Reset Bayesian Optimizer with GPT-generated search space from {code_recommendation.model_name}")
