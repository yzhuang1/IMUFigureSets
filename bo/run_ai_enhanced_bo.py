"""
AI-Enhanced Bayesian Optimization Runner
Integrates universal data converter and AI model selector
"""

import logging
import numpy as np
import torch
from typing import Dict, Any, List, Tuple
import json
import time

from adapters.universal_converter import convert_to_torch_dataset
from models.ai_model_selector import select_model_for_data
from bo.ai_enhanced_objective import create_ai_enhanced_objective
from bo.run_bo import suggest, observe, reset_optimizer, get_optimizer_info

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIEnhancedBO:
    """AI-enhanced Bayesian optimization class"""
    
    def __init__(self, data, labels=None, device="cpu", n_trials=None, **kwargs):
        """
        Initialize AI-enhanced BO
        
        Args:
            data: Input data
            labels: Label data
            device: Device
            n_trials: Number of optimization trials (uses config default if None)
            **kwargs: Other parameters
        """
        from config import config
        self.data = data
        self.labels = labels
        self.device = device
        self.n_trials = n_trials or config.max_bo_trials
        self.kwargs = kwargs
        
        logger.info(f"AIEnhancedBO initialized with n_trials={self.n_trials}")
        
        # Reset BO optimizer for clean start
        reset_optimizer()
        logger.info("Reset Bayesian Optimizer for new optimization run")
        
        # Preprocess data
        self._preprocess_data()
        
        # Get AI recommendation
        self._get_ai_recommendation()
        
        # Create objective function
        self.objective = create_ai_enhanced_objective(
            data, labels, device, **kwargs
        )
        
        # Store results
        self.results = []
        self.best_value = -np.inf
        self.best_params = None
        self.best_metrics = None
    
    def _preprocess_data(self):
        """Preprocess data"""
        logger.info("Preprocessing data...")
        
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
        logger.info(f"Confidence: {self.recommendation.confidence:.2f}")
    
    def run_optimization(self) -> Dict[str, Any]:
        """
        Run Bayesian optimization
        
        Returns:
            Dict: Optimization results
        """
        logger.info(f"Starting AI-enhanced Bayesian optimization, {self.n_trials} trials")
        
        start_time = time.time()
        
        for trial in range(self.n_trials):
            logger.info(f"Trial {trial + 1}/{self.n_trials}")
            
            # Suggest hyperparameters
            hparams = suggest()
            
            # Evaluate objective function
            try:
                value, metrics = self.objective(hparams)
                
                # Record results
                result = {
                    'trial': trial + 1,
                    'hparams': hparams,
                    'value': value,
                    'metrics': metrics,
                    'timestamp': time.time()
                }
                self.results.append(result)
                
                # Update best results
                if value > self.best_value:
                    self.best_value = value
                    self.best_params = hparams.copy()
                    self.best_metrics = metrics.copy()
                    logger.info(f"New best result: {value:.4f}")
                
                # Observe results (for BO algorithm)
                observe(hparams, value)
                
                logger.info(f"Trial {trial + 1} completed: value={value:.4f}, model={metrics.get('model_name', 'Unknown')}")
                
            except Exception as e:
                logger.error(f"Trial {trial + 1} evaluation failed: {e}")
                continue
        
        end_time = time.time()
        
        # Get BO convergence information
        convergence_info = get_optimizer_info()
        
        # Summarize results
        summary = {
            'best_value': self.best_value,
            'best_params': self.best_params,
            'best_metrics': self.best_metrics,
            'total_trials': len(self.results),
            'successful_trials': len([r for r in self.results if 'error' not in r.get('metrics', {})]),
            'total_time': end_time - start_time,
            'bo_convergence': convergence_info,
            'data_profile': self.data_profile.to_dict(),
            'ai_recommendation': {
                'model_name': self.recommendation.model_name,
                'model_type': self.recommendation.model_type,
                'architecture': self.recommendation.architecture,
                'reasoning': self.recommendation.reasoning,
                'confidence': self.recommendation.confidence
            },
            'all_results': self.results
        }
        
        logger.info("Optimization completed!")
        logger.info(f"Best value: {self.best_value:.4f}")
        logger.info(f"Best parameters: {self.best_params}")
        logger.info(f"Total time: {end_time - start_time:.2f} seconds")
        logger.info(f"BO convergence status: {convergence_info.get('status', 'unknown')}")
        if 'recent_avg_improvement' in convergence_info:
            logger.info(f"Recent average improvement: {convergence_info['recent_avg_improvement']:.6f}")
        
        return summary
    
    def save_results(self, filename: str):
        """Save results to file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"Results saved to: {filename}")

def run_ai_enhanced_bo(data, labels=None, device="cpu", n_trials=None, **kwargs):
    """
    Run AI-enhanced Bayesian optimization
    
    Args:
        data: Input data
        labels: Label data
        device: Device
        n_trials: Number of optimization trials (uses config default if None)
        **kwargs: Other parameters
    
    Returns:
        Dict: Optimization results
    """
    bo = AIEnhancedBO(data, labels, device, n_trials, **kwargs)
    return bo.run_optimization()

def demo_ai_enhanced_bo():
    """Demonstrate AI-enhanced Bayesian optimization"""
    
    print("=" * 80)
    print("AI-Enhanced Bayesian Optimization Demo")
    print("=" * 80)
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Check OpenAI API key - required for AI model selection
    from config import config
    if not config.is_openai_configured():
        print("ERROR: OpenAI API key is required for AI-enhanced Bayesian Optimization!")
        print("Please set OPENAI_API_KEY environment variable:")
        print("  export OPENAI_API_KEY='your-api-key'")
        print("  or create .env file with: OPENAI_API_KEY=your-api-key")
        exit(1)
    
    print(f"✓ OpenAI configured (model: {config.openai_model})")
    
    print("\n" + "=" * 60)
    print("Demo 1: Tabular Data Optimization")
    print("=" * 60)
    
    # Tabular data
    X_tabular = np.random.randn(500, 15).astype("float32")
    y_tabular = np.random.choice(["A", "B", "C"], size=500)
    
    result1 = run_ai_enhanced_bo(X_tabular, y_tabular, device=device, n_trials=5)  # Override default for demo
    print(f"Tabular data optimization results:")
    print(f"  Best value: {result1['best_value']:.4f}")
    print(f"  Best parameters: {result1['best_params']}")
    print(f"  AI recommended model: {result1['ai_recommendation']['model_name']}")
    print(f"  Recommendation reason: {result1['ai_recommendation']['reasoning']}")
    
    print("\n" + "=" * 60)
    print("Demo 2: Image Data Optimization")
    print("=" * 60)
    
    # Image data
    X_image = np.random.randn(300, 3, 28, 28).astype("float32")
    y_image = np.random.choice([0, 1], size=300)
    
    result2 = run_ai_enhanced_bo(X_image, y_image, device=device, n_trials=5)
    print(f"Image data optimization results:")
    print(f"  Best value: {result2['best_value']:.4f}")
    print(f"  Best parameters: {result2['best_params']}")
    print(f"  AI recommended model: {result2['ai_recommendation']['model_name']}")
    print(f"  Recommendation reason: {result2['ai_recommendation']['reasoning']}")
    
    print("\n" + "=" * 60)
    print("Demo 3: Sequence Data Optimization")
    print("=" * 60)
    
    # Sequence data
    X_sequence = np.random.randn(400, 30, 8).astype("float32")
    y_sequence = np.random.choice([0, 1, 2], size=400)
    
    result3 = run_ai_enhanced_bo(X_sequence, y_sequence, device=device, n_trials=5)
    print(f"Sequence data optimization results:")
    print(f"  Best value: {result3['best_value']:.4f}")
    print(f"  Best parameters: {result3['best_params']}")
    print(f"  AI recommended model: {result3['ai_recommendation']['model_name']}")
    print(f"  Recommendation reason: {result3['ai_recommendation']['reasoning']}")

if __name__ == "__main__":
    demo_ai_enhanced_bo()
