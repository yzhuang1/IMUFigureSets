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
from models.ai_template_selector import select_template_for_data
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
        
        # Reset BO optimizer for clean start (will be done after AI recommendation)
        logger.info("Will reset Bayesian Optimizer after AI template selection")
        
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
        """Get AI template recommendation"""
        logger.info("Getting AI template recommendation...")
        
        # We need input_shape and num_classes for template selection
        if self.data_profile.is_sequence:
            input_shape = (self.data_profile.feature_count,)
        elif self.data_profile.is_image:
            if self.data_profile.channels and self.data_profile.height and self.data_profile.width:
                input_shape = (self.data_profile.channels, self.data_profile.height, self.data_profile.width)
            else:
                input_shape = (3, 32, 32)  # Default
        else:
            input_shape = (self.data_profile.feature_count,)
        
        num_classes = self.data_profile.label_count if self.data_profile.has_labels else 2
        
        self.template_rec = select_template_for_data(
            self.data_profile.to_dict(),
            input_shape,
            num_classes
        )
        
        logger.info(f"AI selected template: {self.template_rec.template_name} -> {self.template_rec.model_name}")
        logger.info(f"Selection reason: {self.template_rec.reasoning}")
        logger.info(f"Confidence: {self.template_rec.confidence:.2f}")
        
        # Now reset BO optimizer with template-aware search space
        reset_optimizer(self.template_rec.template_name)
        logger.info(f"Reset BO optimizer with template-aware search space for {self.template_rec.template_name}")
    
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
            
            # Suggest hyperparameters (template-aware)
            hparams = suggest(self.template_rec.template_name)
            
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
                'template_name': self.template_rec.template_name,
                'model_name': self.template_rec.model_name,
                'model_type': 'template_based',
                'architecture': f'{self.template_rec.template_name} template',
                'reasoning': self.template_rec.reasoning,
                'confidence': self.template_rec.confidence,
                'config': self.template_rec.config
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

if __name__ == "__main__":
    print("Use run_ai_enhanced_bo() function with your real data from data/ folder")
    print("Example: run_ai_enhanced_bo(your_data, your_labels, device='cpu')")
