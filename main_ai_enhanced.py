"""
AI-Enhanced Main Process with Iterative Model Selection
Integrates ChatGPT-powered evaluation for automatic model re-selection
"""

import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Dict, Any, Tuple

from adapters.universal_converter import convert_to_torch_dataset
from models.dynamic_model_registry import build_model_from_recommendation
from evaluation.ai_enhanced_evaluate import IterativeModelSelector, AIEnhancedEvaluator
from train import train_one_model

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_with_iterative_selection(data, labels=None, device="cpu", epochs=5, max_model_attempts=None, **kwargs):
    """
    Train model with AI-enhanced iterative model selection
    
    Args:
        data: Input data
        labels: Label data
        device: Device for training
        epochs: Number of training epochs
        max_model_attempts: Maximum number of model architectures to try (uses config default if None)
        **kwargs: Additional parameters
    
    Returns:
        Dict: Training results with final model and evaluation
    """
    logger.info("Starting AI-enhanced training with iterative model selection")
    
    # Convert data and get profile
    dataset, collate_fn, data_profile = convert_to_torch_dataset(data, labels, **kwargs)
    
    logger.info(f"Data profile: {data_profile}")
    
    # Create data loader
    batch_size = kwargs.get('batch_size', 64)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    # Define training function
    def train_model(recommendation, **train_kwargs):
        """Train a model from recommendation"""
        logger.info(f"Training model: {recommendation.model_name}")
        
        # Determine input shape
        if data_profile.is_sequence:
            input_shape = (data_profile.feature_count,)
        elif data_profile.is_image:
            if data_profile.channels and data_profile.height and data_profile.width:
                input_shape = (data_profile.channels, data_profile.height, data_profile.width)
            else:
                input_shape = (3, 32, 32)  # Default
        else:
            input_shape = (data_profile.feature_count,)
        
        num_classes = data_profile.label_count if data_profile.has_labels else 2
        
        # Build model
        model = build_model_from_recommendation(recommendation, input_shape, num_classes)
        model.to(device)
        
        # Train model
        trained_model = train_one_model(model, loader, device=device, epochs=epochs)
        
        return trained_model
    
    # Define evaluation function
    def evaluate_model_with_ai(evaluator, model, **eval_kwargs):
        """Evaluate model using AI analysis"""
        return evaluator.evaluate_with_ai_analysis(model, loader, device)
    
    # Create iterative model selector (uses config default if max_model_attempts is None)
    selector = IterativeModelSelector(
        data_profile=data_profile.to_dict(),
        max_iterations=max_model_attempts
    )
    
    # Find best model
    best_model, best_analysis = selector.find_best_model(
        train_func=train_model,
        evaluate_func=evaluate_model_with_ai
    )
    
    # Prepare results
    results = {
        'model': best_model,
        'analysis': best_analysis,
        'data_profile': data_profile,
        'attempt_summary': selector.get_attempt_summary(),
        'final_metrics': best_analysis.metrics,
        'dataset': dataset,
        'data_loader': loader,
        'collate_fn': collate_fn
    }
    
    logger.info("AI-enhanced training completed!")
    logger.info(f"Final model achieved: {best_analysis.metrics}")
    logger.info(f"Total model attempts: {results['attempt_summary']['total_attempts']}")
    
    return results

def process_data_with_ai_enhanced_evaluation(data, labels=None, **kwargs):
    """
    Process data with AI-enhanced evaluation and iterative model selection
    
    Args:
        data: Input data (any format)
        labels: Label data
        **kwargs: Other parameters
    
    Returns:
        dict: Dictionary containing best model, analysis, and attempt history
    """
    logger.info("Starting AI-enhanced data processing...")
    
    # Set device
    device = kwargs.get('device', "cuda" if torch.cuda.is_available() else "cpu")
    
    # Train with iterative selection
    result = train_with_iterative_selection(data, labels, device=device, **kwargs)
    
    return result

def demo_ai_enhanced_evaluation():
    """Demonstrate AI-enhanced evaluation with iterative model selection"""
    
    print("=" * 80)
    print("AI-Enhanced Evaluation Demo")
    print("=" * 80)
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Check OpenAI API key - fail fast if not configured
    from config import config
    if not config.is_openai_configured():
        print("ERROR: OpenAI API key is required but not configured!")
        print("Please set OPENAI_API_KEY environment variable:")
        print("  export OPENAI_API_KEY='your-api-key'")
        print("  or create .env file with: OPENAI_API_KEY=your-api-key")
        exit(1)
    
    print(f"✓ OpenAI configured (model: {config.openai_model})")
    print("\n" + "=" * 60)
    print("Demo 1: Tabular Data with AI-Enhanced Evaluation")
    print("=" * 60)
    
    # Create challenging tabular data
    np.random.seed(42)
    X_tabular = np.random.randn(800, 25).astype("float32")
    # Make it somewhat challenging - add some noise and correlation
    X_tabular[:, :5] = X_tabular[:, :5] * 2 + np.random.randn(800, 5) * 0.5
    y_tabular = ((X_tabular[:, 0] + X_tabular[:, 1] - X_tabular[:, 2]) > 0).astype(int)
    # Add some label noise to make it more challenging
    noise_indices = np.random.choice(len(y_tabular), size=int(0.1 * len(y_tabular)), replace=False)
    y_tabular[noise_indices] = 1 - y_tabular[noise_indices]
    
    result1 = process_data_with_ai_enhanced_evaluation(
        X_tabular, y_tabular, 
        device=device, 
        epochs=8
        # max_model_attempts will use config default (3)
    )
    
    print(f"Final Results:")
    print(f"  Best model metrics: {result1['final_metrics']}")
    print(f"  AI decision: {result1['analysis'].decision}")
    print(f"  AI analysis: {result1['analysis'].analysis}")
    print(f"  Total model attempts: {result1['attempt_summary']['total_attempts']}")
    
    if result1['attempt_summary']['total_attempts'] > 1:
        print(f"  Model selection history:")
        for attempt in result1['attempt_summary']['attempts']:
            print(f"    Attempt {attempt['iteration']}: {attempt['model_name']} -> "
                  f"F1: {attempt['metrics'].get('macro_f1', 0):.3f} -> {attempt['decision']}")
    
    print("\n" + "=" * 60)
    print("Demo 2: Image Data with AI-Enhanced Evaluation")
    print("=" * 60)
    
    # Create challenging image data
    X_image = np.random.randn(400, 3, 28, 28).astype("float32")
    # Add some structure to make it more realistic
    for i in range(len(X_image)):
        # Add some patterns
        X_image[i, :, 10:18, 10:18] += np.random.randn(3, 8, 8) * 0.5
    
    # Create labels based on some image characteristics
    y_image = ((X_image.mean(axis=(2, 3))[:, 0] > X_image.mean(axis=(2, 3))[:, 1])).astype(int)
    
    result2 = process_data_with_ai_enhanced_evaluation(
        X_image, y_image,
        device=device,
        epochs=6
        # max_model_attempts will use config default (3)
    )
    
    print(f"Final Results:")
    print(f"  Best model metrics: {result2['final_metrics']}")
    print(f"  AI decision: {result2['analysis'].decision}")
    print(f"  AI analysis: {result2['analysis'].analysis}")
    print(f"  Total model attempts: {result2['attempt_summary']['total_attempts']}")
    
    if result2['attempt_summary']['total_attempts'] > 1:
        print(f"  Model selection history:")
        for attempt in result2['attempt_summary']['attempts']:
            print(f"    Attempt {attempt['iteration']}: {attempt['model_name']} -> "
                  f"F1: {attempt['metrics'].get('macro_f1', 0):.3f} -> {attempt['decision']}")

if __name__ == "__main__":
    demo_ai_enhanced_evaluation()