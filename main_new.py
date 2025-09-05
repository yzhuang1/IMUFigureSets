"""
New main process file
Integrates universal data converter and AI model selector
"""

import logging
import numpy as np
import torch
from torch.utils.data import DataLoader

from adapters.universal_converter import convert_to_torch_dataset, DataProfile
from models.ai_model_selector import select_model_for_data
from models.dynamic_model_registry import build_model_from_recommendation
from train import train_one_model
from evaluation.evaluate import evaluate_model

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_data_with_ai(data, labels=None, **kwargs):
    """
    Use AI to automatically process data and select models
    
    Args:
        data: Input data (any format)
        labels: Label data
        **kwargs: Other parameters
    
    Returns:
        dict: Dictionary containing dataset, model, recommendation info, etc.
    """
    logger.info("Starting data processing...")
    
    # 1. Convert data to PyTorch format
    logger.info("Step 1: Convert data to PyTorch format")
    dataset, collate_fn, data_profile = convert_to_torch_dataset(
        data, labels, **kwargs
    )
    
    logger.info(f"Data conversion completed: {data_profile}")
    
    # 2. Use AI to select the most suitable model
    logger.info("Step 2: Use AI to select the most suitable model")
    recommendation = select_model_for_data(data_profile.to_dict())
    
    logger.info(f"AI recommended model: {recommendation.model_name}")
    logger.info(f"Recommendation reason: {recommendation.reasoning}")
    logger.info(f"Confidence: {recommendation.confidence:.2f}")
    
    # 3. Build model based on recommendation
    logger.info("Step 3: Build recommended model")
    
    # Determine input shape
    if data_profile.is_sequence:
        input_shape = (data_profile.feature_count,)
    elif data_profile.is_image:
        if data_profile.channels and data_profile.height and data_profile.width:
            input_shape = (data_profile.channels, data_profile.height, data_profile.width)
        else:
            input_shape = (3, 32, 32)  # Default image size
    elif data_profile.is_tabular:
        input_shape = (data_profile.feature_count,)
    else:
        input_shape = (data_profile.feature_count,)
    
    num_classes = data_profile.label_count if data_profile.has_labels else 2
    
    model = build_model_from_recommendation(
        recommendation, input_shape, num_classes
    )
    
    # 4. Prepare data loader
    logger.info("Step 4: Prepare data loader")
    loader = DataLoader(
        dataset, 
        batch_size=kwargs.get('batch_size', 64), 
        shuffle=True, 
        collate_fn=collate_fn
    )
    
    return {
        'dataset': dataset,
        'data_loader': loader,
        'model': model,
        'data_profile': data_profile,
        'recommendation': recommendation,
        'collate_fn': collate_fn
    }

def train_and_evaluate(data, labels=None, device="cpu", epochs=5, **kwargs):
    """
    Train and evaluate model
    
    Args:
        data: Input data
        labels: Label data
        device: Device
        epochs: Number of training epochs
        **kwargs: Other parameters
    
    Returns:
        dict: Training results
    """
    # Process data
    result = process_data_with_ai(data, labels, **kwargs)
    
    model = result['model']
    loader = result['data_loader']
    
    # Move to device
    device = torch.device(device)
    model.to(device)
    
    # Train model
    logger.info(f"Starting model training: {result['recommendation'].model_name}")
    trained_model = train_one_model(
        model, loader, device=device, epochs=epochs
    )
    
    # Evaluate model
    logger.info("Starting model evaluation")
    metrics = evaluate_model(trained_model, loader, device=device)
    
    return {
        'model': trained_model,
        'metrics': metrics,
        'data_profile': result['data_profile'],
        'recommendation': result['recommendation']
    }

def demo_with_different_data_types():
    """Demonstrate processing of different data types"""
    
    print("=" * 60)
    print("Demo 1: Tabular Data")
    print("=" * 60)
    
    # Tabular data
    X_tabular = np.random.randn(1000, 20).astype("float32")
    y_tabular = np.random.choice(["A", "B", "C", "D"], size=1000)
    
    result1 = train_and_evaluate(X_tabular, y_tabular, epochs=3)
    print(f"Tabular data results: {result1['metrics']}")
    print(f"Recommended model: {result1['recommendation'].model_name}")
    print(f"Recommendation reason: {result1['recommendation'].reasoning}")
    
    print("\n" + "=" * 60)
    print("Demo 2: Image Data")
    print("=" * 60)
    
    # Image data
    X_image = np.random.randn(500, 3, 32, 32).astype("float32")
    y_image = np.random.choice([0, 1], size=500)
    
    result2 = train_and_evaluate(X_image, y_image, epochs=3)
    print(f"Image data results: {result2['metrics']}")
    print(f"Recommended model: {result2['recommendation'].model_name}")
    print(f"Recommendation reason: {result2['recommendation'].reasoning}")
    
    print("\n" + "=" * 60)
    print("Demo 3: Sequence Data")
    print("=" * 60)
    
    # Sequence data
    X_sequence = np.random.randn(300, 50, 10).astype("float32")  # (N, T, C)
    y_sequence = np.random.choice([0, 1, 2], size=300)
    
    result3 = train_and_evaluate(X_sequence, y_sequence, epochs=3)
    print(f"Sequence data results: {result3['metrics']}")
    print(f"Recommended model: {result3['recommendation'].model_name}")
    print(f"Recommendation reason: {result3['recommendation'].reasoning}")
    
    print("\n" + "=" * 60)
    print("Demo 4: Irregular Sequence Data")
    print("=" * 60)
    
    # Irregular sequence data
    X_irregular = [
        np.random.randn(np.random.randint(10, 50), 5) for _ in range(200)
    ]
    y_irregular = np.random.choice([0, 1], size=200)
    
    result4 = train_and_evaluate(X_irregular, y_irregular, epochs=3)
    print(f"Irregular sequence data results: {result4['metrics']}")
    print(f"Recommended model: {result4['recommendation'].model_name}")
    print(f"Recommendation reason: {result4['recommendation'].reasoning}")

if __name__ == "__main__":
    # Set OpenAI API key (if available)
    import os
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not set, will use default model selection")
        print("To use AI model selection feature, set environment variable: export OPENAI_API_KEY='your-api-key'")
    
    # Run demo
    demo_with_different_data_types()
