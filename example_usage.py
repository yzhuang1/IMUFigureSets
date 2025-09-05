"""
Usage examples: Demonstrating how to use the new AI-enhanced machine learning pipeline
"""

import numpy as np
import torch
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def example_1_simple_usage():
    """Example 1: Simple usage"""
    print("=" * 60)
    print("Example 1: Simple usage")
    print("=" * 60)
    
    from main_new import process_data_with_ai, train_and_evaluate
    
    # Prepare data
    X = np.random.randn(1000, 20).astype("float32")
    y = np.random.choice(["A", "B", "C"], size=1000)
    
    # Use AI to automatically process data and select model
    result = process_data_with_ai(X, y)
    
    print(f"Data profile: {result['data_profile']}")
    print(f"AI recommended model: {result['recommendation'].model_name}")
    print(f"Recommendation reason: {result['recommendation'].reasoning}")
    
    # Train and evaluate
    training_result = train_and_evaluate(X, y, epochs=3)
    print(f"Training results: {training_result['metrics']}")

def example_2_different_data_types():
    """Example 2: Different data types"""
    print("\n" + "=" * 60)
    print("Example 2: Different data types")
    print("=" * 60)
    
    from main_new import train_and_evaluate
    
    # Tabular data
    print("Processing tabular data...")
    X_tabular = np.random.randn(500, 15).astype("float32")
    y_tabular = np.random.choice([0, 1, 2], size=500)
    result1 = train_and_evaluate(X_tabular, y_tabular, epochs=2)
    print(f"Tabular data - Recommended model: {result1['recommendation'].model_name}")
    
    # Image data
    print("Processing image data...")
    X_image = np.random.randn(200, 3, 32, 32).astype("float32")
    y_image = np.random.choice([0, 1], size=200)
    result2 = train_and_evaluate(X_image, y_image, epochs=2)
    print(f"Image data - Recommended model: {result2['recommendation'].model_name}")
    
    # Sequence data
    print("Processing sequence data...")
    X_sequence = np.random.randn(300, 50, 10).astype("float32")
    y_sequence = np.random.choice([0, 1, 2], size=300)
    result3 = train_and_evaluate(X_sequence, y_sequence, epochs=2)
    print(f"Sequence data - Recommended model: {result3['recommendation'].model_name}")

def example_3_bo_optimization():
    """Example 3: Bayesian Optimization"""
    print("\n" + "=" * 60)
    print("Example 3: Bayesian Optimization")
    print("=" * 60)
    
    from bo.run_ai_enhanced_bo import run_ai_enhanced_bo
    
    # Prepare data
    X = np.random.randn(800, 12).astype("float32")
    y = np.random.choice(["X", "Y", "Z"], size=800)
    
    # Run BO optimization
    result = run_ai_enhanced_bo(X, y, n_trials=3)
    
    print(f"BO optimization results:")
    print(f"  Best value: {result['best_value']:.4f}")
    print(f"  Best parameters: {result['best_params']}")
    print(f"  AI recommended model: {result['ai_recommendation']['model_name']}")

def example_4_custom_data_conversion():
    """Example 4: Custom data conversion"""
    print("\n" + "=" * 60)
    print("Example 4: Custom data conversion")
    print("=" * 60)
    
    from adapters.universal_converter import convert_to_torch_dataset
    
    # Irregular sequence data
    X_irregular = [
        np.random.randn(np.random.randint(10, 50), 5) for _ in range(100)
    ]
    y_irregular = np.random.choice([0, 1], size=100)
    
    # Convert data
    dataset, collate_fn, profile = convert_to_torch_dataset(X_irregular, y_irregular)
    
    print(f"Data profile: {profile}")
    print(f"Dataset size: {len(dataset)}")
    print(f"Requires collate function: {collate_fn is not None}")

def example_5_model_registry():
    """Example 5: Model registry system"""
    print("\n" + "=" * 60)
    print("Example 5: Model registry system")
    print("=" * 60)
    
    from models.dynamic_model_registry import model_registry, build_model
    
    # View registered models
    models = model_registry.list_models()
    print("Registered models:")
    for model_info in models:
        print(f"  - {model_info['name']}: {model_info['description']}")
    
    # Build model
    model = build_model("TabMLP", input_shape=(20,), num_classes=3, hidden=128)
    print(f"\nBuilt model: {model}")

def example_6_ai_model_selection():
    """Example 6: AI model selection"""
    print("\n" + "=" * 60)
    print("Example 6: AI model selection")
    print("=" * 60)
    
    from models.ai_model_selector import select_model_for_data
    from adapters.universal_converter import analyze_data_profile
    
    # Analyze data characteristics
    X = np.random.randn(500, 25).astype("float32")
    y = np.random.choice([0, 1, 2, 3], size=500)
    
    profile = analyze_data_profile(X, y)
    print(f"Data profile: {profile}")
    
    # AI model selection
    recommendation = select_model_for_data(profile.to_dict())
    print(f"AI recommendation: {recommendation.model_name}")
    print(f"Recommendation reason: {recommendation.reasoning}")
    print(f"Confidence: {recommendation.confidence:.2f}")

if __name__ == "__main__":
    print("AI-Enhanced Machine Learning Pipeline Usage Examples")
    print("=" * 80)
    
    # Set OpenAI API key (if available)
    import os
    if not os.getenv("OPENAI_API_KEY"):
        print("Note: OPENAI_API_KEY not set, will use default model selection")
        print("To use AI model selection feature, set environment variable: export OPENAI_API_KEY='your-api-key'")
    
    try:
        example_1_simple_usage()
        example_2_different_data_types()
        example_3_bo_optimization()
        example_4_custom_data_conversion()
        example_5_model_registry()
        example_6_ai_model_selection()
        
        print("\n" + "=" * 80)
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()
