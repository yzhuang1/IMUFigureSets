"""
AI-Enhanced Main Process with Iterative Model Selection
Processes real data files from data/ directory only
"""

import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Dict, Any, Tuple
import pandas as pd
import os
from pathlib import Path

from adapters.universal_converter import convert_to_torch_dataset
from models.dynamic_model_registry import build_model_from_recommendation
from evaluation.ai_enhanced_evaluate import IterativeModelSelector, AIEnhancedEvaluator
from train import train_one_model
from bo.run_ai_enhanced_bo import run_ai_enhanced_bo
from visualization import generate_bo_charts, create_charts_folder

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
    
    # Set device - remove from kwargs to avoid duplicate parameter error
    device = kwargs.pop('device', "cuda" if torch.cuda.is_available() else "cpu")
    
    # Force CPU if CUDA is not properly available
    try:
        torch.cuda.current_device()
    except:
        device = "cpu"
    
    # Train with iterative selection
    result = train_with_iterative_selection(data, labels, device=device, **kwargs)
    
    return result

def load_data_from_files():
    """
    Load data from files in the data/ directory
    Supports CSV, NPY, and other common formats
    
    Returns:
        tuple: (X, y, data_info) or None if no data found
    """
    data_dir = Path("data")
    
    if not data_dir.exists():
        logger.warning("data/ directory not found")
        return None
    
    # Look for common data files
    data_files = []
    for ext in ['*.csv', '*.npy', '*.npz', '*.pkl', '*.json']:
        data_files.extend(list(data_dir.glob(ext)))
    
    if not data_files:
        logger.warning("No data files found in data/ directory")
        logger.info("Supported formats: CSV, NPY, NPZ, PKL, JSON")
        logger.info("Expected structure: features + target column (CSV) or X.npy + y.npy")
        return None
    
    logger.info(f"Found {len(data_files)} data files: {[f.name for f in data_files]}")
    
    # Try to load data
    for file_path in data_files:
        try:
            logger.info(f"Attempting to load: {file_path.name}")
            
            if file_path.suffix == '.csv':
                # Load CSV file
                df = pd.read_csv(file_path)
                logger.info(f"CSV loaded: shape {df.shape}, columns: {list(df.columns)}")
                
                # Try to identify target column
                target_cols = [col for col in df.columns if any(keyword in col.lower() 
                             for keyword in ['target', 'label', 'class', 'y', 'output'])]
                
                if target_cols:
                    target_col = target_cols[0]
                    X = df.drop(target_col, axis=1).values.astype('float32')
                    y = df[target_col].values
                    
                    data_info = {
                        'source': file_path.name,
                        'format': 'CSV',
                        'shape': X.shape,
                        'target_column': target_col,
                        'classes': len(np.unique(y)) if hasattr(y[0], '__len__') == False else 'varied'
                    }
                    
                    logger.info(f"Successfully loaded CSV data: X{X.shape}, y{y.shape}")
                    return X, y, data_info
                else:
                    logger.warning(f"No target column found in {file_path.name}")
                    logger.info("Expected target column names: 'target', 'label', 'class', 'y', 'output'")
            
            elif file_path.suffix == '.npy':
                # Check for X.npy and y.npy pair
                if 'X' in file_path.stem or 'features' in file_path.stem.lower():
                    X = np.load(file_path)
                    
                    # Look for corresponding y file
                    y_candidates = [
                        data_dir / f"y{file_path.suffix}",
                        data_dir / f"labels{file_path.suffix}",
                        data_dir / f"target{file_path.suffix}",
                        data_dir / f"{file_path.stem.replace('X', 'y')}{file_path.suffix}"
                    ]
                    
                    for y_path in y_candidates:
                        if y_path.exists():
                            y = np.load(y_path)
                            
                            data_info = {
                                'source': f"{file_path.name} + {y_path.name}",
                                'format': 'NPY',
                                'shape': X.shape,
                                'classes': len(np.unique(y)) if y.ndim == 1 else 'varied'
                            }
                            
                            logger.info(f"Successfully loaded NPY data: X{X.shape}, y{y.shape}")
                            return X.astype('float32'), y, data_info
                    
                    logger.warning(f"Found {file_path.name} but no corresponding target file")
                
                else:
                    # Single NPY file - assume it contains both X and y
                    data = np.load(file_path, allow_pickle=True)
                    if isinstance(data, dict) and 'X' in data and 'y' in data:
                        X, y = data['X'], data['y']
                        
                        data_info = {
                            'source': file_path.name,
                            'format': 'NPY (dict)',
                            'shape': X.shape,
                            'classes': len(np.unique(y)) if y.ndim == 1 else 'varied'
                        }
                        
                        logger.info(f"Successfully loaded NPY dict: X{X.shape}, y{y.shape}")
                        return X.astype('float32'), y, data_info
            
            elif file_path.suffix == '.npz':
                # NPZ file
                data = np.load(file_path)
                if 'X' in data.files and 'y' in data.files:
                    X, y = data['X'], data['y']
                    
                    data_info = {
                        'source': file_path.name,
                        'format': 'NPZ',
                        'shape': X.shape,
                        'classes': len(np.unique(y)) if y.ndim == 1 else 'varied'
                    }
                    
                    logger.info(f"Successfully loaded NPZ data: X{X.shape}, y{y.shape}")
                    return X.astype('float32'), y, data_info
                else:
                    logger.warning(f"NPZ file {file_path.name} missing 'X' or 'y' keys")
        
        except Exception as e:
            logger.error(f"Failed to load {file_path.name}: {e}")
            continue
    
    logger.error("Could not load any data files")
    return None

def process_real_data():
    """Process real data from data/ directory"""
    
    print("\n" + "=" * 80)
    print("PROCESSING REAL DATA FROM data/ DIRECTORY")
    print("=" * 80)
    
    # Try to load data
    data_result = load_data_from_files()
    
    if data_result is None:
        print("[ERROR] No data files found or could not load data")
        print("\nTo use your own data, place files in the data/ directory:")
        print("  - CSV: dataset.csv (with target column named 'target', 'label', etc.)")
        print("  - NumPy: X.npy + y.npy (or features.npy + labels.npy)")
        print("  - NPZ: data.npz (containing 'X' and 'y' arrays)")
        return False
    
    X, y, data_info = data_result
    
    print(f"[SUCCESS] Successfully loaded data:")
    print(f"  Source: {data_info['source']}")
    print(f"  Format: {data_info['format']}")
    print(f"  Shape: {data_info['shape']}")
    print(f"  Classes/Labels: {data_info['classes']}")
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    
    # Check OpenAI API key
    from config import config
    if not config.is_openai_configured():
        print("[ERROR] OpenAI API key required for AI-enhanced processing!")
        return False
    
    print(f"  OpenAI: {config.openai_model}")
    
    print("\n" + "=" * 60)
    print("Running AI-Enhanced Processing on Your Data")
    print("=" * 60)
    
    # Process with AI-enhanced evaluation
    result = process_data_with_ai_enhanced_evaluation(
        X, y,
        device=device,
        epochs=8,  # Reasonable number for real data
        max_model_attempts=1  # Use single model to avoid selection issues
    )
    
    print(f"\nAI-Enhanced Processing Results:")
    print(f"  Final model metrics: {result['final_metrics']}")
    print(f"  AI decision: {result['analysis'].decision}")
    print(f"  Total model attempts: {result['attempt_summary']['total_attempts']}")
    
    # Run BO with subset for demonstration
    if X.shape[0] <= 50000:  # Allow BO to run, will use subset anyway
        print(f"\n" + "=" * 60)
        print("Running Bayesian Optimization on Your Data")
        print("=" * 60)
        
        # Create charts folder
        charts_dir = create_charts_folder()
        print(f"Charts will be saved to: {charts_dir}")
        
        # Use subset if still large
        subset_size = min(500, len(X))
        X_subset = X[:subset_size]
        y_subset = y[:subset_size]
        
        n_trials = config.max_bo_trials
        print(f"Running {n_trials} BO trials on subset ({subset_size} samples)...")
        
        bo_results = run_ai_enhanced_bo(
            X_subset, y_subset,
            device=device,
            n_trials=n_trials
        )
        
        print(f"\nBO Results:")
        print(f"  Best F1 score: {bo_results['best_value']:.4f}")
        print(f"  Best parameters: {bo_results['best_params']}")
        print(f"  AI recommended model: {bo_results['ai_recommendation']['model_name']}")
        
        # Generate charts
        generate_bo_charts(bo_results, save_folder="charts")
        print(f"\n[SUCCESS] BO charts saved to charts/ folder")
        
        # Final evaluation: retrain best model on full dataset
        print(f"\n" + "=" * 60)
        print("Final Evaluation: Retraining Best Model on Full Dataset")
        print("=" * 60)
        
        best_params = bo_results['best_params']
        print(f"Best hyperparameters from BO: {best_params}")
        
        # Import train/test split for proper evaluation
        from sklearn.model_selection import train_test_split
        
        # Split full dataset for final evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        
        # Retrain with best BO parameters on full training set
        final_result = process_data_with_ai_enhanced_evaluation(
            X_train, y_train,
            device=device,
            epochs=best_params.get('epochs', 8),
            lr=best_params.get('lr', 0.001),
            hidden=best_params.get('hidden', 64),
            max_model_attempts=1  # Use the model we know works
        )
        
        # Evaluate on held-out test set
        final_model = final_result['model']
        final_dataset = final_result['dataset']
        final_loader = final_result['data_loader']
        
        # Get test loader
        from torch.utils.data import DataLoader, TensorDataset
        test_dataset = TensorDataset(
            torch.tensor(X_test, dtype=torch.float32), 
            torch.tensor(y_test, dtype=torch.long)
        )
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        # Evaluate on test set
        from evaluation.ai_enhanced_evaluate import AIEnhancedEvaluator
        from models.ai_model_selector import ModelRecommendation
        
        # Create a simple model recommendation for evaluation
        mock_recommendation = ModelRecommendation(
            model_name="TabMLP",
            model_type="tabular",
            architecture="Multi-layer perceptron",
            input_shape=(2000,),
            reasoning="Optimized TabMLP with best BO parameters",
            confidence=0.95,
            hyperparameters=best_params
        )
        
        evaluator = AIEnhancedEvaluator(
            data_profile=result['data_profile'].to_dict(),
            model_recommendation=mock_recommendation
        )
        final_analysis = evaluator.evaluate_with_ai_analysis(final_model, test_loader, device)
        final_test_metrics = final_analysis.metrics
        
        # Performance comparison
        print(f"\n" + "=" * 60)
        print("PERFORMANCE COMPARISON")
        print("=" * 60)
        print("Initial AI-selected model performance:")
        print(f"  Accuracy: {result['final_metrics']['acc']:.4f}")
        print(f"  Macro F1: {result['final_metrics']['macro_f1']:.4f}")
        
        print(f"\nBO-optimized model performance on test set:")
        print(f"  Accuracy: {final_test_metrics['acc']:.4f}")
        print(f"  Macro F1: {final_test_metrics['macro_f1']:.4f}")
        
        # Calculate improvement
        acc_improvement = final_test_metrics['acc'] - result['final_metrics']['acc']
        f1_improvement = final_test_metrics['macro_f1'] - result['final_metrics']['macro_f1']
        
        print(f"\nImprovement from Bayesian Optimization:")
        print(f"  Accuracy: {acc_improvement:+.4f} ({acc_improvement*100:+.2f}%)")
        print(f"  Macro F1: {f1_improvement:+.4f} ({f1_improvement*100:+.2f}%)")
        
        if acc_improvement > 0 or f1_improvement > 0:
            print(f"\n✓ Bayesian Optimization improved model performance!")
        else:
            print(f"\n- Initial model was already quite good (no improvement from BO)")
        
        # Save final results (convert numpy types to Python types for JSON)
        def convert_numpy_types(obj):
            if hasattr(obj, 'item'):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(v) for v in obj]
            else:
                return obj
        
        final_summary = {
            'initial_performance': result['final_metrics'],
            'bo_optimized_performance': final_test_metrics,
            'best_hyperparameters': convert_numpy_types(best_params),
            'accuracy_improvement': float(acc_improvement),
            'f1_improvement': float(f1_improvement),
            'total_bo_trials': int(bo_results['total_trials'])
        }
        
        import json
        with open("charts/final_comparison.json", 'w') as f:
            json.dump(final_summary, f, indent=2)
        
        print(f"\n[SUCCESS] Complete pipeline results saved to charts/final_comparison.json")
    
    else:
        print(f"\n[WARNING] Dataset too large ({X.shape[0]} samples) for BO demo, skipping...")
    
    return True

if __name__ == "__main__":
    print("AI-Enhanced Machine Learning Pipeline")
    print("=" * 80)
    
    # Process real data from data/ directory
    processed_real_data = process_real_data()
    
    # If no real data found, show instructions
    if not processed_real_data:
        print("\n[ERROR] No data files found in data/ directory")
        print("\nTo use the AI-enhanced ML pipeline, add your data files to the data/ directory:")
        print("  - CSV: dataset.csv (with target column named 'target', 'label', 'class', 'y', or 'output')")
        print("  - NumPy: X.npy + y.npy (or features.npy + labels.npy)")
        print("  - NPZ: data.npz (containing 'X' and 'y' arrays)")
        print("\nThe pipeline will automatically:")
        print("  1. Load and analyze your data")
        print("  2. Use AI to select optimal model architecture")
        print("  3. Train with iterative model improvement")
        print("  4. Run Bayesian Optimization with visualization")
        print("  5. Generate charts in charts/ folder")
        print("\nExample file structure:")
        print("  data/")
        print("    └── my_dataset.csv  # with features + target column")
        print("  Or:")
        print("  data/")
        print("    ├── X.npy          # features array")
        print("    └── y.npy          # labels array")