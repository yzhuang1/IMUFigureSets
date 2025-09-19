"""
AI-Enhanced Main Process with New Pipeline Flow
Model Generation → BO → Evaluation → Feedback Loop
"""


import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Dict, Any, Tuple
import pandas as pd
import os
from pathlib import Path

from adapters.universal_converter import convert_to_torch_dataset
from evaluation.code_generation_pipeline_orchestrator import CodeGenerationPipelineOrchestrator
# from visualization import create_charts_folder  # No longer needed - charts handled by orchestrator

# Setup centralized logging
from logging_config import get_pipeline_logger, get_log_file_path
from datetime import datetime

# Initialize pipeline logger
logger = get_pipeline_logger(__name__)

# Print setup confirmation to terminal
log_path = get_log_file_path()
if log_path:
    print(f"✅ Logging configured: {os.path.abspath(log_path)}")
else:
    print(f"⚠️  Logging not configured properly")

# Test logging (goes to file only)
logger.info("Logging system initialized successfully")

def train_with_iterative_selection(data, labels=None, device="cpu", epochs=5, **kwargs):
    """
    Train model with AI-enhanced pipeline: Model Generation → BO → Evaluation (single attempt, fail fast)
    
    Args:
        data: Input data
        labels: Label data
        device: Device for training
        epochs: Number of training epochs
        **kwargs: Additional parameters
    
    Returns:
        Dict: Training results with final model and evaluation
    """
    logger.info("Starting AI-enhanced training (single attempt, fail fast)")
    logger.info("Flow: Code Generation → BO → Evaluation")
    
    # Convert data and get profile
    dataset, collate_fn, data_profile = convert_to_torch_dataset(data, labels, **kwargs)
    
    logger.info(f"Data profile: {data_profile.to_dict() if hasattr(data_profile, 'to_dict') else str(data_profile)}")
    
    # Create data loader
    batch_size = kwargs.get('batch_size', 64)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    # Determine input shape for models
    if data_profile.is_sequence:
        # For sequence data like ECG (samples, seq_len, features), use (seq_len, features)
        if len(data_profile.shape) == 3:
            input_shape = data_profile.shape[1:]  # (seq_len, features)
        else:
            input_shape = (data_profile.feature_count,)
    elif data_profile.is_image:
        if data_profile.channels and data_profile.height and data_profile.width:
            input_shape = (data_profile.channels, data_profile.height, data_profile.width)
        else:
            input_shape = (3, 32, 32)  # Default
    else:
        input_shape = (data_profile.feature_count,)
    
    num_classes = data_profile.label_count if data_profile.has_labels else 2
    
    # Create code generation orchestrator for complete pipeline
    orchestrator = CodeGenerationPipelineOrchestrator(
        data_profile=data_profile.to_dict()
    )
    
    # Run complete pipeline
    best_model, pipeline_results = orchestrator.run_complete_pipeline(
        X=data, y=labels,
        device=device,
        input_shape=input_shape,
        num_classes=num_classes,
        epochs=epochs,
        **kwargs
    )
    
    # Prepare results from new pipeline
    results = {
        'model': best_model,
        'data_profile': data_profile,
        'pipeline_results': pipeline_results,
        'final_metrics': pipeline_results['final_metrics'],
        'attempt_summary': orchestrator.get_pipeline_summary(),
        'dataset': dataset,
        'data_loader': loader,
        'collate_fn': collate_fn
    }
    
    logger.info("AI-enhanced training completed!")
    logger.info(f"Final model achieved: {dict(results['final_metrics']) if results['final_metrics'] else 'None'}")
    logger.info(f"Pipeline completed successfully in single attempt")
    
    return results

def process_data_with_ai_enhanced_evaluation(data, labels=None, **kwargs):
    """
    Process data with AI-enhanced evaluation and new pipeline flow
    
    Args:
        data: Input data (any format)
        labels: Label data
        **kwargs: Other parameters
    
    Returns:
        dict: Dictionary containing best model, analysis, and attempt history
    """
    logger.info("Starting AI-enhanced data processing with new pipeline flow...")
    
    # Set device - remove from kwargs to avoid duplicate parameter error
    device = kwargs.pop('device', "cuda" if torch.cuda.is_available() else "cpu")
    
    # Force CPU if CUDA is not properly available
    try:
        torch.cuda.current_device()
    except:
        device = "cpu"
    
    # Train with new pipeline flow
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
    
    # Prioritize NPY files if they exist
    npy_files = [f for f in data_files if f.suffix == '.npy']
    if npy_files:
        logger.info("Found NPY files, prioritizing them over CSV")
        data_files = npy_files + [f for f in data_files if f.suffix != '.npy']
    
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
                    try:
                        X = np.load(file_path, allow_pickle=True)
                    except Exception as e:
                        logger.error(f"Failed to load {file_path.name}: {e}")
                        continue
                    
                    # Look for corresponding y file
                    y_candidates = [
                        data_dir / f"y{file_path.suffix}",
                        data_dir / f"labels{file_path.suffix}",
                        data_dir / f"target{file_path.suffix}",
                        data_dir / f"{file_path.stem.replace('X', 'y')}{file_path.suffix}"
                    ]
                    
                    for y_path in y_candidates:
                        if y_path.exists():
                            try:
                                y = np.load(y_path, allow_pickle=True)
                                
                                data_info = {
                                    'source': f"{file_path.name} + {y_path.name}",
                                    'format': 'NPY',
                                    'shape': X.shape,
                                    'classes': len(np.unique(y)) if y.ndim == 1 else 'varied'
                                }
                                
                                logger.info(f"Successfully loaded NPY data: X{X.shape}, y{y.shape}")
                                return X.astype('float32'), y, data_info
                            except Exception as e:
                                logger.error(f"Failed to load {y_path.name}: {e}")
                                continue
                    
                    logger.warning(f"Found {file_path.name} but no corresponding target file")
                
                else:
                    # Single NPY file - assume it contains both X and y
                    try:
                        data = np.load(file_path, allow_pickle=True)
                    except Exception as e:
                        logger.error(f"Failed to load {file_path.name}: {e}")
                        continue
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
                try:
                    data = np.load(file_path, allow_pickle=True)
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
    logger.info("Starting real data processing from data/ directory")
    
    # Try to load data
    data_result = load_data_from_files()
    
    if data_result is None:
        print("\n[NOTICE] No data files found or could not load data")
        print("\nTo use your own data, place files in the data/ directory:")
        print("  - CSV: dataset.csv (with target column named 'target', 'label', etc.)")
        print("  - NumPy: X.npy + y.npy (or features.npy + labels.npy)")
        print("  - NPZ: data.npz (containing 'X' and 'y' arrays)")
        logger.info("No data files found in data/ directory")
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

    logger.info(f"Data loaded successfully: {data_info['source']}, shape: {data_info['shape']}, device: {device}")
    
    # Check OpenAI API key
    from config import config
    if not config.is_openai_configured():
        print("\n[NOTICE] OpenAI API key required for AI-enhanced processing!")
        logger.warning("OpenAI API key not configured")
        return False
    
    print(f"  OpenAI: {config.openai_model}")
    
    print("\n⏳ Starting AI-enhanced processing...")
    print("\n" + "=" * 60)
    print("Running AI-Enhanced Pipeline with Code Generation")
    print("=" * 60)
    print("Flow: AI Code Generation → JSON Storage → BO → Training Execution → Evaluation")
    
    # Process with AI-enhanced evaluation
    result = process_data_with_ai_enhanced_evaluation(
        X, y,
        device=device,
        epochs=8  # Reasonable number for real data
    )
    
    print(f"\nAI-Enhanced Processing Results:")
    print(f"  Final model metrics: {result['final_metrics']}")
    print(f"  Pipeline success: {result['attempt_summary']['final_success']}")
    print(f"  Total model attempts: {result['attempt_summary']['total_attempts']}")
    print(f"  Best model: {result['pipeline_results']['model_name']}")

    logger.info(f"Pipeline completed: {result['pipeline_results']['model_name']}, metrics: {result['final_metrics']}")
    
    # Charts are automatically generated by the pipeline orchestrator in subfolders
    print(f"\n📊 BO charts have been automatically generated by the pipeline orchestrator")
    
    # Save pipeline summary
    import json
    from datetime import datetime
    
    timestamp_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def convert_numpy_types(obj):
        if hasattr(obj, 'item'):
            return obj.item()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(v) for v in obj]
        elif hasattr(obj, '__dict__'):
            return convert_numpy_types(obj.__dict__)
        else:
            return obj
    
    pipeline_summary = {
        'pipeline_type': 'AI Code Generation with BO and Feedback Loop',
        'final_performance': convert_numpy_types(result['final_metrics']),
        'pipeline_history': result['pipeline_results'].get('pipeline_history', []),
        'total_attempts': result['attempt_summary']['total_attempts'],
        'successful_attempts': result['attempt_summary']['successful_attempts'],
        'final_success': result['attempt_summary']['final_success'],
        'best_model': result['pipeline_results']['model_name'],
        'generated_functions': result['pipeline_results'].get('generated_functions', [])
    }
    
    with open(f"charts/pipeline_summary_{timestamp_suffix}.json", 'w') as f:
        json.dump(pipeline_summary, f, indent=2, default=str)
    
    print(f"\n[SUCCESS] Pipeline summary saved to charts/pipeline_summary_{timestamp_suffix}.json")
    logger.info(f"Pipeline summary saved to charts/pipeline_summary_{timestamp_suffix}.json")

    # Save the final trained model with best parameters to trained_models folder
    if 'model' in result and result['model'] is not None:
        model_filename = f"best_model_{result['pipeline_results']['model_name']}_{timestamp_suffix}.pth"
        model_save_path = f"trained_models/{model_filename}"

        # Save model state dict and metadata
        model_data = {
            'model_state_dict': result['model'].state_dict(),
            'model_name': result['pipeline_results']['model_name'],
            'final_metrics': convert_numpy_types(result['final_metrics']),
            'best_hyperparameters': result['pipeline_results'].get('optimized_hyperparameters', {}),
            'data_profile': result['data_profile'].to_dict() if hasattr(result['data_profile'], 'to_dict') else str(result['data_profile']),
            'timestamp': timestamp_suffix,
            'bo_best_score': result['pipeline_results'].get('bo_results', {}).get('best_value', 0.0)
        }

        torch.save(model_data, model_save_path)
        print(f"\n[SUCCESS] Final trained model saved to {model_save_path}")
        print(f"  📊 Model: {result['pipeline_results']['model_name']}")
        print(f"  📈 Performance: {result['final_metrics']}")
        print(f"  ⚙️  Best hyperparameters: {result['pipeline_results'].get('optimized_hyperparameters', {})}")

        logger.info(f"Model saved: {model_save_path}, performance: {result['final_metrics']}")

        # Update pipeline summary with model save path
        pipeline_summary['saved_model_path'] = model_save_path

        # Re-save pipeline summary with model path
        with open(f"charts/pipeline_summary_{timestamp_suffix}.json", 'w') as f:
            json.dump(pipeline_summary, f, indent=2, default=str)
    
    print(f"\n" + "=" * 80)
    print(f"🎉 AI-ENHANCED PROCESSING COMPLETED SUCCESSFULLY!")
    print(f"=" * 80)

    logger.info("AI-enhanced processing completed successfully")

    return True

def setup_global_exception_handler():
    """Setup global exception handler to capture unhandled exceptions and log them"""
    import sys
    import traceback
    from logging_config import get_pipeline_logger

    def handle_exception(exc_type, exc_value, exc_traceback):
        """Global exception handler that logs exceptions before termination"""
        # Skip if it's KeyboardInterrupt
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        # Get logger and log the exception
        logger = get_pipeline_logger(__name__)
        error_msg = f"Unhandled exception: {exc_type.__name__}: {exc_value}"

        # Log the full traceback
        tb_str = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        logger.error(f"{error_msg}\n{tb_str}")

        # Also write to stderr (which the error monitor will catch)
        sys.stderr.write(f"ERROR: {error_msg}\n")
        sys.stderr.flush()

        # Give error monitor time to process the error
        import time
        time.sleep(0.1)  # Small delay to ensure error monitor can react

        # Call the default handler
        sys.__excepthook__(exc_type, exc_value, exc_traceback)

    # Set the global exception handler
    sys.excepthook = handle_exception

if __name__ == "__main__":
    # Setup global exception handler first
    setup_global_exception_handler()

    print("\n🤖 AI-Enhanced Machine Learning Pipeline")
    print("Code Generation Flow: AI Code Generation → JSON Storage → BO → Training Execution → Evaluation")
    print("=" * 80)
    logger.info("Starting AI-Enhanced Machine Learning Pipeline")

    # Process real data from data/ directory
    processed_real_data = process_real_data()
    
    # If no real data found, show instructions
    if not processed_real_data:
        print("\n[NOTICE] No data files found in data/ directory")
        print("\n📁 To use the AI-enhanced ML pipeline, add your data files to the data/ directory:")
        print("  • CSV: dataset.csv (with target column named 'target', 'label', 'class', 'y', or 'output')")
        print("  • NumPy: X.npy + y.npy (or features.npy + labels.npy)")
        print("  • NPZ: data.npz (containing 'X' and 'y' arrays)")
        print("\n🚀 The pipeline will automatically:")
        print("  1. Load and analyze your data → 2. Generate AI training code → 3. Optimize hyperparameters")
        print("  4. Execute training → 5. Evaluate performance → 6. Generate charts and summaries")
        print("\n📂 Example: data/my_dataset.csv  or  data/X.npy + data/y.npy")

        logger.info("Main execution completed - no data files found")