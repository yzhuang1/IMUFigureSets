"""
Graph Pipeline - Starts from BO Process
Takes a training function JSON path and runs: BO → Training → Evaluation
Saves all outputs to graph/new/ with original folder structure
"""

import sys
import os
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import torch
import numpy as np
import json
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from _models.training_function_executor import training_executor, BO_TrainingObjective
from visualization import generate_bo_charts
from config import config
from data_splitting import get_bo_subset, get_current_splits, create_consistent_splits
from error_monitor import set_bo_process_mode
from logging_config import get_pipeline_logger
from adapters.universal_converter import convert_to_torch_dataset

logger = get_pipeline_logger(__name__)


class GraphPipeline:
    """Pipeline that starts from BO process using existing training function"""

    def __init__(self, training_function_path: str):
        """
        Initialize graph pipeline with training function

        Args:
            training_function_path: Path to training function JSON file
        """
        self.training_function_path = training_function_path
        self.training_data = None
        self.output_base_dir = Path("graph/new")

        # Load training function
        self._load_training_function()

        logger.info(f"GraphPipeline initialized with: {self.training_data['model_name']}")

    def _load_training_function(self):
        """Load training function from JSON"""
        try:
            self.training_data = training_executor.load_training_function(self.training_function_path)
            logger.info(f"Loaded training function: {self.training_data['model_name']}")
        except Exception as e:
            logger.error(f"Failed to load training function: {e}")
            raise

    def run(self, X, y, device: str = None, **kwargs) -> Dict[str, Any]:
        """
        Run the graph pipeline: BO → Training → Evaluation

        Args:
            X: Input data
            y: Labels
            device: Training device
            **kwargs: Additional parameters

        Returns:
            Dict: Pipeline results
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info("="*60)
        logger.info("GRAPH PIPELINE EXECUTION")
        logger.info("="*60)
        logger.info(f"Model: {self.training_data['model_name']}")
        logger.info(f"Device: {device}")

        # Create consistent data splits
        logger.info("Creating centralized data splits")
        splits = create_consistent_splits(X, y, test_size=0.2, val_size=0.2)

        # Enable BO process mode
        from logging_config import get_log_file_path
        log_file_path = get_log_file_path()
        set_bo_process_mode(True, log_file_path)

        # STEP 1: Bayesian Optimization
        logger.info("🔍 STEP 1: Bayesian Optimization")
        bo_results = self._run_bayesian_optimization(X, y, device)

        # Disable BO process mode
        set_bo_process_mode(False)

        # STEP 2: Final Training with Optimized Parameters
        logger.info("🚀 STEP 2: Final Training Execution")
        final_model, training_results = self._execute_final_training(
            X, y, device, bo_results
        )

        # STEP 3: Save results to graph/new/
        logger.info("💾 STEP 3: Saving Results")
        self._save_results(final_model, training_results, bo_results)

        logger.info("="*60)
        logger.info("GRAPH PIPELINE COMPLETE")
        logger.info("="*60)

        results = {
            'model': final_model,
            'model_name': self.training_data['model_name'],
            'training_function_path': self.training_function_path,
            'bo_results': bo_results,
            'training_metrics': training_results['training_metrics'],
            'final_metrics': training_results['final_metrics'],
            'optimized_hyperparameters': bo_results['best_params']
        }

        return results

    def _run_bayesian_optimization(self, X, y, device: str) -> Dict[str, Any]:
        """Run Bayesian Optimization"""
        logger.info(f"Running BO for: {self.training_data['model_name']}")

        # Install dependencies
        logger.info("📦 Installing dependencies...")
        from package_installer import install_gpt_code_dependencies
        try:
            install_gpt_code_dependencies(self.training_data['training_code'], raise_on_failure=True)
            logger.info("✅ Dependencies installed")
        except RuntimeError as e:
            logger.error(f"❌ Failed to install packages: {e}")
            raise

        # Get BO subset
        X_bo, y_bo = get_bo_subset()
        logger.info(f"BO dataset size: {len(X_bo)} samples")

        # Create BO objective function
        objective_func = BO_TrainingObjective(self.training_data, None, None, device)

        # Set up search space
        from bo.run_bo import BayesianOptimizer

        search_space = {}
        for param_name, param_config in self.training_data['bo_config'].items():
            search_space[param_name] = {k: v for k, v in param_config.items() if k != "default"}

        bo_optimizer = BayesianOptimizer(gpt_search_space=search_space, n_initial_points=3)

        # Run BO
        results = []
        best_score = 0.0

        for trial in range(config.max_bo_trials):
            hparams = bo_optimizer.suggest()
            print(f"🔍 BO Trial {trial + 1}/{config.max_bo_trials}: {hparams}")

            value, metrics = objective_func(hparams)

            # Check for errors
            if value == 0.0 and "error" in metrics:
                logger.error(f"BO Trial {trial + 1} FAILED: {metrics['error']}")
                results.append({
                    'trial': trial + 1,
                    'hparams': hparams,
                    'value': value,
                    'metrics': metrics
                })
                # For graph pipeline, we'll continue instead of fail-fast
                logger.warning("Continuing with remaining trials despite error")
                continue

            results.append({
                'trial': trial + 1,
                'hparams': hparams,
                'value': value,
                'metrics': metrics
            })

            bo_optimizer.observe(hparams, value)
            best_score = max(best_score, value)

            logger.info(f"BO Trial {trial + 1}: {hparams} -> {value:.4f}")

        # Get best results
        if results:
            best_result = max(results, key=lambda x: x['value'])
            bo_results = {
                'best_value': best_result['value'],
                'best_params': best_result['hparams'],
                'total_trials': len(results),
                'all_results': results,
                'convergence_info': bo_optimizer.get_convergence_info()
            }
        else:
            bo_results = {
                'best_value': 0.0,
                'best_params': {param: param_config["default"] for param, param_config in self.training_data['bo_config'].items()},
                'total_trials': 0,
                'all_results': []
            }

        logger.info(f"BO completed - Best score: {bo_results['best_value']:.4f}")
        logger.info(f"Best params: {bo_results['best_params']}")

        return bo_results

    def _execute_final_training(
        self,
        X, y,
        device: str,
        bo_results: Dict[str, Any]
    ) -> Tuple[torch.nn.Module, Dict[str, Any]]:
        """Execute final training with optimized hyperparameters"""

        splits = get_current_splits()
        logger.info(f"Using splits - Train: {splits.X_train.shape}, Val: {splits.X_val.shape}, Test: {splits.X_test.shape}")

        # Create datasets
        train_dataset, _, _ = convert_to_torch_dataset(
            splits.X_train, splits.y_train,
            standardize=True, standardization_stats=None
        )
        val_dataset, _, _ = convert_to_torch_dataset(
            splits.X_val, splits.y_val,
            standardize=True, standardization_stats=train_dataset.standardization_stats
        )
        test_dataset, _, _ = convert_to_torch_dataset(
            splits.X_test, splits.y_test,
            standardize=True, standardization_stats=train_dataset.standardization_stats
        )

        X_train_final = train_dataset.X
        y_train_final = train_dataset.y
        X_test_tensor = test_dataset.X
        y_test_tensor = test_dataset.y

        # Get best hyperparameters
        best_params = bo_results['best_params']

        logger.info(f"Executing final training with: {best_params}")

        trained_model, training_metrics = training_executor.execute_training_function(
            self.training_data,
            X_train_final, y_train_final,
            X_test_tensor, y_test_tensor,
            device=device,
            **best_params
        )

        # Extract performance metrics
        val_f1 = training_metrics.get('val_f1', [])
        val_acc = training_metrics.get('val_acc', [])

        if val_f1 and isinstance(val_f1, list) and len(val_f1) > 0:
            performance_metric = val_f1[-1]
        elif 'macro_f1' in training_metrics:
            performance_metric = training_metrics['macro_f1']
        elif val_acc and isinstance(val_acc, list) and len(val_acc) > 0:
            performance_metric = val_acc[-1]
        elif 'val_accuracy' in training_metrics:
            performance_metric = training_metrics['val_accuracy']
        elif 'best_val_acc' in training_metrics:
            performance_metric = training_metrics['best_val_acc']
        else:
            performance_metric = 0.0

        final_metrics = {
            'acc': performance_metric,
            'macro_f1': training_metrics.get('macro_f1', None)
        }

        logger.info(f"Final metrics: {final_metrics}")

        results = {
            'model': trained_model,
            'training_metrics': training_metrics,
            'final_metrics': final_metrics,
            'X_test': X_test_tensor,
            'y_test': y_test_tensor
        }

        return trained_model, results

    def _save_results(self, model, training_results, bo_results):
        """Save all results to graph/new/ folders (matching main.py structure)"""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = self.training_data['model_name']

        # ========== SAVE 1: graph/new/trained_models - Model .pt files ==========
        trained_models_dir = self.output_base_dir / "trained_models" / f"{timestamp}_{model_name}"
        trained_models_dir.mkdir(parents=True, exist_ok=True)

        # Save model state dict
        model_path = trained_models_dir / "model.pt"
        torch.save(model.state_dict(), model_path)
        logger.info(f"Model saved to: {model_path}")

        # Save test tensors
        X_test_path = trained_models_dir / "X_test.pt"
        y_test_path = trained_models_dir / "y_test.pt"
        torch.save(training_results['X_test'], X_test_path)
        torch.save(training_results['y_test'], y_test_path)
        logger.info(f"Test tensors saved to: {trained_models_dir}")

        # ========== SAVE 2: graph/new/charts - BO visualizations ==========
        charts_dir = self.output_base_dir / "charts" / f"{timestamp}_BO_{model_name}"
        charts_dir.mkdir(parents=True, exist_ok=True)

        # Generate BO charts
        chart_results = {
            'all_results': bo_results.get('all_results', []),
            'best_value': bo_results.get('best_value', 0),
            'best_params': bo_results.get('best_params', {}),
            'model_name': model_name
        }
        generate_bo_charts(chart_results, save_folder=str(charts_dir))
        logger.info(f"BO charts saved to: {charts_dir}")

        # ========== SAVE 3: graph/new/data/best_model_* - Training function and metadata ==========
        # This mirrors the structure in graph/data where training functions are stored
        best_model_dir = self.output_base_dir / "data" / f"best_model_{model_name}_{timestamp}"
        best_model_dir.mkdir(parents=True, exist_ok=True)

        # Copy training function JSON to best_model folder
        import shutil
        training_json_dest = best_model_dir / Path(self.training_function_path).name
        shutil.copy2(self.training_function_path, training_json_dest)
        logger.info(f"Training function copied to: {training_json_dest}")

        # Save a reference file pointing to the trained model
        model_reference = {
            'model_name': model_name,
            'timestamp': timestamp,
            'training_function_path': str(training_json_dest),
            'trained_model_path': str(model_path),
            'test_tensors_path': str(trained_models_dir),
            'bo_charts_path': str(charts_dir),
            'final_metrics': self._convert_numpy_types(training_results['final_metrics']),
            'bo_best_score': bo_results['best_value'],
            'bo_best_params': self._convert_numpy_types(bo_results['best_params']),
            'bo_total_trials': bo_results['total_trials']
        }

        reference_path = best_model_dir / "model_reference.json"
        with open(reference_path, 'w') as f:
            json.dump(model_reference, f, indent=2, default=str)
        logger.info(f"Model reference saved to: {reference_path}")

        # ========== SAVE 4: Pipeline summary in charts folder ==========
        pipeline_summary = {
            'pipeline_type': 'Graph Pipeline (BO → Training → Evaluation)',
            'training_function_path': self.training_function_path,
            'model_name': model_name,
            'final_metrics': self._convert_numpy_types(training_results['final_metrics']),
            'bo_best_score': bo_results['best_value'],
            'bo_best_params': self._convert_numpy_types(bo_results['best_params']),
            'bo_total_trials': bo_results['total_trials'],
            'saved_model_path': str(model_path),
            'best_model_folder': str(best_model_dir),
            'timestamp': timestamp
        }

        summary_path = self.output_base_dir / "charts" / f"pipeline_summary_{timestamp}.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)

        with open(summary_path, 'w') as f:
            json.dump(pipeline_summary, f, indent=2, default=str)

        logger.info(f"Pipeline summary saved to: {summary_path}")

    def _convert_numpy_types(self, obj):
        """Convert numpy types to native Python types for JSON serialization"""
        if hasattr(obj, 'item'):
            return obj.item()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(v) for v in obj]
        else:
            return obj


def run_graph_pipeline(training_function_path: str, X, y, device: str = None, **kwargs):
    """
    Run graph pipeline with training function

    Args:
        training_function_path: Path to training function JSON
        X: Input data
        y: Labels
        device: Training device
        **kwargs: Additional parameters

    Returns:
        Dict: Pipeline results
    """
    pipeline = GraphPipeline(training_function_path)
    return pipeline.run(X, y, device=device, **kwargs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Graph Pipeline - Start from BO process")
    parser.add_argument("--training_function", type=str, required=True,
                       help="Path to training function JSON file")
    parser.add_argument("--data_folder", type=str, default=None,
                       help="Data folder name (default: use config.data_folder)")

    args = parser.parse_args()

    # Load data
    if args.data_folder:
        config.data_folder = args.data_folder

    from main import load_data_from_files

    print("\n" + "=" * 80)
    print(f"GRAPH PIPELINE - Starting from BO Process")
    print("=" * 80)

    data_result = load_data_from_files()

    if data_result is None:
        print("\n[ERROR] No data files found")
        sys.exit(1)

    X, y, data_info = data_result

    print(f"[SUCCESS] Data loaded: {data_info['source']}")
    print(f"  Shape: {data_info['shape']}")
    print(f"  Classes: {data_info['classes']}")

    # Run graph pipeline
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    print(f"\n⏳ Running graph pipeline with: {args.training_function}")

    results = run_graph_pipeline(
        args.training_function,
        X, y,
        device=device,
        epochs=8
    )

    print(f"\n[SUCCESS] Graph pipeline completed!")
    print(f"  Model: {results['model_name']}")
    print(f"  Final metrics: {results['final_metrics']}")
    print(f"  Best BO score: {results['bo_results']['best_value']:.4f}")
    print(f"  Best params: {results['optimized_hyperparameters']}")
    print(f"\n📊 Results saved to: graph/new/")
