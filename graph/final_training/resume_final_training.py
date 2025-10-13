"""
Standalone Final Training Pipeline
Resumes training from BO results in logs and generated_training_functions
"""
import os
import re
import json
import torch
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple, Optional
import gc

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FinalTrainingResumer:
    """Resume final training from interrupted BO runs"""

    def __init__(self, logs_dir: str = "logs", generated_functions_dir: str = "generated_training_functions"):
        self.logs_dir = Path(logs_dir)
        self.generated_functions_dir = Path(generated_functions_dir)

    def find_incomplete_runs(self) -> list:
        """Find runs that completed BO but failed during final training"""
        incomplete_runs = []

        for log_file in sorted(self.logs_dir.glob("*.log"), reverse=True):
            run_info = self._analyze_log_file(log_file)
            if run_info and run_info['status'] == 'incomplete':
                incomplete_runs.append(run_info)

        return incomplete_runs

    def _analyze_log_file(self, log_path: Path) -> Optional[Dict[str, Any]]:
        """Analyze a log file to check if it needs final training"""
        try:
            with open(log_path, 'r') as f:
                content = f.read()

            # Check if BO completed
            if "STEP 4: Final Training Execution" not in content:
                return None

            # Check if final training completed
            if "STEP 5: Performance Analysis" in content:
                return None  # Already completed

            # Extract best params
            best_params_match = re.search(
                r'Best params: (\{[^}]+\})',
                content.replace('\n', ' ')
            )
            if not best_params_match:
                return None

            # Extract model name from training results
            model_name_match = re.search(
                r"'model_name':\s*'([^']+)'",
                content
            )
            if not model_name_match:
                # Try alternative patterns
                model_name_match = re.search(
                    r"Loaded training function: ([^\n]+)",
                    content
                )
            if not model_name_match:
                model_name_match = re.search(
                    r"Model Name: ([^\n]+)",
                    content
                )

            if not model_name_match:
                logger.warning(f"Could not find model name in {log_path}")
                return None

            model_name = model_name_match.group(1).strip()

            # Find corresponding training function JSON
            training_function_path = self._find_training_function(model_name, content)
            if not training_function_path:
                logger.warning(f"Could not find training function for {model_name}")
                return None

            # Parse best params
            best_params_str = best_params_match.group(1)
            # Need to parse the numpy types string representation
            best_params = self._parse_best_params(content)

            return {
                'status': 'incomplete',
                'log_file': str(log_path),
                'model_name': model_name,
                'training_function_path': training_function_path,
                'best_params': best_params,
                'log_timestamp': log_path.stem
            }

        except Exception as e:
            logger.error(f"Error analyzing {log_path}: {e}")
            return None

    def _parse_best_params(self, log_content: str) -> Dict[str, Any]:
        """Parse best params from log content"""
        # Find the best params line
        match = re.search(r"Best params: (\{.+?\})", log_content, re.DOTALL)
        if not match:
            return {}

        params_str = match.group(1)

        # Extract key-value pairs
        params = {}

        # Parse each parameter
        param_pattern = r"'(\w+)':\s*([^,}]+)"
        for key, value in re.findall(param_pattern, params_str):
            # Convert numpy types
            value = value.strip()

            if 'np.int64(' in value:
                params[key] = int(re.search(r'np\.int64\((\d+)\)', value).group(1))
            elif 'np.True_' in value:
                params[key] = True
            elif 'np.False_' in value:
                params[key] = False
            elif 'np.str_' in value:
                params[key] = re.search(r"np\.str_\('([^']+)'\)", value).group(1)
            elif value in ['True', 'true']:
                params[key] = True
            elif value in ['False', 'false']:
                params[key] = False
            else:
                try:
                    # Try as float first
                    if '.' in value or 'e-' in value or 'e+' in value:
                        params[key] = float(value)
                    else:
                        params[key] = int(value)
                except ValueError:
                    # Keep as string
                    params[key] = value.strip("'\"")

        return params

    def _find_training_function(self, model_name: str, log_content: str) -> Optional[str]:
        """Find the training function JSON file for this model"""
        # Try to find in log content
        json_match = re.search(
            r"generated_training_functions/training_function_[^.]+\.json",
            log_content
        )
        if json_match:
            json_path = json_match.group(0)
            if Path(json_path).exists():
                return json_path

        # Search by model name similarity
        for json_file in self.generated_functions_dir.glob("training_function_*.json"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    if data.get('model_name', '').strip() == model_name.strip():
                        return str(json_file)
            except Exception:
                continue

        return None

    def execute_final_training(self, run_info: Dict[str, Any]) -> Dict[str, Any]:
        """Execute final training for an incomplete run"""
        logger.info("=" * 80)
        logger.info(f"🚀 Resuming Final Training")
        logger.info(f"Model: {run_info['model_name']}")
        logger.info(f"Log: {run_info['log_file']}")
        logger.info(f"Training Function: {run_info['training_function_path']}")
        logger.info("=" * 80)

        # Add project root to path if needed
        import sys
        project_root = Path(__file__).parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        # Import necessary modules
        from adapters.universal_converter import convert_to_torch_dataset
        from _models.training_function_executor import training_executor
        from data_splitting import get_current_splits

        # Aggressive memory cleanup before starting
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.info("🧹 GPU memory cleaned")

        # Get device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {device}")

        # Load training function
        with open(run_info['training_function_path'], 'r') as f:
            training_data = json.load(f)

        # Get data splits - if not available, need to load original data
        try:
            splits = get_current_splits()
            logger.info(f"Using cached splits - Train: {splits.X_train.shape}, Val: {splits.X_val.shape}, Test: {splits.X_test.shape}")
        except ValueError:
            # Splits not created yet - need to load data from log
            logger.warning("Data splits not found - loading original data from log...")
            import numpy as np

            # Extract dataset path from log file
            data_folder = None
            with open(run_info['log_file'], 'r') as f:
                for line in f:
                    if 'Starting real data processing from' in line:
                        # Extract: "Starting real data processing from data/dataset3/ directory"
                        match = re.search(r'from\s+([^\s]+)\s+directory', line)
                        if match:
                            data_folder = Path(match.group(1))
                            break

            if data_folder is None:
                # Fallback to config
                from config import config
                data_folder = Path("data") / Path(config.data_folder).name
                logger.warning(f"Could not find dataset in log, using config: {data_folder}")

            logger.info(f"Loading data from: {data_folder}")

            X_file = data_folder / "X.npy"
            y_file = data_folder / "y.npy"

            if not X_file.exists() or not y_file.exists():
                raise FileNotFoundError(
                    f"Data files not found in {data_folder}. "
                    f"Expected: {X_file} and {y_file}"
                )

            X = np.load(X_file)
            y = np.load(y_file)

            logger.info(f"Loaded data: X={X.shape}, y={y.shape}")

            # Create centralized splits with same random seed as main pipeline (42)
            from data_splitting import create_consistent_splits
            create_consistent_splits(X, y, test_size=0.2, val_size=0.2, random_state=42)

            splits = get_current_splits()
            logger.info(f"Created splits - Train: {splits.X_train.shape}, Val: {splits.X_val.shape}, Test: {splits.X_test.shape}")

        # Create datasets with standardization
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

        # Extract tensors
        X_train_final = train_dataset.X
        y_train_final = train_dataset.y
        X_test_tensor = test_dataset.X
        y_test_tensor = test_dataset.y

        # Delete intermediate datasets and splits to free memory before training
        del train_dataset, val_dataset, test_dataset, splits
        gc.collect()
        logger.info("Deleted intermediate datasets to free memory")

        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = training_data['model_name']
        subfolder_name = f"{timestamp}_{model_name}_resumed"
        trained_models_dir = Path("trained_models") / subfolder_name
        trained_models_dir.mkdir(parents=True, exist_ok=True)

        # Get best params
        best_params = run_info['best_params']
        logger.info(f"Best params: {best_params}")

        # Execute training
        logger.info("Starting final training execution...")
        try:
            trained_model, training_metrics = training_executor.execute_training_function(
                training_data,
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
            elif 'best_val_acc' in training_metrics:
                performance_metric = training_metrics['best_val_acc']
            else:
                performance_metric = 0.0

            final_metrics = {
                'acc': performance_metric,
                'macro_f1': training_metrics.get('macro_f1', None)
            }

            logger.info(f"✅ Final test metrics: {final_metrics}")

            # Save model and test tensors (format 1: .pt files in subfolder)
            model_path = trained_models_dir / "model.pt"
            X_test_path = trained_models_dir / "X_test.pt"
            y_test_path = trained_models_dir / "y_test.pt"

            torch.save(trained_model.state_dict(), model_path)
            torch.save(X_test_tensor, X_test_path)
            torch.save(y_test_tensor, y_test_path)

            logger.info(f"💾 Model saved to: {trained_models_dir}")

            # Save model with metadata (format 2: .pth file in trained_models root)
            model_filename = f"best_model_{model_name}_{timestamp}.pth"
            model_save_path = Path("trained_models") / model_filename

            model_data = {
                'model_state_dict': trained_model.state_dict(),
                'model_name': model_name,
                'final_metrics': final_metrics,
                'best_hyperparameters': best_params,
                'timestamp': timestamp,
                'model_code': training_data.get('training_code'),
                'training_json_path': run_info['training_function_path']
            }

            torch.save(model_data, model_save_path)
            logger.info(f"💾 Model with metadata saved to: {model_save_path}")

            # Save results summary
            results = {
                'model_name': model_name,
                'timestamp': timestamp,
                'best_params': best_params,
                'final_metrics': final_metrics,
                'training_metrics': training_metrics,
                'original_log': run_info['log_file'],
                'training_function': run_info['training_function_path']
            }

            results_path = trained_models_dir / "training_results.json"
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)

            return results

        except Exception as e:
            logger.error(f"❌ Final training failed: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e)}

    def resume_all(self):
        """Resume all incomplete runs"""
        incomplete_runs = self.find_incomplete_runs()

        if not incomplete_runs:
            logger.info("✅ No incomplete runs found")
            return

        logger.info(f"Found {len(incomplete_runs)} incomplete run(s)")

        for i, run_info in enumerate(incomplete_runs, 1):
            logger.info(f"\n{'='*80}")
            logger.info(f"Processing run {i}/{len(incomplete_runs)}")
            logger.info(f"{'='*80}\n")

            try:
                results = self.execute_final_training(run_info)
                if 'error' not in results:
                    logger.info(f"✅ Successfully completed final training for {run_info['model_name']}")
                else:
                    logger.error(f"❌ Failed to complete final training for {run_info['model_name']}")
            except Exception as e:
                logger.error(f"❌ Error processing run: {e}")
                import traceback
                traceback.print_exc()

    def resume_specific(self, log_file: str):
        """Resume a specific run by log filename"""
        log_path = self.logs_dir / log_file
        if not log_path.exists():
            logger.error(f"Log file not found: {log_path}")
            return

        run_info = self._analyze_log_file(log_path)
        if not run_info:
            logger.error(f"Could not analyze log file or run is already complete")
            return

        results = self.execute_final_training(run_info)
        return results


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Resume final training from incomplete BO runs")
    parser.add_argument('--all', action='store_true', help="Resume all incomplete runs")
    parser.add_argument('--log', type=str, help="Resume specific log file (e.g., 2025-10-13_10-57-10.log)")
    parser.add_argument('--list', action='store_true', help="List all incomplete runs")

    args = parser.parse_args()

    resumer = FinalTrainingResumer()

    if args.list:
        incomplete_runs = resumer.find_incomplete_runs()
        if not incomplete_runs:
            print("✅ No incomplete runs found")
        else:
            print(f"\nFound {len(incomplete_runs)} incomplete run(s):\n")
            for i, run in enumerate(incomplete_runs, 1):
                print(f"{i}. Log: {Path(run['log_file']).name}")
                print(f"   Model: {run['model_name']}")
                print(f"   Timestamp: {run['log_timestamp']}")
                print()

    elif args.all:
        resumer.resume_all()

    elif args.log:
        resumer.resume_specific(args.log)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
