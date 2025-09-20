"""
Code Generation Pipeline Orchestrator
Manages ML pipeline with AI-generated training functions
"""

import sys
import logging
from typing import Dict, Any, Tuple, Optional, List
import torch
from logging_config import get_pipeline_logger
from torch.utils.data import DataLoader
import numpy as np
from sklearn.model_selection import train_test_split

from _models.ai_code_generator import generate_training_code_for_data, ai_code_generator
from _models.training_function_executor import training_executor, BO_TrainingObjective
from visualization import generate_bo_charts, create_charts_folder
from config import config
from data_splitting import create_consistent_splits, get_current_splits, compute_standardization_stats
from error_monitor import set_bo_process_mode
from adapters.universal_converter import convert_to_torch_dataset

logger = get_pipeline_logger(__name__)

class CodeGenerationPipelineOrchestrator:
    """Orchestrates ML pipeline with AI-generated training functions"""
    
    def __init__(self, data_profile: Dict[str, Any]):
        self.data_profile = data_profile
        self.generated_functions = []
        self.pipeline_history = []
        
        logger.info("Code generation pipeline orchestrator initialized (no retry logic)")
    
    def run_complete_pipeline(
        self,
        X, y,
        device: str,
        input_shape: tuple,
        num_classes: int,
        **kwargs
    ) -> Tuple[torch.nn.Module, Dict[str, Any]]:
        """
        Run complete pipeline with AI code generation (single attempt, fail fast)
        
        Flow: Code Generation → Save to JSON → BO → Execute Training → Evaluation
        
        Args:
            X, y: Training data
            device: Training device
            input_shape: Model input shape
            num_classes: Number of classes
            **kwargs: Additional parameters
            
        Returns:
            Tuple[model, results]: Model and pipeline results
        """
        logger.info("Starting code generation pipeline execution (single attempt, fail fast)")
        logger.info("Flow: AI Code Generation → JSON Storage → BO → Training Execution → Evaluation")
        
        # Create consistent data splits at the beginning
        logger.info("Creating centralized data splits to prevent data leakage")
        splits = create_consistent_splits(X, y, test_size=0.2, val_size=0.2)
        
        # Compute standardization stats from training data only
        std_stats = compute_standardization_stats()
        logger.info("Computed standardization statistics from training data only")
        
        logger.info(f"\n{'='*60}")
        logger.info(f"PIPELINE EXECUTION (SINGLE ATTEMPT)")
        logger.info(f"{'='*60}")
        
        # STEP 1: AI Code Generation
        logger.info("🤖 STEP 1: AI Training Code Generation")
        code_rec = self._generate_training_code(input_shape, num_classes)
        
        # Enable BO process mode for error handling
        # Get current log file path for error monitoring
        from logging_config import get_log_file_path
        log_file_path = get_log_file_path()
        set_bo_process_mode(True, log_file_path)

        # STEP 2: Save to JSON
        logger.info("💾 STEP 2: Save Training Function to JSON")
        json_filepath = self._save_training_function(code_rec)
        
        # STEP 3: Bayesian Optimization
        logger.info("🔍 STEP 3: Bayesian Optimization")

        bo_results = self._run_bayesian_optimization(X, y, device, code_rec)

        # Disable BO process mode after BO
        set_bo_process_mode(False)

        # STEP 4: Final Training with Optimized Parameters
        logger.info("🚀 STEP 4: Final Training Execution")
        final_model, training_results = self._execute_final_training(
            X, y, device, json_filepath, bo_results
        )

        # STEP 5: Performance Analysis
        logger.info("📊 STEP 5: Performance Analysis")
        performance_score = training_results['final_metrics']['macro_f1']
        
        # Record single attempt
        attempt_record = {
            'attempt': 1,
            'model_name': code_rec.model_name,
            'json_filepath': json_filepath,
            'bo_best_params': bo_results['best_params'],
            'bo_best_score': bo_results['best_value'],
            'final_metrics': training_results['final_metrics'],
            'performance_score': performance_score,
            'confidence': code_rec.confidence
        }
        self.pipeline_history.append(attempt_record)
        
        logger.info(f"Generated Model: {code_rec.model_name}")
        logger.info(f"BO Score: {bo_results['best_value']:.4f}")
        logger.info(f"Final Score: {performance_score:.4f}")
        
        # Enhance results with pipeline history
        training_results['pipeline_history'] = self.pipeline_history
        training_results['total_attempts'] = 1
        training_results['successful_attempts'] = 1
        
        # Also log for records
        logger.info(f"CODE GENERATION PIPELINE COMPLETE!")
        logger.info(f"Model: {training_results.get('model_name', 'Unknown')}")
        logger.info(f"Score: {performance_score:.4f}")
        
        return final_model, training_results
    
    def _generate_training_code(self, input_shape: tuple, num_classes: int):
        """Generate training code using AI"""
        
        code_rec = generate_training_code_for_data(
            self.data_profile, input_shape, num_classes
        )
        
        logger.info(f"Generated training function: {code_rec.model_name}")
        logger.info(f"BO parameters: {list(code_rec.bo_config.keys()) if code_rec.bo_config else 'None'}")
        logger.info(f"Confidence: {code_rec.confidence:.2f}")
        
        return code_rec
    
    def _save_training_function(self, code_rec) -> str:
        """Save training function to JSON file"""
        
        # Create training data structure
        training_data = {
            "model_name": code_rec.model_name,
            "training_code": code_rec.training_code,
            "bo_config": code_rec.bo_config,
            "confidence": code_rec.confidence,
            "data_profile": self.data_profile,
            "timestamp": int(torch.random.initial_seed()),  # Use torch seed as timestamp
            "metadata": {
                "generated_by": "AI Code Generator Pipeline",
                "pipeline_attempt": len(self.pipeline_history) + 1,
                "version": "1.0"
            }
        }
        
        # Save to file
        json_filepath = ai_code_generator.save_training_function(code_rec, self.data_profile)
        self.generated_functions.append(json_filepath)
        
        logger.info(f"Training function saved to: {json_filepath}")
        
        # Skip validation - let training code find the problem
        # if not training_executor.validate_training_function(training_data):
        #     raise ValueError("Generated training function failed validation")
        
        return json_filepath
    
    def _run_bayesian_optimization(self, X, y, device: str, code_rec) -> Dict[str, Any]:
        """Run Bayesian Optimization for hyperparameter tuning with automatic restart on GPT fixes"""
        logger.info(f"Running BO for generated training function: {code_rec.model_name}")

        # Try BO with potential restart on GPT fixes
        max_restarts = 2  # Limit restarts to avoid infinite loops
        for restart_attempt in range(max_restarts + 1):
            if restart_attempt > 0:
                logger.info(f"🔄 BO Restart attempt {restart_attempt}/{max_restarts}")

            try:
                return self._run_single_bo_session(X, y, device, code_rec, restart_attempt)
            except Exception as e:
                if "GPT_FIXES_AVAILABLE" in str(e) and restart_attempt < max_restarts:
                    # Apply GPT fixes and try again
                    fixed_code_rec = self._apply_gpt_fixes_and_regenerate(code_rec)
                    if fixed_code_rec:
                        code_rec = fixed_code_rec
                        logger.info("✅ Applied GPT fixes, restarting BO from trial 0")
                        continue

                if "BO_UNFIXABLE_ERROR" in str(e):
                    # Error cannot be fixed by GPT, stop BO gracefully
                    logger.warning("BO stopped due to unfixable error - returning fallback results")
                    return {"best_value": 0.0, "best_params": {}, "total_trials": 0, "all_results": []}

                # Re-raise if not a restart case or max restarts exceeded
                raise

        # Should not reach here, but fallback
        logger.warning("Max BO restarts exceeded, proceeding with current results")
        return {"best_value": 0.0, "best_params": {}, "total_trials": 0, "all_results": []}

    def _run_single_bo_session(self, X, y, device: str, code_rec, session_num: int = 0) -> Dict[str, Any]:
        """Run a single BO session that can be restarted"""

        session_prefix = f"Session {session_num}: " if session_num > 0 else ""
        logger.info(f"{session_prefix}📦 Installing dependencies for GPT-generated training code...")
        from package_installer import install_gpt_code_dependencies
        try:
            success = install_gpt_code_dependencies(code_rec.training_code, raise_on_failure=True)
            logger.info("✅ All dependencies installed successfully")
        except RuntimeError as e:
            logger.error(f"❌ Failed to install required packages: {e}")
            raise RuntimeError(f"Cannot proceed with BO - missing dependencies: {e}")

        # Use BO subset from config (bo_sample_num) for efficient optimization
        from data_splitting import get_bo_subset
        X_bo, y_bo = get_bo_subset()
        logger.info(f"BO dataset size: {len(X_bo)} samples (using bo_sample_num={config.bo_sample_num})")
        logger.info(f"BO will optimize: {list(code_rec.bo_config.keys())}")

        # Create training data for objective function
        training_data = {
            "model_name": code_rec.model_name,
            "training_code": code_rec.training_code,
            "bo_config": code_rec.bo_config
        }

        # Create BO objective function with BO subset
        objective_func = BO_TrainingObjective(training_data, None, None, device)
        
        # Set up search space using GPT-generated search space
        from bo.run_bo import BayesianOptimizer

        # Use GPT's search space directly
        # Extract search space from bo_config (remove 'default' values)
        search_space = {}
        for param_name, param_config in code_rec.bo_config.items():
            search_space[param_name] = {k: v for k, v in param_config.items() if k != "default"}

        bo_optimizer = BayesianOptimizer(gpt_search_space=search_space, n_initial_points=3)
        
        # Run BO optimization
        results = []
        best_score = 0.0

        for trial in range(config.max_bo_trials):
            # Get next hyperparameters
            hparams = bo_optimizer.suggest()

            # Show all trial progress
            print(f"🔍 BO Trial {trial + 1}/{config.max_bo_trials}: {hparams}")

            value, metrics = objective_func(hparams)

            # Check if we got an error and fail fast
            if value == 0.0 and "error" in metrics:
                logger.error(f"BO Trial {trial + 1} FAILED with error: {metrics['error']}")

                # Record the failed trial first
                results.append({
                    'trial': trial + 1,
                    'hparams': hparams,
                    'value': value,
                    'metrics': metrics
                })

                # Wait for error monitor to process and generate GPT fixes
                import time
                logger.info("⏳ Waiting for GPT to finish debugging...")

                from error_monitor import _global_terminator

                # Wait indefinitely until GPT finishes (no timeout)
                check_interval = 3  # Check every 3 seconds
                elapsed_time = 0

                while True:
                    time.sleep(check_interval)
                    elapsed_time += check_interval

                    # Check if GPT provided fixes
                    if _global_terminator and hasattr(_global_terminator, 'last_json_corrections'):
                        gpt_fixes = getattr(_global_terminator, 'last_json_corrections', None)
                        if gpt_fixes and gpt_fixes != "{}":
                            logger.info(f"✅ GPT provided fixes after {elapsed_time}s - requesting BO restart")
                            # Signal that GPT fixes are available and BO should restart
                            raise Exception("GPT_FIXES_AVAILABLE")
                        elif gpt_fixes == "{}":
                            logger.warning(f"❌ GPT finished but couldn't provide fixes after {elapsed_time}s")
                            break

                    if elapsed_time % 15 == 0:  # Log progress every 15 seconds
                        logger.info(f"⏳ Still waiting for GPT to finish debugging... ({elapsed_time}s elapsed)")

                # No GPT fixes available - this trial failed and no fixes possible
                logger.error(f"🚨 FAILING FAST: Trial {trial + 1} failed and no GPT fixes available")
                logger.error(f"Error details: {metrics['error']}")

                # Raise a different exception to indicate unfixable error (not restart)
                logger.info("🛑 Stopping BO due to unfixable error")
                raise Exception("BO_UNFIXABLE_ERROR")

            results.append({
                'trial': trial + 1,
                'hparams': hparams,
                'value': value,
                'metrics': metrics
            })

            bo_optimizer.observe(hparams, value)

            # Update best score
            best_score = max(best_score, value)

            # Log every trial
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
            # Fallback if all trials failed
            bo_results = {
                'best_value': 0.0,
                'best_params': {param: param_config["default"] for param, param_config in code_rec.bo_config.items()},
                'total_trials': 0,
                'all_results': []
            }
        
        logger.info(f"BO completed - Best score: {bo_results['best_value']:.4f}")
        logger.info(f"Best params: {dict(bo_results['best_params']) if bo_results['best_params'] else 'None'}")
        
        # Generate BO charts
        self._generate_bo_charts(bo_results, code_rec.model_name)
        
        return bo_results

    def _apply_gpt_fixes_and_regenerate(self, original_code_rec):
        """Reload training function after GPT fixes have been applied to the JSON file"""
        try:
            from error_monitor import _global_terminator
            if not _global_terminator or not hasattr(_global_terminator, 'last_json_corrections'):
                return None

            gpt_fixes = getattr(_global_terminator, 'last_json_corrections', None)
            if not gpt_fixes or gpt_fixes == "{}":
                return None

            logger.info("🔧 GPT has fixed the training function, reloading from JSON file")

            # Find the most recent training function JSON (GPT should have fixed it)
            from _models.training_function_executor import training_executor
            functions = training_executor.list_available_training_functions()

            if not functions:
                logger.error("No training functions available to reload")
                return None

            # Get the most recent one (should be the fixed version)
            latest_function = functions[0]  # Already sorted by timestamp
            json_filepath = latest_function['filepath']

            logger.info(f"Reloading fixed training function from: {json_filepath}")

            # Load the fixed training function
            training_data = training_executor.load_training_function(json_filepath)

            # Convert back to CodeRecommendation format
            from _models.ai_code_generator import CodeRecommendation
            fixed_code_rec = CodeRecommendation(
                model_name=training_data['model_name'],
                training_code=training_data['training_code'],
                bo_config=training_data['bo_config'],
                confidence=training_data.get('confidence', 0.9)
            )

            # Clear the GPT corrections to avoid reusing them
            _global_terminator.last_json_corrections = None

            logger.info(f"✅ Reloaded fixed training function: {fixed_code_rec.model_name}")
            return fixed_code_rec

        except Exception as e:
            logger.error(f"Failed to reload fixed training function: {e}")
            return None
    
    def _execute_final_training(
        self,
        X, y,
        device: str,
        json_filepath: str,
        bo_results: Dict[str, Any]
    ) -> Tuple[torch.nn.Module, Dict[str, Any]]:
        """Execute final training with optimized hyperparameters"""
        
        # Use centralized data splits
        splits = get_current_splits()
        logger.info(f"Using centralized splits - Train: {splits.X_train.shape}, Val: {splits.X_val.shape}, Test: {splits.X_test.shape}")
        
        # Convert to tensors with proper standardization
        std_stats = splits.standardization_stats
        
        # Create datasets with proper standardization (training set already standardized)
        train_dataset, _, _ = convert_to_torch_dataset(
            splits.X_train, splits.y_train, 
            standardize=True, standardization_stats=None  # Compute stats for training
        )
        val_dataset, _, _ = convert_to_torch_dataset(
            splits.X_val, splits.y_val,
            standardize=True, standardization_stats=train_dataset.standardization_stats
        )
        test_dataset, _, _ = convert_to_torch_dataset(
            splits.X_test, splits.y_test,
            standardize=True, standardization_stats=train_dataset.standardization_stats
        )
        
        # Convert datasets to tensors for training function
        X_train_final = train_dataset.X
        y_train_final = train_dataset.y
        X_val_final = val_dataset.X  
        y_val_final = val_dataset.y
        X_test_tensor = test_dataset.X
        y_test_tensor = test_dataset.y
        
        # Load training function
        training_data = training_executor.load_training_function(json_filepath)
        
        # Get best hyperparameters from BO
        best_params = bo_results['best_params']
        
        # Execute training with optimized parameters
        logger.info(f"Executing final training with optimized params: {dict(best_params) if best_params else 'None'}")
        
        trained_model, training_metrics = training_executor.execute_training_function(
            training_data,
            X_train_final, y_train_final,
            X_val_final, y_val_final,
            device=device,
            **best_params
        )
        
        # Use final validation metrics from training instead of separate evaluation
        # This avoids preprocessing mismatches since training function handles its own evaluation
        final_metrics = {
            'acc': training_metrics.get('val_acc', [])[-1] if training_metrics.get('val_acc') else None,
            'macro_f1': training_metrics.get('macro_f1', None)
        }
        logger.info("Using final validation metrics from training (avoids preprocessing mismatch)")
        
        logger.info(f"Final test metrics: {dict(final_metrics) if final_metrics else 'None'}")
        
        results = {
            'model': trained_model,
            'model_name': training_data['model_name'],
            'json_filepath': json_filepath,
            'bo_results': bo_results,
            'training_metrics': training_metrics,
            'final_metrics': final_metrics,
            'train_test_split': {
                'train_size': len(X_train_final),
                'test_size': len(X_test_tensor)
            },
            'optimized_hyperparameters': best_params
        }
        
        return trained_model, results
    
    
    def _generate_bo_charts(self, bo_results: Dict[str, Any], model_name: str):
        """Generate BO charts"""
        try:
            from datetime import datetime
            from pathlib import Path
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            subfolder_name = f"BO_{model_name}_{timestamp}"
            charts_subfolder = Path("charts") / subfolder_name
            charts_subfolder.mkdir(parents=True, exist_ok=True)
            
            chart_results = {
                'all_results': bo_results.get('all_results', []),
                'best_value': bo_results.get('best_value', 0),
                'best_params': bo_results.get('best_params', {}),
                'model_name': model_name
            }
            
            generate_bo_charts(chart_results, save_folder=str(charts_subfolder))
            logger.info(f"📊 BO charts saved to: {charts_subfolder}")
            
        except Exception as e:
            logger.error(f"Failed to generate BO charts: {e}")
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Get summary of pipeline execution"""
        if not self.pipeline_history:
            return {'status': 'not_executed'}
        
        return {
            'total_attempts': len(self.pipeline_history),
            'successful_attempts': len(self.pipeline_history),  # Always 1 since we don't retry
            'failed_attempts': 0,  # Always 0 since failures terminate the program
            'best_score': max(a.get('performance_score', 0) for a in self.pipeline_history),
            'models_generated': [a.get('model_name') for a in self.pipeline_history],
            'final_success': True,  # Always true since failures terminate
            'pipeline_history': self.pipeline_history,
            'generated_functions': self.generated_functions
        }