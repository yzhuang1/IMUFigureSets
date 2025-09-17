"""
Code Generation Pipeline Orchestrator
Manages ML pipeline with AI-generated training functions
"""

import logging
import sys
from typing import Dict, Any, Tuple, Optional, List
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from _models.ai_code_generator import generate_training_code_for_data, ai_code_generator
from _models.training_function_executor import training_executor, BO_TrainingObjective
from evaluation.evaluate import evaluate_model
from visualization import generate_bo_charts, create_charts_folder
from config import config
from data_splitting import create_consistent_splits, get_current_splits, compute_standardization_stats
from error_monitor import enable_error_termination, disable_error_termination
from adapters.universal_converter import convert_to_torch_dataset

logger = logging.getLogger(__name__)

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
        
        # STEP 2: Save to JSON
        logger.info("💾 STEP 2: Save Training Function to JSON")
        json_filepath = self._save_training_function(code_rec)
        
        # STEP 3: Bayesian Optimization
        logger.info("🔍 STEP 3: Bayesian Optimization")

        # Enable error monitoring for BO process
        enable_error_termination(
        error_keywords=["ERROR"],
        ignore_patterns=[]
        )

        bo_results = self._run_bayesian_optimization(X, y, device, code_rec)

        # STEP 4: Final Training with Optimized Parameters
        logger.info("🚀 STEP 4: Final Training Execution")
        final_model, training_results = self._execute_final_training(
            X, y, device, json_filepath, bo_results
        )

        # Disable error monitoring before evaluation
        disable_error_termination()

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
            'reasoning': code_rec.reasoning,
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
        logger.info(f"Reasoning: {code_rec.reasoning}")
        logger.info(f"BO parameters: {list(code_rec.bo_parameters) if code_rec.bo_parameters else 'None'}")
        logger.info(f"Confidence: {code_rec.confidence:.2f}")
        
        return code_rec
    
    def _save_training_function(self, code_rec) -> str:
        """Save training function to JSON file"""
        
        # Create training data structure
        training_data = {
            "model_name": code_rec.model_name,
            "training_code": code_rec.training_code,
            "hyperparameters": code_rec.hyperparameters,
            "reasoning": code_rec.reasoning,
            "confidence": code_rec.confidence,
            "bo_parameters": code_rec.bo_parameters,
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
        
        # Validate the saved function
        if not training_executor.validate_training_function(training_data):
            raise ValueError("Generated training function failed validation")
        
        return json_filepath
    
    def _run_bayesian_optimization(self, X, y, device: str, code_rec) -> Dict[str, Any]:
        """Run Bayesian Optimization for hyperparameter tuning"""
        logger.info(f"Running BO for generated training function: {code_rec.model_name}")
        
        # Use full centralized dataset for BO
        logger.info(f"BO dataset size: {len(X)} samples (using full dataset)")
        logger.info(f"BO will optimize: {code_rec.bo_parameters}")
        
        # Create training data for objective function
        training_data = {
            "model_name": code_rec.model_name,
            "training_code": code_rec.training_code,
            "hyperparameters": code_rec.hyperparameters,
            "bo_parameters": code_rec.bo_parameters
        }
        
        # Create BO objective function with centralized splits (no subset)
        objective_func = BO_TrainingObjective(training_data, None, None, device)
        
        # Set up search space
        from bo.run_bo import BayesianOptimizer
        from skopt.space import Real, Integer, Categorical
        
        search_space = []
        for param in code_rec.bo_parameters:
            if param == 'lr':
                search_space.append(Real(1e-5, 1e-1, prior='log-uniform', name='lr'))
            elif param == 'epochs':
                search_space.append(Integer(3, 30, name='epochs'))
            elif param == 'batch_size':
                search_space.append(Categorical([8, 16, 32, 64, 128, 256], name='batch_size'))
            elif param == 'hidden_size':
                search_space.append(Integer(16, 512, name='hidden_size'))
            elif param == 'dropout':
                search_space.append(Real(0.0, 0.7, name='dropout'))
            elif param == 'num_layers':
                search_space.append(Integer(1, 5, name='num_layers'))
        
        # Initialize BO optimizer
        bo_optimizer = BayesianOptimizer(search_space=search_space, n_initial_points=3)
        
        # Run BO optimization
        results = []
        best_score = 0.0
        
        # Progress bar for BO trials  
        logger_level = logging.getLogger().getEffectiveLevel()
        disable_progress = logger_level <= logging.INFO
        
        pbar = tqdm(range(config.max_bo_trials), 
                   desc="🔍 Bayesian Optimization", 
                   unit="trial",
                   position=1,
                   leave=False,
                   disable=disable_progress,
                   file=sys.stdout,
                   dynamic_ncols=True,
                   ascii=True,
                   bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] Best: {postfix}")
        
        for trial in pbar:
            # Get next hyperparameters
            hparams = bo_optimizer.suggest()
            
            # Show trial progress when progress bars are disabled (every 3rd trial to reduce noise)
            if disable_progress and (trial + 1) % 3 == 0:
                print(f"🔍 BO Trial {trial + 1}/{config.max_bo_trials}: {hparams}")
            
            value, metrics = objective_func(hparams)
            results.append({
                'trial': trial + 1,
                'hparams': hparams,
                'value': value,
                'metrics': metrics
            })
            
            bo_optimizer.observe(hparams, value)
            
            # Update best score for progress bar
            best_score = max(best_score, value)
            pbar.set_postfix_str(f"{best_score:.4f}")
            
            # Only log every 3rd trial or best scores to reduce noise
            if (trial + 1) % 3 == 0 or value >= best_score:
                logger.info(f"BO Trial {trial + 1}: {hparams} -> {value:.4f}")
        
        pbar.close()
        
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
                'best_params': code_rec.hyperparameters,
                'total_trials': 0,
                'all_results': []
            }
        
        logger.info(f"BO completed - Best score: {bo_results['best_value']:.4f}")
        logger.info(f"Best params: {dict(bo_results['best_params']) if bo_results['best_params'] else 'None'}")
        
        # Generate BO charts
        self._generate_bo_charts(bo_results, code_rec.model_name)
        
        return bo_results
    
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
        
        # Evaluate on held-out test set
        from torch.utils.data import TensorDataset, DataLoader
        test_dataset_eval = TensorDataset(X_test_tensor, y_test_tensor)
        test_loader = DataLoader(test_dataset_eval, batch_size=64, shuffle=False)
        
        final_metrics = evaluate_model(trained_model, test_loader, device)
        
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