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

from models.ai_code_generator import generate_training_code_for_data, ai_code_generator
from models.training_function_executor import training_executor, BO_TrainingObjective
from evaluation.evaluate import evaluate_model
from visualization import generate_bo_charts, create_charts_folder
from config import config

logger = logging.getLogger(__name__)

class CodeGenerationPipelineOrchestrator:
    """Orchestrates ML pipeline with AI-generated training functions"""
    
    def __init__(self, data_profile: Dict[str, Any], max_model_attempts: Optional[int] = None):
        self.data_profile = data_profile
        self.max_model_attempts = max_model_attempts or config.max_model_selection_attempts
        self.generated_functions = []
        self.pipeline_history = []
        
        # Performance thresholds
        self.min_acceptable_f1 = 0.3
        self.min_acceptable_accuracy = 0.4
        
        logger.info(f"Code generation pipeline orchestrator initialized with max {self.max_model_attempts} attempts")
    
    def run_complete_pipeline(
        self,
        X, y,
        device: str,
        input_shape: tuple,
        num_classes: int,
        **kwargs
    ) -> Tuple[torch.nn.Module, Dict[str, Any]]:
        """
        Run complete pipeline with AI code generation
        
        Flow: Code Generation → Save to JSON → BO → Execute Training → Evaluation
        
        Args:
            X, y: Training data
            device: Training device
            input_shape: Model input shape
            num_classes: Number of classes
            **kwargs: Additional parameters
            
        Returns:
            Tuple[model, results]: Best model and pipeline results
        """
        logger.info("Starting code generation pipeline execution")
        logger.info("Flow: AI Code Generation → JSON Storage → BO → Training Execution → Evaluation")
        
        best_model = None
        best_results = None
        best_score = -float('inf')
        
        # Check if we should disable progress bars due to verbose logging
        logger_level = logging.getLogger().getEffectiveLevel()
        disable_progress = logger_level <= logging.INFO
        
        # Progress bar for pipeline attempts
        pipeline_pbar = tqdm(range(self.max_model_attempts), 
                            desc="🚀 ML Pipeline", 
                            unit="attempt",
                            position=0,
                            leave=True,
                            disable=disable_progress,
                            file=sys.stdout,
                            dynamic_ncols=True,
                            ascii=True)
        
        for attempt in pipeline_pbar:
            logger.info(f"\n{'='*60}")
            logger.info(f"PIPELINE ATTEMPT {attempt + 1}/{self.max_model_attempts}")
            logger.info(f"{'='*60}")
            
            try:
                # STEP 1: AI Code Generation
                if disable_progress:
                    print(f"\n🤖 AI Code Generation [{attempt + 1}/{self.max_model_attempts}]")
                else:
                    pipeline_pbar.set_description(f"🤖 AI Code Generation [{attempt + 1}/{self.max_model_attempts}]")
                logger.info("🤖 STEP 1: AI Training Code Generation")
                code_rec = self._generate_training_code(input_shape, num_classes)
                
                # STEP 2: Save to JSON
                if disable_progress:
                    print(f"💾 Saving Function [{attempt + 1}/{self.max_model_attempts}]")
                else:
                    pipeline_pbar.set_description(f"💾 Saving Function [{attempt + 1}/{self.max_model_attempts}]")
                logger.info("💾 STEP 2: Save Training Function to JSON")
                json_filepath = self._save_training_function(code_rec)
                
                # STEP 3: Bayesian Optimization
                if disable_progress:
                    print(f"🔍 Bayesian Optimization [{attempt + 1}/{self.max_model_attempts}]")
                else:
                    pipeline_pbar.set_description(f"🔍 Bayesian Optimization [{attempt + 1}/{self.max_model_attempts}]")
                logger.info("🔍 STEP 3: Bayesian Optimization")
                bo_results = self._run_bayesian_optimization(X, y, device, code_rec)
                
                # STEP 4: Final Training with Optimized Parameters
                pipeline_pbar.set_description(f"🚀 Final Training [{attempt + 1}/{self.max_model_attempts}]")
                logger.info("🚀 STEP 4: Final Training Execution")
                final_model, training_results = self._execute_final_training(
                    X, y, device, json_filepath, bo_results
                )
                
                # STEP 5: Performance Analysis
                if disable_progress:
                    print(f"📊 Performance Analysis [{attempt + 1}/{self.max_model_attempts}]")
                else:
                    pipeline_pbar.set_description(f"📊 Performance Analysis [{attempt + 1}/{self.max_model_attempts}]")
                logger.info("📊 STEP 5: Performance Analysis")
                performance_score = training_results['final_metrics']['macro_f1']
                is_acceptable = self._is_performance_acceptable(training_results['final_metrics'])
                
                # Record attempt
                attempt_record = {
                    'attempt': attempt + 1,
                    'model_name': code_rec.model_name,
                    'json_filepath': json_filepath,
                    'bo_best_params': bo_results['best_params'],
                    'bo_best_score': bo_results['best_value'],
                    'final_metrics': training_results['final_metrics'],
                    'performance_score': performance_score,
                    'acceptable': is_acceptable,
                    'reasoning': code_rec.reasoning,
                    'confidence': code_rec.confidence
                }
                self.pipeline_history.append(attempt_record)
                
                logger.info(f"Generated Model: {code_rec.model_name}")
                logger.info(f"BO Score: {bo_results['best_value']:.4f}")
                logger.info(f"Final Score: {performance_score:.4f}")
                logger.info(f"Acceptable: {is_acceptable}")
                
                # Update best if this is better
                if performance_score > best_score:
                    best_model = final_model
                    best_results = training_results
                    best_score = performance_score
                    logger.info(f"🏆 New best model: {code_rec.model_name} (score: {performance_score:.4f})")
                
                # STEP 6: Decision - Accept or Continue
                if is_acceptable:
                    logger.info("✅ Model performance is acceptable! Stopping pipeline.")
                    logger.info(f"Final model: {code_rec.model_name}")
                    logger.info(f"Final metrics: {training_results['final_metrics']}")
                    break
                else:
                    logger.info("❌ Model performance below threshold. Generating new training function...")
                    if attempt < self.max_model_attempts - 1:
                        logger.info(f"Will generate new training function for attempt {attempt + 2}")
                    
            except Exception as e:
                logger.error(f"Pipeline attempt {attempt + 1} failed: {e}")
                attempt_record = {
                    'attempt': attempt + 1,
                    'model_name': f"failed_attempt_{attempt + 1}",
                    'error': str(e),
                    'acceptable': False,
                    'performance_score': 0.0
                }
                self.pipeline_history.append(attempt_record)
                continue
        
        # Close pipeline progress bar
        pipeline_pbar.close()
        
        if best_model is None:
            raise RuntimeError("Pipeline failed: No successful code generation after all attempts")
        
        # Enhance results with pipeline history
        best_results['pipeline_history'] = self.pipeline_history
        best_results['total_attempts'] = len(self.pipeline_history)
        best_results['successful_attempts'] = len([a for a in self.pipeline_history if a.get('acceptable', False)])
        
        # Clean completion message
        print(f"\n🎉 CODE GENERATION PIPELINE COMPLETE!")
        print(f"📊 Best model: {best_results.get('model_name', 'Unknown')}")
        print(f"🏆 Best score: {best_score:.4f}")
        print(f"🔄 Total attempts: {len(self.pipeline_history)}")
        
        # Also log for records
        logger.info(f"CODE GENERATION PIPELINE COMPLETE!")
        logger.info(f"Best model: {best_results.get('model_name', 'Unknown')}")
        logger.info(f"Best score: {best_score:.4f}")
        logger.info(f"Total attempts: {len(self.pipeline_history)}")
        
        return best_model, best_results
    
    def _generate_training_code(self, input_shape: tuple, num_classes: int):
        """Generate training code using AI"""
        
        # Modify data profile to encourage variety if we've tried models before
        modified_profile = self.data_profile.copy()
        if len(self.pipeline_history) > 0:
            tried_models = [a.get('model_name', '') for a in self.pipeline_history]
            modified_profile['previous_attempts'] = tried_models
        
        code_rec = generate_training_code_for_data(
            modified_profile, input_shape, num_classes
        )
        
        logger.info(f"Generated training function: {code_rec.model_name}")
        logger.info(f"Reasoning: {code_rec.reasoning}")
        logger.info(f"BO parameters: {code_rec.bo_parameters}")
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
        
        # Use subset for BO
        subset_size = min(5000, len(X))
        import random
        indices = random.sample(range(len(X)), subset_size)
        X_subset = torch.tensor(X[indices], dtype=torch.float32)
        y_subset = torch.tensor(y[indices], dtype=torch.long)
        
        logger.info(f"BO dataset size: {len(X_subset)} samples")
        logger.info(f"BO will optimize: {code_rec.bo_parameters}")
        
        # Create training data for objective function
        training_data = {
            "model_name": code_rec.model_name,
            "training_code": code_rec.training_code,
            "hyperparameters": code_rec.hyperparameters,
            "bo_parameters": code_rec.bo_parameters
        }
        
        # Create BO objective function
        objective_func = BO_TrainingObjective(training_data, X_subset, y_subset, device)
        
        # Set up search space
        from bo.run_bo import BayesianOptimizer
        from skopt.space import Real, Integer, Categorical
        
        search_space = []
        for param in code_rec.bo_parameters:
            if param == 'lr':
                search_space.append(Real(1e-4, 1e-2, prior='log-uniform', name='lr'))
            elif param == 'epochs':
                search_space.append(Integer(5, 15, name='epochs'))
            elif param == 'batch_size':
                search_space.append(Categorical([32, 64, 128], name='batch_size'))
            elif param == 'hidden_size':
                search_space.append(Integer(64, 256, name='hidden_size'))
            elif param == 'dropout':
                search_space.append(Real(0.1, 0.5, name='dropout'))
            elif param == 'num_layers':
                search_space.append(Integer(1, 3, name='num_layers'))
        
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
                print(f"  BO Trial {trial + 1}/{config.max_bo_trials}: {hparams}")
            
            try:
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
                
            except Exception as e:
                logger.error(f"BO Trial {trial + 1} failed: {e}")
                bo_optimizer.observe(hparams, 0.0)
                pbar.set_postfix_str(f"{best_score:.4f} (Failed)")
                continue
        
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
        logger.info(f"Best params: {bo_results['best_params']}")
        
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
        
        # Split data for final training
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        
        # Convert to tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)
        
        # Split training data into train/val for the training function
        X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
            X_train_tensor.numpy(), y_train_tensor.numpy(), 
            test_size=0.2, random_state=42, stratify=y_train_tensor.numpy()
        )
        
        X_train_final = torch.tensor(X_train_final, dtype=torch.float32)
        y_train_final = torch.tensor(y_train_final, dtype=torch.long)
        X_val_final = torch.tensor(X_val_final, dtype=torch.float32)
        y_val_final = torch.tensor(y_val_final, dtype=torch.long)
        
        # Load training function
        training_data = training_executor.load_training_function(json_filepath)
        
        # Get best hyperparameters from BO
        best_params = bo_results['best_params']
        
        # Execute training with optimized parameters
        logger.info(f"Executing final training with optimized params: {best_params}")
        
        trained_model, training_metrics = training_executor.execute_training_function(
            training_data,
            X_train_final, y_train_final,
            X_val_final, y_val_final,
            device=device,
            **best_params
        )
        
        # Evaluate on held-out test set
        from torch.utils.data import TensorDataset, DataLoader
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        final_metrics = evaluate_model(trained_model, test_loader, device)
        
        logger.info(f"Final test metrics: {final_metrics}")
        
        results = {
            'model': trained_model,
            'model_name': training_data['model_name'],
            'json_filepath': json_filepath,
            'bo_results': bo_results,
            'training_metrics': training_metrics,
            'final_metrics': final_metrics,
            'train_test_split': {
                'train_size': len(X_train),
                'test_size': len(X_test)
            },
            'optimized_hyperparameters': best_params
        }
        
        return trained_model, results
    
    def _is_performance_acceptable(self, metrics: Dict[str, float]) -> bool:
        """Determine if model performance is acceptable"""
        f1_score = metrics.get('macro_f1', 0.0)
        accuracy = metrics.get('acc', 0.0)
        
        acceptable_f1 = f1_score >= self.min_acceptable_f1
        acceptable_acc = accuracy >= self.min_acceptable_accuracy
        
        logger.info(f"Performance check:")
        logger.info(f"  F1: {f1_score:.4f} ≥ {self.min_acceptable_f1} ? {acceptable_f1}")
        logger.info(f"  Acc: {accuracy:.4f} ≥ {self.min_acceptable_accuracy} ? {acceptable_acc}")
        
        return acceptable_f1 and acceptable_acc
    
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
        
        successful_attempts = [a for a in self.pipeline_history if a.get('acceptable', False)]
        failed_attempts = [a for a in self.pipeline_history if not a.get('acceptable', False)]
        
        return {
            'total_attempts': len(self.pipeline_history),
            'successful_attempts': len(successful_attempts),
            'failed_attempts': len(failed_attempts),
            'best_score': max(a.get('performance_score', 0) for a in self.pipeline_history),
            'models_generated': [a.get('model_name') for a in self.pipeline_history],
            'final_success': len(successful_attempts) > 0,
            'pipeline_history': self.pipeline_history,
            'generated_functions': self.generated_functions
        }