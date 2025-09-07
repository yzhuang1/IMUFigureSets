"""
AI-Enhanced Evaluation System
Uses ChatGPT API to analyze model performance and decide on re-selection
"""

import logging
import json
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import torch
from torch import nn

from evaluation.evaluate import evaluate_model
from models.ai_model_selector import select_model_for_data, ModelRecommendation
from config import config
from openai import OpenAI

logger = logging.getLogger(__name__)

class ModelPerformanceAnalysis:
    """Container for model performance analysis results"""
    
    def __init__(self, metrics: Dict[str, Any], analysis: str, 
                 decision: str, confidence: float, suggestions: List[str]):
        self.metrics = metrics
        self.analysis = analysis
        self.decision = decision  # "accept", "reject", "uncertain"
        self.confidence = confidence
        self.suggestions = suggestions
    
    def should_accept(self) -> bool:
        """Check if model should be accepted"""
        return self.decision.lower() == "accept"
    
    def should_reject(self) -> bool:
        """Check if model should be rejected"""
        return self.decision.lower() == "reject"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "metrics": self.metrics,
            "analysis": self.analysis,
            "decision": self.decision,
            "confidence": self.confidence,
            "suggestions": self.suggestions
        }

class AIEnhancedEvaluator:
    """AI-enhanced model evaluator with ChatGPT integration"""
    
    def __init__(self, data_profile: Dict[str, Any], model_recommendation: ModelRecommendation):
        self.data_profile = data_profile
        self.model_recommendation = model_recommendation
        
        if not config.is_openai_configured():
            raise ValueError("OpenAI API key is required but not configured. Please set OPENAI_API_KEY environment variable.")
        
        self.client = OpenAI(api_key=config.openai_api_key)
        
    def evaluate_with_ai_analysis(self, model: nn.Module, loader, device: str = "cpu") -> ModelPerformanceAnalysis:
        """
        Evaluate model and get AI analysis of performance
        
        Args:
            model: Trained model
            loader: Data loader
            device: Device for evaluation
            
        Returns:
            ModelPerformanceAnalysis: Complete analysis results
        """
        # Get standard metrics
        metrics = evaluate_model(model, loader, device)
        logger.info(f"Model metrics: {metrics}")
        
        # Get AI analysis (no fallback - fail if OpenAI fails)
        analysis = self._get_ai_performance_analysis(metrics)
        return analysis
    
    def _get_ai_performance_analysis(self, metrics: Dict[str, Any]) -> ModelPerformanceAnalysis:
        """Get AI analysis of model performance"""
        
        # Prepare context for AI
        context = {
            "data_profile": self.data_profile,
            "model_recommendation": {
                "model_name": self.model_recommendation.model_name,
                "model_type": self.model_recommendation.model_type,
                "architecture": self.model_recommendation.architecture,
                "reasoning": self.model_recommendation.reasoning,
                "confidence": self.model_recommendation.confidence
            },
            "performance_metrics": metrics
        }
        
        # Create evaluation prompt
        prompt = self._create_evaluation_prompt(context)
        
        # Call ChatGPT
        response = self.client.chat.completions.create(
            model=config.openai_model,
            messages=[
                {"role": "system", "content": self._get_evaluation_system_prompt()},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        # Parse response
        return self._parse_ai_response(response.choices[0].message.content, metrics)
    
    def _create_evaluation_prompt(self, context: Dict[str, Any]) -> str:
        """Create evaluation prompt for ChatGPT"""
        
        data_info = context["data_profile"]
        model_info = context["model_recommendation"]
        metrics = context["performance_metrics"]
        
        prompt = f"""
Please analyze the performance of a machine learning model and decide if we should accept it or try a different model.

**DATA CHARACTERISTICS:**
- Data type: {data_info.get('data_type', 'unknown')}
- Sample count: {data_info.get('sample_count', 'unknown')}
- Feature count: {data_info.get('feature_count', 'unknown')}
- Label count: {data_info.get('label_count', 'unknown')}
- Is balanced: {data_info.get('is_balanced', 'unknown')}
- Data complexity: {data_info.get('complexity_analysis', 'unknown')}

**SELECTED MODEL:**
- Model name: {model_info['model_name']}
- Model type: {model_info['model_type']}
- Architecture: {model_info['architecture']}
- Selection reasoning: {model_info['reasoning']}
- Selection confidence: {model_info['confidence']:.2f}

**PERFORMANCE RESULTS:**
- Accuracy: {metrics.get('acc', 'N/A'):.4f if metrics.get('acc') else 'N/A'}
- Macro F1 Score: {metrics.get('macro_f1', 'N/A'):.4f if metrics.get('macro_f1') else 'N/A'}

**ANALYSIS REQUIRED:**
1. Is this performance acceptable for this type of data and problem?
2. Does the model choice seem appropriate given the results?
3. Should we accept this model or try a different architecture?
4. What specific improvements could be made?

Please provide your analysis in the following JSON format:
{{
    "analysis": "Detailed analysis of the performance...",
    "decision": "accept|reject|uncertain",
    "confidence": 0.0-1.0,
    "suggestions": ["suggestion1", "suggestion2", ...]
}}
"""
        return prompt
    
    def _get_evaluation_system_prompt(self) -> str:
        """Get system prompt for evaluation"""
        return """
You are an expert machine learning engineer evaluating model performance. Your job is to:

1. Analyze model performance metrics in context of the data characteristics
2. Determine if the selected model architecture is appropriate
3. Decide whether to accept the current model or recommend trying a different one
4. Provide actionable suggestions for improvement

DECISION CRITERIA:
- "accept": Performance is good enough for the data type and problem complexity
- "reject": Performance is clearly inadequate, different model architecture needed
- "uncertain": Performance is borderline, could go either way

PERFORMANCE THRESHOLDS (guidelines):
- Accuracy/F1 > 0.85: Generally good
- Accuracy/F1 0.70-0.85: Acceptable depending on data complexity
- Accuracy/F1 0.50-0.70: Poor, likely needs different model
- Accuracy/F1 < 0.50: Very poor, definitely needs different model

Consider data characteristics:
- Simple tabular data: Higher thresholds expected
- Complex image data: Lower thresholds may be acceptable
- Imbalanced data: Focus more on F1 than accuracy
- Small datasets: Lower performance may be expected

Always respond in valid JSON format.
"""
    
    def _parse_ai_response(self, response_text: str, metrics: Dict[str, Any]) -> ModelPerformanceAnalysis:
        """Parse AI response into ModelPerformanceAnalysis"""
        try:
            # Try to extract JSON from response
            response_text = response_text.strip()
            if not response_text.startswith('{'):
                # Find JSON in response
                start = response_text.find('{')
                end = response_text.rfind('}') + 1
                if start != -1 and end > start:
                    response_text = response_text[start:end]
            
            parsed = json.loads(response_text)
            
            return ModelPerformanceAnalysis(
                metrics=metrics,
                analysis=parsed.get('analysis', 'No analysis provided'),
                decision=parsed.get('decision', 'uncertain'),
                confidence=float(parsed.get('confidence', 0.5)),
                suggestions=parsed.get('suggestions', [])
            )
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Failed to parse AI response: {e}")
            logger.debug(f"Response text: {response_text}")
            raise ValueError(f"Failed to parse AI evaluation response: {e}") from e

class IterativeModelSelector:
    """Iterative model selection with AI-enhanced evaluation"""
    
    def __init__(self, data_profile: Dict[str, Any], max_iterations: int = None):
        from config import config
        self.data_profile = data_profile
        self.max_iterations = max_iterations or config.max_model_selection_attempts
        self.attempt_history = []
        
        logger.info(f"IterativeModelSelector initialized with max_iterations={self.max_iterations}")
    
    def find_best_model(self, train_func, evaluate_func, **kwargs) -> Tuple[Any, ModelPerformanceAnalysis]:
        """
        Iteratively find the best model using AI evaluation
        
        Args:
            train_func: Function that trains a model given a recommendation
            evaluate_func: Function that evaluates a trained model
            **kwargs: Additional arguments
            
        Returns:
            Tuple[model, analysis]: Best model and its analysis
        """
        logger.info(f"Starting iterative model selection (max {self.max_iterations} iterations)")
        
        excluded_models = set()
        
        for iteration in range(self.max_iterations):
            logger.info(f"Model selection iteration {iteration + 1}/{self.max_iterations}")
            
            # Get AI recommendation (excluding previously tried models)
            recommendation = select_model_for_data(
                self.data_profile, 
                exclude_models=list(excluded_models)
            )
            
            logger.info(f"Selected model: {recommendation.model_name}")
            
            # Train model
            model = train_func(recommendation, **kwargs)
            
            # Evaluate with AI analysis
            evaluator = AIEnhancedEvaluator(self.data_profile, recommendation)
            analysis = evaluate_func(evaluator, model, **kwargs)
            
            # Record attempt
            attempt = {
                "iteration": iteration + 1,
                "recommendation": recommendation,
                "analysis": analysis,
                "model": model
            }
            self.attempt_history.append(attempt)
            
            logger.info(f"Analysis: {analysis.analysis}")
            logger.info(f"Decision: {analysis.decision} (confidence: {analysis.confidence:.2f})")
            
            if analysis.should_accept():
                logger.info("Model accepted!")
                return model, analysis
            elif analysis.should_reject():
                logger.info("Model rejected, trying different architecture")
                excluded_models.add(recommendation.model_name)
            else:
                logger.info("Uncertain result, continuing search")
                excluded_models.add(recommendation.model_name)
        
        # If we reach here, return the best attempt
        logger.warning("Max iterations reached, returning best attempt")
        best_attempt = max(self.attempt_history, 
                          key=lambda x: (x['analysis'].metrics.get('macro_f1', 0) or 0))
        
        return best_attempt['model'], best_attempt['analysis']
    
    def get_attempt_summary(self) -> Dict[str, Any]:
        """Get summary of all attempts"""
        return {
            "total_attempts": len(self.attempt_history),
            "attempts": [
                {
                    "iteration": attempt["iteration"],
                    "model_name": attempt["recommendation"].model_name,
                    "metrics": attempt["analysis"].metrics,
                    "decision": attempt["analysis"].decision,
                    "confidence": attempt["analysis"].confidence
                }
                for attempt in self.attempt_history
            ]
        }