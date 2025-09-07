"""
AI Model Selector
Uses ChatGPT API to automatically select the most suitable neural network architecture based on data characteristics
"""

import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from openai import OpenAI
from prompts import prompt_loader
from config import config

logger = logging.getLogger(__name__)

@dataclass
class ModelRecommendation:
    """Model recommendation result"""
    model_name: str
    model_type: str
    architecture: str
    input_shape: Tuple[int, ...]
    reasoning: str
    confidence: float
    hyperparameters: Dict[str, Any]

class AIModelSelector:
    """AI model selector"""
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or config.openai_api_key
        self.base_url = base_url or config.openai_base_url
        self.model = model or config.openai_model
        
        # Require OpenAI configuration
        if not self.api_key:
            raise ValueError("OpenAI API key is required but not configured. Please set OPENAI_API_KEY environment variable.")
        
        # Initialize OpenAI client
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
    
    def _create_prompt(self, data_profile: Dict[str, Any], exclude_models: Optional[List[str]] = None) -> str:
        """Create prompt for model selection"""
        return prompt_loader.format_model_selection_prompt(data_profile, exclude_models)
    
    def _call_openai_api(self, prompt: str) -> str:
        """Call OpenAI API using official library"""        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": prompt_loader.load_system_prompt()
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_tokens=1000
        )
        
        return response.choices[0].message.content
    
    def _parse_recommendation(self, response: str) -> ModelRecommendation:
        """Parse API response as model recommendation"""
        try:
            # Try to extract JSON part
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON format found in response")
            
            json_str = response[start_idx:end_idx]
            data = json.loads(json_str)
            
            return ModelRecommendation(
                model_name=data.get("model_name", "Unknown"),
                model_type=data.get("model_type", "general"),
                architecture=data.get("architecture", "Unknown"),
                input_shape=tuple(data.get("input_shape", [])),
                reasoning=data.get("reasoning", "No reasoning provided"),
                confidence=float(data.get("confidence", 0.5)),
                hyperparameters=data.get("hyperparameters", {})
            )
        
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.error(f"Failed to parse recommendation result: {e}")
            logger.error(f"Original response: {response}")
            raise ValueError(f"Failed to parse AI model recommendation: {e}") from e
    
    def select_model(self, data_profile: Dict[str, Any], exclude_models: Optional[List[str]] = None) -> ModelRecommendation:
        """
        Select the most suitable model based on data characteristics
        
        Args:
            data_profile: Data feature profile
            exclude_models: List of model names to exclude from selection
        
        Returns:
            ModelRecommendation: Model recommendation result
        """
        prompt = self._create_prompt(data_profile, exclude_models)
        response = self._call_openai_api(prompt)
        recommendation = self._parse_recommendation(response)
        
        logger.info(f"AI recommended model: {recommendation.model_name} (confidence: {recommendation.confidence:.2f})")
        logger.info(f"Recommendation reason: {recommendation.reasoning}")
        
        return recommendation
    

# Global model selector instance
ai_model_selector = AIModelSelector()

def select_model_for_data(data_profile: Dict[str, Any], exclude_models: Optional[List[str]] = None) -> ModelRecommendation:
    """
    Convenience function: Select the most suitable model for data
    
    Args:
        data_profile: Data feature profile
        exclude_models: List of model names to exclude from selection
    
    Returns:
        ModelRecommendation: Model recommendation result
    """
    return ai_model_selector.select_model(data_profile, exclude_models)
