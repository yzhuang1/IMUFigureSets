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
        
        # Initialize OpenAI client
        if self.api_key:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
        else:
            self.client = None
            logger.warning("OpenAI API key not configured, will use default model selection")
    
    def _create_prompt(self, data_profile: Dict[str, Any]) -> str:
        """Create prompt for model selection"""
        return prompt_loader.format_model_selection_prompt(data_profile)
    
    def _call_openai_api(self, prompt: str) -> str:
        """Call OpenAI API using official library"""
        if not self.client:
            raise ValueError("OpenAI client not initialized. Please check your API key configuration.")
        
        try:
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
        
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            raise
    
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
            
            # Return default recommendation
            return ModelRecommendation(
                model_name="MLP",
                model_type="general",
                architecture="Multi-layer Perceptron",
                input_shape=(data_profile.get("feature_count", 10),),
                reasoning="Parsing failed, using default MLP model",
                confidence=0.1,
                hyperparameters={"hidden_size": 64, "num_layers": 2}
            )
    
    def select_model(self, data_profile: Dict[str, Any]) -> ModelRecommendation:
        """
        Select the most suitable model based on data characteristics
        
        Args:
            data_profile: Data feature profile
        
        Returns:
            ModelRecommendation: Model recommendation result
        """
        try:
            if not self.client:
                logger.warning("OpenAI client not available, using default model selection")
                return self._get_default_recommendation(data_profile)
            
            prompt = self._create_prompt(data_profile)
            response = self._call_openai_api(prompt)
            recommendation = self._parse_recommendation(response)
            
            logger.info(f"AI recommended model: {recommendation.model_name} (confidence: {recommendation.confidence:.2f})")
            logger.info(f"Recommendation reason: {recommendation.reasoning}")
            
            return recommendation
        
        except Exception as e:
            logger.error(f"Model selection failed: {e}")
            return self._get_default_recommendation(data_profile)
    
    def _get_default_recommendation(self, data_profile: Dict[str, Any]) -> ModelRecommendation:
        """Get default model recommendation (when API is unavailable)"""
        data_type = data_profile.get("data_type", "unknown")
        is_image = data_profile.get("is_image", False)
        is_sequence = data_profile.get("is_sequence", False)
        is_tabular = data_profile.get("is_tabular", False)
        
        if is_image:
            return ModelRecommendation(
                model_name="SmallCNN",
                model_type="image",
                architecture="Small Convolutional Neural Network",
                input_shape=(3, 32, 32),  # Default image size
                reasoning="Detected image data, recommend using convolutional neural network",
                confidence=0.8,
                hyperparameters={"hidden_size": 64, "num_layers": 2}
            )
        elif is_sequence:
            return ModelRecommendation(
                model_name="TinyCNN1D",
                model_type="sequence",
                architecture="1D Convolutional Neural Network",
                input_shape=(data_profile.get("feature_count", 10),),
                reasoning="Detected sequence data, recommend using 1D convolutional neural network",
                confidence=0.8,
                hyperparameters={"hidden_size": 64, "num_layers": 2}
            )
        elif is_tabular:
            return ModelRecommendation(
                model_name="TabMLP",
                model_type="tabular",
                architecture="Multi-layer Perceptron for Tabular Data",
                input_shape=(data_profile.get("feature_count", 10),),
                reasoning="Detected tabular data, recommend using multi-layer perceptron",
                confidence=0.8,
                hyperparameters={"hidden_size": 64, "num_layers": 2}
            )
        else:
            return ModelRecommendation(
                model_name="MLP",
                model_type="general",
                architecture="Multi-layer Perceptron",
                input_shape=(data_profile.get("feature_count", 10),),
                reasoning="Unknown data type, using general multi-layer perceptron",
                confidence=0.5,
                hyperparameters={"hidden_size": 64, "num_layers": 2}
            )

# Global model selector instance
ai_model_selector = AIModelSelector()

def select_model_for_data(data_profile: Dict[str, Any]) -> ModelRecommendation:
    """
    Convenience function: Select the most suitable model for data
    
    Args:
        data_profile: Data feature profile
    
    Returns:
        ModelRecommendation: Model recommendation result
    """
    return ai_model_selector.select_model(data_profile)
