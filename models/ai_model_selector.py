"""
AI Model Selector
Uses ChatGPT API to automatically select the most suitable neural network architecture based on data characteristics
"""

import json
import logging
from typing import Dict, Any, List, Optional, Tuple
import requests
import os
from dataclasses import dataclass

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
    
    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://api.openai.com/v1"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url
        self.model = "gpt-4"  # Use GPT-4 for better recommendations
        
        if not self.api_key:
            logger.warning("OPENAI_API_KEY not set, will use default model selection")
    
    def _create_prompt(self, data_profile: Dict[str, Any]) -> str:
        """Create prompt for model selection"""
        prompt = f"""
You are a machine learning expert who needs to recommend the most suitable neural network architecture based on data characteristics.

Data feature information:
{json.dumps(data_profile, indent=2, ensure_ascii=False)}

Please recommend the most suitable neural network architecture based on the following information:

1. Data feature analysis:
   - Data type: {data_profile.get('data_type', 'unknown')}
   - Data shape: {data_profile.get('shape', 'unknown')}
   - Sample count: {data_profile.get('sample_count', 0)}
   - Feature count: {data_profile.get('feature_count', 0)}
   - Has labels: {data_profile.get('has_labels', False)}
   - Label count: {data_profile.get('label_count', 0)}

2. Data characteristics:
   - Is sequence data: {data_profile.get('is_sequence', False)}
   - Is image data: {data_profile.get('is_image', False)}
   - Is tabular data: {data_profile.get('is_tabular', False)}

Please select the most suitable from the following predefined model types:

**Tabular Data Models:**
- TabMLP: Multi-layer perceptron for tabular data
- TabTransformer: Transformer-based tabular data model
- TabNet: Interpretable tabular data model

**Image Data Models:**
- SmallCNN: Small convolutional neural network
- ResNet: Residual network
- EfficientNet: Efficient convolutional neural network
- VisionTransformer: Transformer-based image model

**Sequence Data Models:**
- TinyCNN1D: 1D convolutional neural network
- LSTM: Long Short-Term Memory network
- GRU: Gated Recurrent Unit
- Transformer: Attention-based sequence model

**General Models:**
- MLP: General multi-layer perceptron
- AutoEncoder: Autoencoder

Please return the recommendation result in JSON format as follows:
{{
    "model_name": "Recommended model name",
    "model_type": "Model type (e.g., tabular, image, sequence, general)",
    "architecture": "Specific architecture description",
    "input_shape": [Array of input shapes],
    "reasoning": "Detailed explanation of recommendation reason",
    "confidence": 0.95,
    "hyperparameters": {{
        "hidden_size": 128,
        "num_layers": 3,
        "dropout": 0.2,
        "learning_rate": 0.001
    }}
}}

Please ensure the recommendation result is based on data characteristics and provide reasonable recommendation reasons.
"""
        return prompt
    
    def _call_openai_api(self, prompt: str) -> str:
        """Call OpenAI API"""
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not set")
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a professional machine learning expert skilled at recommending the most suitable neural network architecture based on data characteristics."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.3,
            "max_tokens": 1000
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
        
        except requests.exceptions.RequestException as e:
            logger.error(f"OpenAI API call failed: {e}")
            raise
        except KeyError as e:
            logger.error(f"API response format error: {e}")
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
            if not self.api_key:
                logger.warning("API key not set, using default model selection")
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
