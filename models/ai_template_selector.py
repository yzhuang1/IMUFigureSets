"""
AI Template Selector
Uses GPT to select and configure pre-built model templates
"""

import json
import logging
import torch
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from openai import OpenAI
# Removed prompt_loader dependency
from config import config
from .model_templates import get_template_info, get_template_class

logger = logging.getLogger(__name__)

@dataclass
class TemplateRecommendation:
    """Template recommendation result"""
    template_name: str
    model_name: str
    config: Dict[str, Any]
    reasoning: str
    confidence: float
    bo_parameters: List[str]

class AITemplateSelector:
    """AI template selector using GPT"""
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or config.openai_api_key
        self.base_url = base_url or config.openai_base_url
        self.model = model or config.openai_model
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required but not configured.")
        
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.template_info = get_template_info()
    
    def _create_prompt(self, data_profile: Dict[str, Any], exclude_templates: Optional[List[str]] = None) -> str:
        """Create prompt for template selection"""
        exclude_text = ""
        if exclude_templates:
            exclude_text = f"\n\nEXCLUDED TEMPLATES (DO NOT SELECT): {', '.join(exclude_templates)}"
        
        prompt = f"""You are a machine learning expert. Select the best model template and configuration for this data.

DATA CHARACTERISTICS:
{json.dumps(data_profile, indent=2)}

AVAILABLE TEMPLATES:
{json.dumps(self.template_info, indent=2)}

TASK: Select the best template and provide configuration.

RESPONSE FORMAT (JSON only, no other text):
{{
    "template_name": "LSTM",
    "model_name": "ECGClassifier", 
    "config": {{
        "input_size": 2,
        "hidden_size": 128,
        "num_classes": 5,
        "dropout": 0.2,
        "num_layers": 2
    }},
    "reasoning": "LSTM chosen for ECG sequence data with temporal dependencies",
    "confidence": 0.95,
    "bo_parameters": ["hidden_size", "dropout", "num_layers", "lr", "batch_size", "epochs"]
}}

IMPORTANT:
- template_name must be one of: {list(self.template_info.keys())}
- config must include ALL required_params for the chosen template
- bo_parameters should include hyperparameters suitable for Bayesian Optimization
- Always include lr, batch_size, epochs in bo_parameters{exclude_text}"""

        return prompt
    
    def _call_openai_api(self, prompt: str) -> str:
        """Call OpenAI API"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a machine learning expert. Respond only with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=800
        )
        return response.choices[0].message.content
    
    def _parse_recommendation(self, response: str) -> TemplateRecommendation:
        """Parse API response as template recommendation"""
        try:
            # Clean response and extract JSON
            response = response.strip()
            if response.startswith('```json'):
                response = response[7:]
            if response.endswith('```'):
                response = response[:-3]
            
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON found in response")
            
            json_str = response[start_idx:end_idx]
            data = json.loads(json_str)
            
            # Validate required fields
            required_fields = ["template_name", "model_name", "config", "reasoning", "confidence", "bo_parameters"]
            for field in required_fields:
                if field not in data:
                    raise ValueError(f"Missing required field: {field}")
            
            # Validate template exists
            template_name = data["template_name"]
            if template_name not in self.template_info:
                raise ValueError(f"Unknown template: {template_name}")
            
            return TemplateRecommendation(
                template_name=template_name,
                model_name=data["model_name"],
                config=data["config"],
                reasoning=data["reasoning"],
                confidence=float(data["confidence"]),
                bo_parameters=data["bo_parameters"]
            )
        
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.error(f"Failed to parse template recommendation: {e}")
            logger.error(f"Original response: {response}")
            raise ValueError(f"Failed to parse AI template recommendation: {e}")
    
    def select_template(self, data_profile: Dict[str, Any], exclude_templates: Optional[List[str]] = None) -> TemplateRecommendation:
        """Select template based on data characteristics"""
        prompt = self._create_prompt(data_profile, exclude_templates)
        response = self._call_openai_api(prompt)
        recommendation = self._parse_recommendation(response)
        
        logger.info(f"AI selected template: {recommendation.template_name} -> {recommendation.model_name}")
        logger.info(f"Confidence: {recommendation.confidence:.2f}")
        logger.info(f"Reasoning: {recommendation.reasoning}")
        
        return recommendation

class TemplateValidator:
    """Validates template configurations"""
    
    def __init__(self):
        self.template_info = get_template_info()
    
    def validate_recommendation(self, recommendation: TemplateRecommendation, input_shape: tuple, num_classes: int) -> TemplateRecommendation:
        """Validate and fix template recommendation"""
        template_name = recommendation.template_name
        config = recommendation.config.copy()
        
        # Get template requirements
        template_info = self.template_info[template_name]
        required_params = template_info["required_params"]
        
        # Ensure required parameters are present
        if "input_size" in required_params:
            if len(input_shape) == 2:  # Sequence data (seq_len, features)
                config["input_size"] = input_shape[-1]  # Feature dimension (last dimension)
            elif len(input_shape) == 1:  # Tabular data (features,)
                config["input_size"] = input_shape[0]
            else:  # Image data or other
                config["input_size"] = input_shape[-1]
        
        if "num_classes" in required_params:
            config["num_classes"] = num_classes
        
        # Validate parameter ranges
        if "hidden_size" in config:
            config["hidden_size"] = max(8, min(512, int(config["hidden_size"])))
        
        if "dropout" in config:
            config["dropout"] = max(0.0, min(0.8, float(config["dropout"])))
        
        if "num_layers" in config:
            config["num_layers"] = max(1, min(6, int(config["num_layers"])))
        
        # Test instantiation
        try:
            template_class = get_template_class(template_name)
            model = template_class(**config)
            
            # Test forward pass
            if len(input_shape) == 2:  # Sequence data (seq_len, features)
                test_input = torch.randn(2, *input_shape)  # (batch, seq_len, features)
            elif len(input_shape) == 3:  # Image data (channels, height, width)
                test_input = torch.randn(2, *input_shape)  # (batch, channels, height, width)
            else:  # Tabular data (features,)
                test_input = torch.randn(2, *input_shape)
            
            with torch.no_grad():
                output = model(test_input)
                if output.shape[1] != num_classes:
                    raise ValueError(f"Output shape mismatch: {output.shape[1]} != {num_classes}")
            
            logger.info(f"Template validation successful: {template_name}")
            
        except Exception as e:
            logger.error(f"Template validation failed: {e}")
            raise ValueError(f"Template validation failed: {e}")
        
        # Update recommendation with validated config
        recommendation.config = config
        return recommendation

class FallbackSelector:
    """Rule-based fallback when AI selection fails"""
    
    def select_fallback_template(self, data_profile: Dict[str, Any], input_shape: tuple, num_classes: int) -> TemplateRecommendation:
        """Select template using rules when AI fails"""
        
        # Rule-based selection
        if data_profile.get('is_sequence', False) or len(input_shape) == 3:
            # Sequence data - prefer LSTM
            template_name = "LSTM"
            config = {
                "input_size": input_shape[-1],
                "hidden_size": 128,
                "num_classes": num_classes,
                "dropout": 0.2,
                "num_layers": 2
            }
            reasoning = "Fallback: LSTM selected for sequence data"
            
        elif data_profile.get('is_image', False):
            # Image-like data - use CNN1D
            template_name = "CNN1D"
            config = {
                "input_size": input_shape[-1],
                "num_classes": num_classes,
                "num_filters": 64,
                "dropout": 0.2
            }
            reasoning = "Fallback: CNN1D selected for image-like data"
            
        else:
            # Default to MLP
            template_name = "MLP"
            input_size = 1
            for dim in input_shape:
                input_size *= dim
            
            config = {
                "input_size": input_size,
                "num_classes": num_classes,
                "hidden_sizes": [256, 128, 64],
                "dropout": 0.2
            }
            reasoning = "Fallback: MLP selected for tabular data"
        
        return TemplateRecommendation(
            template_name=template_name,
            model_name=f"Fallback{template_name}",
            config=config,
            reasoning=reasoning,
            confidence=0.7,
            bo_parameters=["lr", "batch_size", "epochs", "dropout"]
        )

# Global instances
ai_template_selector = AITemplateSelector()
template_validator = TemplateValidator()
fallback_selector = FallbackSelector()

def select_template_for_data(data_profile: Dict[str, Any], input_shape: tuple, num_classes: int, 
                           exclude_templates: Optional[List[str]] = None) -> TemplateRecommendation:
    """
    Convenience function: Select template for data with fallback
    """
    try:
        # Try AI selection
        recommendation = ai_template_selector.select_template(data_profile, exclude_templates)
        # Validate and fix
        validated_recommendation = template_validator.validate_recommendation(recommendation, input_shape, num_classes)
        return validated_recommendation
        
    except Exception as e:
        logger.warning(f"AI template selection failed: {e}, using fallback")
        # Use rule-based fallback
        fallback_recommendation = fallback_selector.select_fallback_template(data_profile, input_shape, num_classes)
        validated_fallback = template_validator.validate_recommendation(fallback_recommendation, input_shape, num_classes)
        return validated_fallback