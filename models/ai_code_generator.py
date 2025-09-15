"""
AI Code Generator
Uses GPT to generate complete training functions as executable code
"""

import json
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from openai import OpenAI
from config import config
from pathlib import Path
from models.literature_review import literature_review_generator, LiteratureReview

logger = logging.getLogger(__name__)

@dataclass
class CodeRecommendation:
    """Code generation result"""
    model_name: str
    training_code: str
    hyperparameters: Dict[str, Any]
    reasoning: str
    confidence: float
    bo_parameters: List[str]
    bo_search_space: Dict[str, Dict[str, Any]]

class AICodeGenerator:
    """AI code generator using GPT"""
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or config.openai_api_key
        self.base_url = base_url or config.openai_base_url
        self.model = model or config.openai_model
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required but not configured.")
        
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        
        # Create directory for storing generated code
        self.code_storage_dir = Path("generated_training_functions")
        self.code_storage_dir.mkdir(exist_ok=True)
    
    def _create_prompt(self, data_profile: Dict[str, Any], input_shape: tuple, num_classes: int, literature_review: Optional[LiteratureReview] = None) -> str:
        """Create enhanced prompt for GPT-5 code generation with literature review insights"""

        # Dataset context for better model selection
        dataset_context = ""
        if config.dataset_name and config.dataset_name != "Unknown Dataset":
            dataset_context = f"""

Dataset: {config.dataset_name}
Source: {config.dataset_source}"""

        # Base prompt
        # Get sample count from data profile (correct field name is 'sample_count')
        num_samples = data_profile.get('sample_count', data_profile.get('num_samples', 'unknown'))
        
        base_prompt = f"""Generate PyTorch training function for {num_classes}-class classification.

Data: {data_profile['data_type']}, shape {input_shape}, {num_samples} samples{dataset_context}"""

        # Add literature review insights if available
        if literature_review:
            base_prompt += f"""

LITERATURE REVIEW INSIGHTS:
Query: {literature_review.query}
Confidence: {literature_review.confidence:.2f}

Key Findings:
{chr(10).join('- ' + str(finding) for finding in literature_review.key_findings[:5] if finding)}

Recommended Approaches:
{chr(10).join('- ' + str(approach) for approach in literature_review.recommended_approaches[:3] if approach)}

Please consider these recent research insights when generating your training function."""

        prompt = base_prompt + f"""

Requirements:
- Function: train_model(X_train, y_train, X_val, y_val, device, **hyperparams)
- X_train, y_train, X_val, y_val are PyTorch tensors
- IMPORTANT: Only use pin_memory=True in DataLoader if tensors are on CPU. Check tensor.device.type == 'cpu' before enabling pin_memory
- Keep code simple - Bayesian Optimization will handle hyperparameter tuning
- Focus on core training loop, avoid complex scheduling/early stopping
- Build model from scratch, include basic training loop, return model and metrics
- Lightweight architecture (<256K parameters)
- Incorporate relevant insights from literature review if provided

Response JSON format:
{{
    "model_name": "ModelName",
    "training_code": "def train_model(...):\\n    # Complete PyTorch training code here",
    "hyperparameters": {{"lr": 0.001, "epochs": 10, "batch_size": 64}},
    "reasoning": "Brief explanation incorporating literature insights",
    "confidence": 0.9,
    "bo_parameters": ["lr", "batch_size", "epochs", "hidden_size", "dropout"],
    "bo_search_space": {{
        "lr": {{"type": "Real", "low": 1e-5, "high": 1e-1, "prior": "log-uniform"}},
        "batch_size": {{"type": "Categorical", "categories": [8, 16, 32, 64, 128]}},
        "epochs": {{"type": "Integer", "low": 5, "high": 50}},
        "hidden_size": {{"type": "Integer", "low": 32, "high": 512}},
        "dropout": {{"type": "Real", "low": 0.0, "high": 0.7}}
    }}
}}"""

        return prompt
    
    def _call_openai_api(self, prompt: str) -> str:
        """Call OpenAI API using responses.create method"""
        logger.info(f"Making API call to {self.model}")
        logger.info(f"Prompt length: {len(prompt)} characters")
        logger.info(f"Using API base URL: {self.base_url}")
        
        try:
            logger.info("Calling self.client.responses.create...")
            logger.info(f"Model parameter: {self.model}")
            logger.info(f"Input prompt preview: {prompt[:200]}...")
            
            response = self.client.responses.create(
                model=self.model,
                input=prompt
            )
            
            logger.info("API call completed successfully")
            logger.info(f"Response type: {type(response)}")
            logger.info(f"Response attributes: {dir(response)}")
            
            # Extract the output text
            if hasattr(response, 'output_text'):
                logger.info("Using response.output_text")
                result = response.output_text
            elif hasattr(response, 'choices') and len(response.choices) > 0:
                logger.info("Using response.choices[0].message.content")
                result = response.choices[0].message.content
            elif hasattr(response, 'text'):
                logger.info("Using response.text")
                result = response.text
            else:
                logger.warning("Unexpected response format, trying to extract content")
                logger.warning(f"Response object: {response}")
                result = str(response)
            
            logger.info(f"Extracted result length: {len(result) if result else 0} characters")
            logger.info(f"Result preview: {result[:200] if result else 'None'}...")
            
            if result:
                logger.info("Successfully extracted response content")
                return result
            else:
                logger.error(f"Empty response from API. Full response: {response}")
                raise ValueError("Empty response from API")
                
        except Exception as e:
            logger.error(f"API call failed with exception: {type(e).__name__}: {e}")
            logger.error(f"Exception details: {str(e)}")
            raise
    
    def _parse_recommendation(self, response: str) -> CodeRecommendation:
        """Parse API response as code recommendation"""
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
            
            # Try to fix common JSON formatting issues
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.warning(f"Initial JSON parse failed: {e}, attempting to fix common issues")
                
                # Common fixes for malformed JSON
                fixed_json = json_str
                
                # Fix missing commas before closing braces/brackets
                import re
                fixed_json = re.sub(r'"\s*\n\s*}', '"\n}', fixed_json)
                fixed_json = re.sub(r'"\s*\n\s*]', '"\n]', fixed_json)
                
                # Fix trailing commas
                fixed_json = re.sub(r',(\s*[}\]])', r'\1', fixed_json)
                
                # Try parsing again
                try:
                    data = json.loads(fixed_json)
                    logger.info("Successfully fixed JSON formatting issues")
                except json.JSONDecodeError:
                    # If still failing, try a more aggressive approach
                    logger.warning("Standard fixes failed, attempting manual JSON repair")
                    
                    # Save the problematic JSON for debugging
                    debug_file = Path("debug_malformed_json.txt")
                    with open(debug_file, 'w') as f:
                        f.write(f"Original JSON:\n{json_str}\n\nFixed JSON:\n{fixed_json}")
                    logger.info(f"Saved malformed JSON to {debug_file} for debugging")
                    
                    raise
            
            # Validate required fields
            required_fields = ["model_name", "training_code", "hyperparameters", "reasoning", "confidence", "bo_parameters", "bo_search_space"]
            for field in required_fields:
                if field not in data:
                    raise ValueError(f"Missing required field: {field}")
            
            return CodeRecommendation(
                model_name=data["model_name"],
                training_code=data["training_code"],
                hyperparameters=data["hyperparameters"],
                reasoning=data["reasoning"],
                confidence=float(data["confidence"]),
                bo_parameters=data["bo_parameters"],
                bo_search_space=data["bo_search_space"]
            )
        
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.error(f"Failed to parse code recommendation: {e}")
            logger.error(f"Original response: {response}")
            raise ValueError(f"Failed to parse AI code recommendation: {e}")
    
    def generate_training_function(self, data_profile: Dict[str, Any], input_shape: tuple, num_classes: int, include_literature_review: bool = None) -> CodeRecommendation:
        """Generate training function code based on data characteristics with optional literature review"""

        literature_review = None
        
        # Use config setting if not explicitly specified
        if include_literature_review is None:
            include_literature_review = not config.skip_literature_review

        # Generate literature review if requested
        if include_literature_review:
            try:
                logger.info("Conducting literature review before code generation...")
                literature_review = literature_review_generator.generate_literature_review(data_profile, input_shape, num_classes)

                # Save literature review
                literature_review_generator.save_literature_review(literature_review, data_profile)

            except Exception as e:
                logger.warning(f"Literature review failed: {e}, proceeding without it")
                literature_review = None

        # Generate training function with literature review insights
        prompt = self._create_prompt(data_profile, input_shape, num_classes, literature_review)
        response = self._call_openai_api(prompt)
        recommendation = self._parse_recommendation(response)

        logger.info(f"AI generated training function: {recommendation.model_name}")
        logger.info(f"Confidence: {recommendation.confidence:.2f}")
        logger.info(f"Reasoning: {recommendation.reasoning}")

        if literature_review:
            logger.info(f"Literature review informed code generation (confidence: {literature_review.confidence:.2f})")

        return recommendation
    
    def save_training_function(self, recommendation: CodeRecommendation, data_profile: Dict[str, Any]) -> str:
        """Save training function to JSON file"""
        
        # Create unique filename based on data characteristics and timestamp
        import time
        timestamp = int(time.time())
        data_type = data_profile.get('data_type', 'unknown')
        filename = f"training_function_{data_type}_{recommendation.model_name}_{timestamp}.json"
        filepath = self.code_storage_dir / filename
        
        # Prepare data for JSON storage
        training_data = {
            "model_name": recommendation.model_name,
            "training_code": recommendation.training_code,
            "hyperparameters": recommendation.hyperparameters,
            "reasoning": recommendation.reasoning,
            "confidence": recommendation.confidence,
            "bo_parameters": recommendation.bo_parameters,
            "bo_search_space": recommendation.bo_search_space,
            "data_profile": data_profile,
            "timestamp": timestamp,
            "metadata": {
                "generated_by": "AI Code Generator",
                "api_model": self.model,
                "version": "1.0"
            }
        }
        
        # Save to JSON file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Training function saved to: {filepath}")
        return str(filepath)
    

# Global instance
ai_code_generator = AICodeGenerator()

def generate_training_code_for_data(data_profile: Dict[str, Any], input_shape: tuple, num_classes: int, include_literature_review: bool = None) -> CodeRecommendation:
    """
    Convenience function: Generate training code for data with literature review
    """
    # Generate AI code with literature review
    recommendation = ai_code_generator.generate_training_function(
        data_profile, input_shape, num_classes, include_literature_review
    )

    return recommendation