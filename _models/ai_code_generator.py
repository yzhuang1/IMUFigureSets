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
from _models.literature_review import literature_review_generator, LiteratureReview

logger = logging.getLogger(__name__)

@dataclass
class CodeRecommendation:
    """Code generation result"""
    model_name: str
    training_code: str
    bo_config: Dict[str, Dict[str, Any]]
    confidence: float

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
    
    def _get_valid_divisors(self, n: int) -> List[int]:
        """Get all valid divisors of n for patch_size constraints"""
        if n <= 0:
            return [1]
        divisors = []
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                divisors.append(i)
                if i != n // i:
                    divisors.append(n // i)
        return sorted(divisors)

    def _create_prompt(self, data_profile: Dict[str, Any], input_shape: tuple, num_classes: int, literature_review: Optional[LiteratureReview] = None) -> str:
        """Create enhanced prompt for GPT-5 code generation with literature review insights"""

        # Dataset context for better model selection
        dataset_context = ""
        if config.dataset_name and config.dataset_name != "Unknown Dataset":
            dataset_context = f"\nDataset: {config.dataset_name}\nSource: {config.dataset_source}"

        # Sample count
        num_samples = data_profile.get("sample_count", data_profile.get("num_samples", "unknown"))

        # Extract sequence length for patch_size constraints
        sequence_length = None
        if len(input_shape) >= 2:
            # For shape like (1000, 2) or (2, 1000), find the larger dimension as sequence length
            sequence_length = max(input_shape[-2:]) if max(input_shape[-2:]) > 10 else None

        # Generate valid divisors for patch_size if sequence length detected
        valid_patch_sizes = []
        if sequence_length and sequence_length > 1:
            valid_patch_sizes = self._get_valid_divisors(sequence_length)

        prompt = f"""
Generate a PyTorch training function for a {num_classes}-class classifier.

Data: {data_profile['data_type']}, input shape {input_shape}, {num_samples} samples{dataset_context}
Sequence length: {sequence_length if sequence_length else 'N/A'}

- Use the first recommended approach if available, otherwise proceed with a reasonable architecture.
"""

        if literature_review:
            rec = (literature_review.recommended_approaches[0]
                if literature_review.recommended_approaches else "No specific recommendation")
            prompt += f"\Recommend: {rec}"

        prompt += """
Output: VALID JSON ONLY.

Requirements:
- Provide "training_code" implementing:
  def train_model(X_train, y_train, X_val, y_val, device, **hyperparams **quantization parameters)
  * tensors as inputs
  * build model from scratch
  * core train loop with epoch-by-epoch logging (log epoch, train_loss, val_loss, val_acc each epoch)
  * return quantized model + metrics (include lists: train_losses, val_losses, val_acc for all epochs)
  * fill in the real input you need for hyperparams and quantization parameters
  * IMPORTANT: For DataLoader, use pin_memory=False to avoid CUDA tensor pinning errors
  * CRITICAL: ALWAYS train on GPU - ensure ALL tensors (model, inputs, targets, losses) are on the same device (GPU). Use .to(device) consistently throughout training loop.
- Final model (after quantization) MUST have ≤ 256KB storage size.
- Implement post-training quantization using torch.quantization / torch.ao.quantization.
  * Include hyperparams: quantization_bits ∈ {8, 16, 32}, quantize_weights ∈ {true,false}, quantize_activations ∈ {true,false}.
  * Choose a sensible strategy for the chosen architecture (e.g., CNN vs Transformer).

Bayesian Optimization:
- Provide "bo_config" with ALL hyperparameters used in training_code.
- Each item MUST have: "default", "type" ∈ {"Real","Integer","Categorical"}, and valid ranges:
  * Real: low, high, optional prior ∈ {"uniform","log-uniform"} (if log-uniform, low > 0, e.g., 1e-6)
  * Integer: low, high (inclusive)
  * Categorical: categories [..]
- Only include params actually consumed by training_code.{f'''
- CRITICAL: For transformer models with patch_size, it must divide sequence length ({sequence_length}). Use only valid divisors: {valid_patch_sizes}''' if valid_patch_sizes else ""}

Response JSON example:
{
  "model_name": "ModelName",
  "training_code": "def train_model(...):\\n    # code",
  "bo_config": {
    "lr": {"default": 0.001, "type": "Real", "low": 1e-6, "high": 1e-1, "prior": "log-uniform"},
    "batch_size": {"default": 64, "type": "Categorical", "categories": [8,16,32,64,128]},
    "epochs": {"default": 10, "type": "Integer", "low": 5, "high": 50},
    "hidden_size": {"default": 128, "type": "Integer", "low": 32, "high": 512},
    "dropout": {"default": 0.1, "type": "Real", "low": 0.0, "high": 0.7},
    "quantization_bits": {"default": 32, "type": "Categorical", "categories": [8,16,32]},
    "quantize_weights": {"default": false, "type": "Categorical", "categories": [true,false]},
    "quantize_activations": {"default": false, "type": "Categorical", "categories": [true,false]}
  },
  "confidence": 0.9
}
"""
        return prompt
    
    def _call_openai_api(self, prompt: str) -> str:
        """Call OpenAI API using responses.create method"""
        logger.info(f"Making API call to {self.model}")
        logger.info(f"Prompt length: {len(prompt)} characters")
        logger.info(f"Using API base URL: {self.base_url}")
        
        try:
            logger.info("Calling self.client.responses.create...")
            logger.info(f"Model parameter: {self.model}")
            
            response = self.client.responses.create(
                model=self.model,
                input=prompt
            )
            
            
            # Extract the output text
            if hasattr(response, 'output_text'):
                result = response.output_text
            elif hasattr(response, 'choices') and len(response.choices) > 0:
                result = response.choices[0].message.content
            elif hasattr(response, 'text'):
                result = response.text
            else:
                logger.warning("Unexpected response format, trying to extract content")
                logger.warning(f"Response object: {response}")
                result = str(response)
            
            
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
    

    def _debug_json_with_gpt(self, training_code: str, error_message: str, bo_config: dict = None) -> str:
        """Use GPT to debug and fix training errors by suggesting hyperparameter corrections"""
        from config import config
        import torch
        import os
        import json
        from datetime import datetime
        from pathlib import Path

        # Create folder for GPT debugging responses
        debug_folder = Path("gpt_debug_responses")
        debug_folder.mkdir(exist_ok=True)

        # Get PyTorch version for context
        pytorch_version = torch.__version__

        # Get debug retry attempts from config
        max_debug_attempts = config.debug_chances
        logger.info(f"Starting debug attempts (max: {max_debug_attempts})")

        # Decode escaped training code for better readability
        decoded_training_code = training_code.replace('\\n', '\n').replace('\\t', '\t').replace('\\"', '"').replace('\\\\', '\\')

        debug_prompt = """CRITICAL: You MUST respond with ONLY valid JSON. No text before or after the JSON.

Analyze this PyTorch training error and provide either hyperparameter corrections OR fixed training code.

PyTorch Version: {}
Training Error: {}
BO Config: {}
Training Code: {}

RESPONSE OPTIONS:
1. HYPERPARAMETER FIX: If error can be fixed by changing hyperparameters
   Output: {{"bo_config": {{"param_name": new_value, "param2": new_value}}}}

2. CODE FIX: If error requires fixing bugs in the training code
   Output: {{"training_code": "complete_corrected_training_function_code"}}

3. CANNOT FIX: If error cannot be resolved
   Output: {{}}

RESPONSE FORMAT REQUIREMENTS:
1. Output ONLY a JSON object with either "bo_config" OR "training_code" field
2. No explanations, no markdown, no ```json``` blocks
3. Start with {{ and end with }}
4. For training_code fixes, include the COMPLETE corrected function

CORRECTION EXAMPLES:
- "Model has X KB storage, exceeds 256KB limit" → {{"bo_config": {{"d_model": 64, "hidden_size": 128}}}}
- "'str' object has no attribute 'type'" → {{"training_code": "def train_model(...):\\n    # fixed implementation"}}
- "Quantization bug in code" → {{"training_code": "corrected_training_function"}}
- "mat1 and mat2 shapes cannot be multiplied" → {{"bo_config": {{"d_model": 128}}}}

OUTPUT ONLY THE JSON OBJECT:""".format(pytorch_version, error_message, bo_config or {}, decoded_training_code)

        # Retry logic for debug attempts
        for attempt in range(1, max_debug_attempts + 1):
            try:
                logger.info(f"Calling GPT to debug training error (attempt {attempt}/{max_debug_attempts})")
                debug_response = self._call_openai_api(debug_prompt)

                # Save the raw GPT response to file for analysis
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                error_type = "training_error"
                response_filename = f"gpt_debug_{error_type}_{timestamp}_attempt{attempt}.txt"
                response_filepath = debug_folder / response_filename

                with open(response_filepath, 'w', encoding='utf-8') as f:
                    f.write(f"=== GPT DEBUG RESPONSE ===\n")
                    f.write(f"Timestamp: {timestamp}\n")
                    f.write(f"Attempt: {attempt}/{max_debug_attempts}\n")
                    f.write(f"Error Type: {error_type}\n")
                    f.write(f"Original Error: {error_message}\n")
                    f.write(f"=== RAW GPT RESPONSE ===\n")
                    f.write(debug_response)
                    f.write(f"\n=== PROMPT USED ===\n")
                    f.write(debug_prompt)

                logger.info(f"Saved GPT debug response to: {response_filepath}")

                # Clean the debug response
                debug_response = debug_response.strip()
                if debug_response.startswith('```json'):
                    debug_response = debug_response[7:]
                if debug_response.endswith('```'):
                    debug_response = debug_response[:-3]

                # Extract JSON from debug response
                start_idx = debug_response.find('{')
                end_idx = debug_response.rfind('}') + 1

                if start_idx != -1 and end_idx > 0:
                    correction_json = debug_response[start_idx:end_idx]
                    logger.info(f"GPT suggested correction: {correction_json}")

                    # Apply the same JSON fixing logic as in main parsing
                    try:
                        # First try to parse as-is
                        import json
                        json.loads(correction_json)
                        logger.info(f"Debug successful on attempt {attempt}")
                        return correction_json
                    except json.JSONDecodeError as e:
                        logger.warning(f"Debug JSON parse failed: {e}, attempting to fix")

                        # Apply the same fixes as in main JSON parsing
                        import re
                        fixed_json = correction_json

                        # Fix missing closing quotes before commas/braces
                        fixed_json = re.sub(r'(\d+)"\s*([,}])', r'\1\2', fixed_json)
                        fixed_json = re.sub(r'([^"])"\s*([,}])', r'\1"\2', fixed_json)

                        # Fix missing commas after closing braces in nested objects
                        fixed_json = re.sub(r'}\s*\n\s*"', r'},\n    "', fixed_json)

                        # Fix missing commas before closing braces/brackets
                        fixed_json = re.sub(r'"\s*\n\s*}', '"\n}', fixed_json)
                        fixed_json = re.sub(r'"\s*\n\s*]', '"\n]', fixed_json)

                        # Fix trailing commas
                        fixed_json = re.sub(r',(\s*[}\]])', r'\1', fixed_json)

                        # Fix malformed numbers with quotes
                        fixed_json = re.sub(r'"(\d+\.?\d*)"([,}])', r'\1\2', fixed_json)

                        try:
                            json.loads(fixed_json)
                            logger.info(f"Successfully fixed debug JSON formatting on attempt {attempt}")
                            return fixed_json
                        except json.JSONDecodeError:
                            logger.warning(f"Could not fix debug JSON formatting on attempt {attempt}")
                            if attempt == max_debug_attempts:
                                return correction_json  # Return original even if broken on last attempt
                            # Continue to next attempt
                else:
                    logger.warning(f"No valid JSON found in debug response (attempt {attempt})")
                    if attempt == max_debug_attempts:
                        return "{}"
                    # Continue to next attempt

            except Exception as e:
                logger.error(f"Failed to debug JSON with GPT on attempt {attempt}: {e}")
                if attempt == max_debug_attempts:
                    return "{}"
                # Continue to next attempt

        # If all attempts failed
        logger.error(f"All {max_debug_attempts} debug attempts failed")
        return "{}"

    def _apply_json_corrections(self, original_json: str, corrections: str) -> str:
        """Apply JSON corrections to the original JSON"""
        try:
            # Parse the corrections
            correction_data = json.loads(corrections)

            # Parse the original JSON (even if malformed, try our best)
            try:
                original_data = json.loads(original_json)
            except json.JSONDecodeError:
                # If original is malformed, try to parse it partially
                logger.warning("Original JSON is malformed, attempting partial repair")
                original_data = {}

            # Apply corrections by updating the original data
            def update_nested_dict(target, source):
                for key, value in source.items():
                    if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                        update_nested_dict(target[key], value)
                    else:
                        target[key] = value

            update_nested_dict(original_data, correction_data)

            # Return the corrected JSON
            corrected_json = json.dumps(original_data, indent=2)
            logger.info("Successfully applied JSON corrections")
            return corrected_json

        except Exception as e:
            logger.error(f"Failed to apply JSON corrections: {e}")
            return original_json

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

                # Enhanced fixes for malformed JSON
                import re
                fixed_json = json_str

                # Fix missing closing quotes before commas/braces
                fixed_json = re.sub(r'(\d+)"\s*([,}])', r'\1\2', fixed_json)
                fixed_json = re.sub(r'([^"])"\s*([,}])', r'\1"\2', fixed_json)

                # Fix missing closing braces for nested objects after array values
                fixed_json = re.sub(r'(\[[^\]]*\])\s*,\s*\n\s*"', r'\1\n    },\n    "', fixed_json)

                # Fix missing commas after closing braces in nested objects
                fixed_json = re.sub(r'}\s*\n\s*"', r'},\n    "', fixed_json)

                # Fix missing commas before closing braces/brackets
                fixed_json = re.sub(r'"\s*\n\s*}', '"\n}', fixed_json)
                fixed_json = re.sub(r'"\s*\n\s*]', '"\n]', fixed_json)

                # Fix trailing commas
                fixed_json = re.sub(r',(\s*[}\]])', r'\1', fixed_json)

                # Fix malformed numbers with quotes
                fixed_json = re.sub(r'"(\d+\.?\d*)"([,}])', r'\1\2', fixed_json)

                # Try parsing again
                try:
                    data = json.loads(fixed_json)
                    logger.info("Successfully fixed JSON formatting issues")
                except json.JSONDecodeError as second_e:
                    # If still failing, use GPT to debug the JSON
                    logger.warning("Standard fixes failed, using GPT to debug JSON")

                    # Save the problematic JSON for debugging
                    debug_file = Path("debug_malformed_json.txt")
                    with open(debug_file, 'w') as f:
                        f.write(f"Original JSON:\n{json_str}\n\nFixed JSON:\n{fixed_json}")
                    logger.info(f"Saved malformed JSON to {debug_file} for debugging")

                    # Use GPT to debug and fix the JSON
                    corrections = self._debug_json_with_gpt(fixed_json, str(second_e))
                    if corrections and corrections != "{}":
                        corrected_json = self._apply_json_corrections(fixed_json, corrections)
                        try:
                            data = json.loads(corrected_json)
                            logger.info("Successfully repaired JSON using GPT debugging")
                        except json.JSONDecodeError:
                            logger.error("GPT-assisted JSON repair also failed")
                            raise
                    else:
                        logger.error("GPT could not provide valid corrections")
                        raise

            # Validate required fields for the new format (aligned with prompt)
            required_fields = ["model_name", "training_code", "bo_config", "confidence"]
            for field in required_fields:
                if field not in data:
                    raise ValueError(f"Missing required field: {field}")

            # Validate bo_config structure
            bo_config = data["bo_config"]
            if not isinstance(bo_config, dict):
                raise ValueError("bo_config must be a dictionary")

            for param_name, config in bo_config.items():
                # Validate bo_config parameter structure
                if not isinstance(config, dict):
                    raise ValueError(f"Invalid bo_config parameter '{param_name}': must be a dictionary")

                # Check required fields for each parameter
                if "default" not in config:
                    raise ValueError(f"Missing 'default' value for parameter '{param_name}'")
                if "type" not in config:
                    raise ValueError(f"Missing 'type' for parameter '{param_name}'")

            return CodeRecommendation(
                model_name=data["model_name"],
                training_code=data["training_code"],
                bo_config=bo_config,
                confidence=float(data["confidence"])
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

        if literature_review:
            logger.info(f"Literature review informed code generation (confidence: {literature_review.confidence:.2f})")

        return recommendation
    
    def save_training_function(self, recommendation: CodeRecommendation, data_profile: Dict[str, Any]) -> str:
        """Save training function to JSON file"""
        
        # Create unique filename based on data characteristics and timestamp
        import time
        import re
        timestamp = int(time.time())
        data_type = data_profile.get('data_type', 'unknown')

        # Sanitize model name for filename (remove/replace invalid characters)
        safe_model_name = re.sub(r'[<>:"/\\|?*()[\]{}]', '_', recommendation.model_name)
        safe_model_name = re.sub(r'\s+', '_', safe_model_name)  # Replace spaces with underscores
        safe_model_name = safe_model_name.strip('_')  # Remove leading/trailing underscores

        filename = f"training_function_{data_type}_{safe_model_name}_{timestamp}.json"
        filepath = self.code_storage_dir / filename
        
        # Prepare data for JSON storage
        training_data = {
            "model_name": recommendation.model_name,
            "training_code": recommendation.training_code,
            "bo_config": recommendation.bo_config,
            "confidence": recommendation.confidence,
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

        # Print filename prominently for visibility
        print(f"💾 SAVED TRAINING FUNCTION: {filename}")
        print(f"📁 Full path: {filepath}")

        logger.info(f"Training function saved to: {filepath}")
        return str(filepath)
    

# Global instance
ai_code_generator = AICodeGenerator()

def generate_training_code_for_data(data_profile: Dict[str, Any], input_shape: tuple, num_classes: int, include_literature_review: bool = None) -> CodeRecommendation:
    """
    Convenience function: Generate training code for data with literature review (JSON errors handled by GPT debugging)
    """
    return ai_code_generator.generate_training_function(
        data_profile, input_shape, num_classes, include_literature_review
    )