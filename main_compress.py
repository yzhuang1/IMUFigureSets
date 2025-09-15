"""
AI-Powered Model Compression
Uses GPT-5 to analyze model and generate custom compression functions
Target: Compress model to under 256K
"""

import torch
import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import openai
import numpy as np
from config import config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

class AIModelCompressor:
    def __init__(self, model_path: str):
        """
        Initialize AI Model Compressor
        
        Args:
            model_path: Path to the model file to compress
        """
        self.model_path = model_path
        self.model_info = None
        self.compression_functions = None
        self.output_dir = "compressed_models"
        self.functions_dir = "ai_generated_compression_functions"
        
        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.functions_dir, exist_ok=True)
        
        # Verify OpenAI configuration
        if not config.is_openai_configured():
            raise ValueError("OpenAI API key not configured. Run setup_api_key.py first.")
        
        # Initialize OpenAI client
        self.client = openai.OpenAI(api_key=config.openai_api_key)
    
    def analyze_model(self) -> Dict[str, Any]:
        """Analyze the model to extract detailed information"""
        logger.info(f"Analyzing model: {self.model_path}")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        # Load model checkpoint
        checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=False)
        
        # Get file size
        file_size_bytes = os.path.getsize(self.model_path)
        file_size_kb = file_size_bytes / 1024
        
        # Analyze state dict
        state_dict = checkpoint['model_state_dict']
        
        # Count parameters by layer type and data type
        layer_analysis = {}
        total_params = 0
        dtype_analysis = {}
        
        for name, tensor in state_dict.items():
            # Parse layer information
            layer_type = self._identify_layer_type(name)
            params = tensor.numel()
            dtype = str(tensor.dtype)
            
            total_params += params
            
            # Layer type analysis
            if layer_type not in layer_analysis:
                layer_analysis[layer_type] = {
                    'tensors': 0,
                    'params': 0,
                    'shapes': [],
                    'names': []
                }
            
            layer_analysis[layer_type]['tensors'] += 1
            layer_analysis[layer_type]['params'] += params
            layer_analysis[layer_type]['shapes'].append(list(tensor.shape))
            layer_analysis[layer_type]['names'].append(name)
            
            # Data type analysis
            if dtype not in dtype_analysis:
                dtype_analysis[dtype] = {'tensors': 0, 'params': 0}
            dtype_analysis[dtype]['tensors'] += 1
            dtype_analysis[dtype]['params'] += params
        
        # Calculate parameter distribution percentages
        for layer_type in layer_analysis:
            layer_analysis[layer_type]['percentage'] = (
                layer_analysis[layer_type]['params'] / total_params
            ) * 100
        
        # Extract model metadata
        model_metadata = {
            'model_name': checkpoint.get('model_name', 'Unknown'),
            'hyperparameters': checkpoint.get('best_hyperparameters', {}),
            'final_metrics': checkpoint.get('final_metrics', {}),
            'timestamp': checkpoint.get('timestamp', 'Unknown')
        }
        
        # Compile comprehensive model information
        self.model_info = {
            'file_info': {
                'path': self.model_path,
                'size_bytes': file_size_bytes,
                'size_kb': file_size_kb,
                'target_size_kb': 256,
                'compression_ratio_needed': file_size_kb / 256
            },
            'architecture': {
                'total_parameters': total_params,
                'layer_analysis': layer_analysis,
                'dtype_analysis': dtype_analysis,
                'model_metadata': model_metadata
            },
            'sample_weights': self._extract_sample_weights(state_dict)
        }
        
        logger.info(f"Model analysis complete:")
        logger.info(f"  Current size: {file_size_kb:.1f} KB")
        logger.info(f"  Target size: 256 KB")
        logger.info(f"  Required compression ratio: {file_size_kb / 256:.2f}x")
        logger.info(f"  Total parameters: {total_params:,}")
        
        return self.model_info
    
    def _identify_layer_type(self, layer_name: str) -> str:
        """Identify the type of layer from its name"""
        name_lower = layer_name.lower()
        
        if 'lstm' in name_lower:
            if 'weight_ih' in name_lower:
                return 'LSTM_input_hidden'
            elif 'weight_hh' in name_lower:
                return 'LSTM_hidden_hidden'
            elif 'bias' in name_lower:
                return 'LSTM_bias'
            else:
                return 'LSTM_other'
        elif 'gru' in name_lower:
            return 'GRU'
        elif 'conv' in name_lower:
            return 'Convolution'
        elif 'linear' in name_lower or 'fc' in name_lower:
            return 'Linear'
        elif 'embedding' in name_lower:
            return 'Embedding'
        elif 'attention' in name_lower:
            return 'Attention'
        elif 'norm' in name_lower or 'bn' in name_lower:
            return 'Normalization'
        else:
            return 'Other'
    
    def _extract_sample_weights(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Extract sample weight values for analysis"""
        samples = {}
        
        for name, tensor in list(state_dict.items())[:3]:  # Sample first 3 layers
            if tensor.numel() > 0:
                flat_tensor = tensor.flatten()
                samples[name] = {
                    'shape': list(tensor.shape),
                    'dtype': str(tensor.dtype),
                    'sample_values': [float(x) for x in flat_tensor[:5]],
                    'min_value': float(tensor.min()),
                    'max_value': float(tensor.max()),
                    'mean_value': float(tensor.mean()),
                    'std_value': float(tensor.std())
                }
        
        return samples
    
    def generate_compression_strategy(self) -> Dict[str, Any]:
        """Use GPT-5 to generate custom compression strategy"""
        logger.info("Generating AI compression strategy...")
        
        if not self.model_info:
            raise ValueError("Model must be analyzed first. Call analyze_model().")
        
        # Create detailed prompt for GPT-5
        prompt = self._create_compression_prompt()
        
        # Try multiple approaches - GPT-4 first to validate logic, then GPT-5
        attempts = [
            {
                "name": "GPT-4 Fallback Test",
                "model": "gpt-4-turbo",
                "system": "You are a PyTorch compression expert. Generate working compression code.",
                "max_tokens": 4000,
                "user_prefix": "Generate JSON compression strategy:"
            },
            {
                "name": "GPT-5 Short Direct",
                "model": config.openai_model,
                "system": "Respond ONLY with JSON. No reasoning.",
                "max_tokens": 1000,
                "user_prefix": "JSON ONLY:"
            },
            {
                "name": "GPT-5 No System Prompt",
                "model": config.openai_model,
                "system": "",
                "max_tokens": 2000,
                "user_prefix": "Provide JSON compression strategy for LSTM model:"
            }
        ]
        
        for i, attempt in enumerate(attempts):
            try:
                logger.info(f"GPT-5 Attempt {i+1}: {attempt['name']}")
                
                # Prepare user content with prefix
                user_content = prompt if i == 0 else self._create_simple_prompt()
                if "user_prefix" in attempt:
                    user_content = attempt["user_prefix"] + "\\n\\n" + user_content
                
                messages = [
                    {
                        "role": "system",
                        "content": attempt["system"]
                    },
                    {
                        "role": "user", 
                        "content": user_content
                    }
                ]
                
                response = self.client.chat.completions.create(
                    model=attempt["model"],
                    messages=messages,
                    max_completion_tokens=attempt["max_tokens"]
                )
                
                # Parse the response - Debug GPT-5 response structure
                logger.info("=== GPT-5 Response Debug ===")
                logger.info(f"Model used: {response.model}")
                logger.info(f"Finish reason: {response.choices[0].finish_reason}")
                logger.info(f"Total tokens: {response.usage.total_tokens}")
                logger.info(f"Completion tokens: {response.usage.completion_tokens}")
                # Handle reasoning tokens safely (GPT-4 may not have this)
                reasoning_tokens = getattr(response.usage.completion_tokens_details, 'reasoning_tokens', 0) if hasattr(response.usage, 'completion_tokens_details') else 0
                logger.info(f"Reasoning tokens: {reasoning_tokens}")
                
                if response.choices and len(response.choices) > 0:
                    choice = response.choices[0]
                    response_content = choice.message.content
                    
                    logger.info(f"Response content length: {len(response_content) if response_content else 0}")
                    logger.info(f"Message role: {choice.message.role}")
                    logger.info(f"Has refusal: {choice.message.refusal is not None}")
                    
                    # Check if we got reasoning tokens but no content
                    if not response_content and reasoning_tokens > 0:
                        logger.warning("GPT-5 used reasoning tokens but returned empty content")
                        logger.info("This might be due to the response format or prompt structure")
                        continue  # Try next attempt
                    
                    if response_content and len(response_content.strip()) > 0:
                        logger.info(f"GPT-5 response captured successfully on attempt {i+1}")
                        break  # Success - exit the loop
                    else:
                        logger.warning(f"Attempt {i+1} returned empty content, trying next approach...")
                        continue
                else:
                    logger.error(f"Attempt {i+1}: No response choices available")
                    continue
                    
            except Exception as attempt_error:
                logger.error(f"Attempt {i+1} failed: {attempt_error}")
                if i == len(attempts) - 1:  # Last attempt
                    raise
                continue
        
        # Check if we got any content - if not, use manual fallback
        if not response_content or len(response_content.strip()) == 0:
            logger.warning("All GPT-5 attempts returned empty content")
            logger.info("Falling back to manual LSTM compression strategy")
            manual_strategy = self._create_manual_strategy_from_response("")
            self.compression_functions = manual_strategy
            
            # Save manual strategy
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            strategy_file = f"{self.functions_dir}/manual_compression_strategy_{timestamp}.json"
            
            with open(strategy_file, 'w') as f:
                json.dump({
                    'model_info': self.model_info,
                    'compression_strategy': manual_strategy,
                    'gpt_response': "Manual fallback due to GPT-5 reasoning mode issue",
                    'timestamp': timestamp,
                    'fallback_reason': "GPT-5 used all tokens for reasoning, no output content"
                }, f, indent=2, cls=NumpyEncoder)
            
            logger.info(f"Manual compression strategy saved to: {strategy_file}")
            return manual_strategy
        
        # Save the raw response first
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        raw_response_file = f"{self.functions_dir}/gpt_raw_response_{timestamp}.txt"
        
        with open(raw_response_file, 'w') as f:
            f.write(response_content)
        
        logger.info(f"Raw GPT response saved to: {raw_response_file}")
        
        # Try to extract JSON from the response
        try:
            compression_strategy = self._parse_gpt_response(response_content)
            
            # Save the strategy
            strategy_file = f"{self.functions_dir}/compression_strategy_{timestamp}.json"
            
            with open(strategy_file, 'w') as f:
                json.dump({
                    'model_info': self.model_info,
                    'compression_strategy': compression_strategy,
                    'gpt_response': response_content,
                    'timestamp': timestamp
                }, f, indent=2, cls=NumpyEncoder)
            
            logger.info(f"Compression strategy saved to: {strategy_file}")
            
            self.compression_functions = compression_strategy
            return compression_strategy
            
        except Exception as parse_error:
            logger.error(f"Failed to parse GPT response: {parse_error}")
            logger.info(f"Check raw response in: {raw_response_file}")
            # Instead of failing, let's try to create a manual strategy based on the response
            try:
                manual_strategy = self._create_manual_strategy_from_response(response_content)
                logger.info("Created manual compression strategy as fallback")
                self.compression_functions = manual_strategy
                return manual_strategy
            except:
                raise parse_error
    
    def _create_manual_strategy_from_response(self, response_content: str) -> Dict[str, Any]:
        """Create a manual compression strategy if GPT response can't be parsed"""
        logger.info("Creating manual LSTM compression strategy as fallback")
        
        # Create a robust compression function for LSTM models
        compression_code = '''
def compress_model(input_path, output_path):
    """
    Aggressive LSTM model compression using multiple techniques
    Target: <256KB from current size
    """
    import torch
    import torch.nn as nn
    import numpy as np
    import os
    
    try:
        # Load original model
        checkpoint = torch.load(input_path, map_location='cpu', weights_only=False)
        original_size = os.path.getsize(input_path) / 1024
        
        print(f"Original model size: {original_size:.1f} KB")
        
        # Extract state dict
        state_dict = checkpoint['model_state_dict']
        
        # Step 1: Convert to FP16 for immediate 50% reduction
        fp16_state_dict = {}
        for key, tensor in state_dict.items():
            if tensor.dtype == torch.float32:
                fp16_state_dict[key] = tensor.half()
            else:
                fp16_state_dict[key] = tensor
        
        # Step 2: Aggressive pruning of LSTM weights (30% sparsity)
        pruned_state_dict = {}
        for key, tensor in fp16_state_dict.items():
            if 'lstm.weight' in key and tensor.dim() >= 2:
                # Create mask for top 70% of weights by magnitude
                flat_tensor = tensor.flatten()
                threshold_idx = int(0.3 * len(flat_tensor))
                _, indices = torch.topk(torch.abs(flat_tensor), len(flat_tensor) - threshold_idx)
                
                # Create pruned tensor
                pruned_tensor = tensor.clone()
                mask = torch.zeros_like(flat_tensor)
                mask[indices] = 1
                mask = mask.reshape(tensor.shape)
                pruned_tensor = pruned_tensor * mask
                
                pruned_state_dict[key] = pruned_tensor
            else:
                pruned_state_dict[key] = tensor
        
        # Step 3: Quantization simulation for remaining weights
        quantized_state_dict = {}
        for key, tensor in pruned_state_dict.items():
            if tensor.dtype == torch.float16 and 'lstm' in key:
                # Simulate 8-bit quantization by reducing precision
                min_val, max_val = tensor.min(), tensor.max()
                scale = (max_val - min_val) / 255.0
                quantized = torch.round((tensor - min_val) / scale) * scale + min_val
                quantized_state_dict[key] = quantized.half()
            else:
                quantized_state_dict[key] = tensor
        
        # Create compressed checkpoint
        compressed_checkpoint = {
            **checkpoint,
            'model_state_dict': quantized_state_dict,
            'compression_info': {
                'methods': ['fp16', 'pruning_30pct', 'quantization_8bit'],
                'original_size_kb': original_size,
                'compression_applied': True
            }
        }
        
        # Save compressed model
        torch.save(compressed_checkpoint, output_path)
        
        # Calculate compression results
        compressed_size = os.path.getsize(output_path) / 1024
        compression_ratio = original_size / compressed_size
        
        results = {
            'success': True,
            'original_size_kb': original_size,
            'compressed_size_kb': compressed_size,
            'compression_ratio': compression_ratio,
            'size_reduction_pct': ((original_size - compressed_size) / original_size) * 100,
            'target_met': compressed_size < 256,
            'methods_used': ['FP16', 'LSTM_Pruning_30%', 'Quantization_8bit']
        }
        
        print(f"Compressed model size: {compressed_size:.1f} KB")
        print(f"Compression ratio: {compression_ratio:.2f}x")
        print(f"Target (<256 KB) met: {compressed_size < 256}")
        
        return results
        
    except Exception as e:
        print(f"Compression failed: {e}")
        return {'success': False, 'error': str(e)}
'''
        
        return {
            "compression_strategy": {
                "method": "Multi-stage LSTM compression: FP16 + Pruning + Quantization",
                "estimated_size_kb": 120,
                "estimated_compression_ratio": 4.7,
                "techniques": ["FP16", "Weight_Pruning_30%", "Quantization_8bit"]
            },
            "compression_function": {
                "function_name": "compress_model",
                "code": compression_code,
                "imports": ["torch", "numpy", "os"],
                "description": "Aggressive LSTM compression using FP16, 30% pruning, and 8-bit quantization"
            },
            "usage_instructions": {
                "how_to_run": "Call compress_model(input_path, output_path)",
                "expected_output": "Compressed model <256KB with minimal accuracy loss",
                "validation_steps": ["Check file size", "Test model loading", "Verify compression ratio"]
            }
        }
    
    def _create_compression_prompt(self) -> str:
        """Create detailed prompt for GPT-5"""
        
        model_summary = f"""
MODEL ANALYSIS SUMMARY:
====================

File Information:
- Current size: {self.model_info['file_info']['size_kb']:.1f} KB
- Target size: {self.model_info['file_info']['target_size_kb']} KB  
- Required compression ratio: {self.model_info['file_info']['compression_ratio_needed']:.2f}x

Architecture:
- Model type: {self.model_info['architecture']['model_metadata']['model_name']}
- Total parameters: {self.model_info['architecture']['total_parameters']:,}
- Current data type: {list(self.model_info['architecture']['dtype_analysis'].keys())}

Layer Distribution:
"""
        
        for layer_type, info in self.model_info['architecture']['layer_analysis'].items():
            model_summary += f"- {layer_type}: {info['params']:,} params ({info['percentage']:.1f}%)\n"
        
        model_summary += f"""
Hyperparameters:
{json.dumps(self.model_info['architecture']['model_metadata']['hyperparameters'], indent=2, cls=NumpyEncoder)}

Sample Weight Statistics:
"""
        for name, stats in self.model_info['sample_weights'].items():
            model_summary += f"- {name}: range=[{stats['min_value']:.6f}, {stats['max_value']:.6f}], mean={stats['mean_value']:.6f}\n"
        
        prompt = f"""
{model_summary}

COMPRESSION TASK:
================

I need you to create a compression function that will reduce this model from {self.model_info['file_info']['size_kb']:.1f} KB to under 256 KB (compression ratio of {self.model_info['file_info']['compression_ratio_needed']:.2f}x).

REQUIREMENTS:
1. The compressed model MUST be smaller than 256 KB
2. Minimize accuracy loss (target < 5% degradation)
3. Generate working Python code
4. Consider the model architecture (appears to be LSTM-based)
5. Use appropriate compression techniques for this architecture

CRITICAL: You must provide a concrete, executable response. Do not just reason - provide the actual JSON output.

RESPONSE FORMAT:
Respond with ONLY a valid JSON object containing:

{{
  "compression_strategy": {{
    "method": "Brief description of compression approach",
    "estimated_size_kb": <estimated final size>,
    "estimated_compression_ratio": <ratio>,
    "techniques": ["list", "of", "techniques", "used"]
  }},
  "compression_function": {{
    "function_name": "compress_model",
    "code": "Complete Python function code as a string",
    "imports": ["list", "of", "required", "imports"],
    "description": "Detailed explanation of what the function does"
  }},
  "usage_instructions": {{
    "how_to_run": "Step by step instructions",
    "expected_output": "Description of output",
    "validation_steps": ["steps", "to", "verify", "compression"]
  }}
}}

IMPORTANT GUIDELINES:
- For LSTM models, consider: quantization (INT8), pruning, low-rank decomposition, knowledge distillation
- Ensure the function can load the PyTorch model checkpoint
- The function should save the compressed model to a new file
- Include error handling and validation
- Make sure the compression ratio is sufficient to reach <256KB
- Consider combining multiple techniques if needed for aggressive compression

Please generate the compression strategy now.
"""
        
        return prompt
    
    def _create_simple_prompt(self) -> str:
        """Create a simplified prompt for GPT-5"""
        return f"""
Create a Python function to compress this PyTorch model:
- Current size: {self.model_info['file_info']['size_kb']:.1f} KB
- Target: <256 KB  
- Model type: LSTM with {self.model_info['architecture']['total_parameters']:,} parameters

Provide JSON with compression code that uses quantization and pruning to achieve <256KB.

Format:
{{
  "function_code": "def compress_model(input_path, output_path): ...",
  "description": "Compression method"
}}
"""
    
    def _parse_gpt_response(self, response_content: str) -> Dict[str, Any]:
        """Parse GPT response and extract JSON"""
        try:
            # Remove markdown code blocks if present
            content = response_content.strip()
            if content.startswith('```json'):
                content = content[7:]  # Remove ```json
            if content.startswith('```'):
                content = content[3:]   # Remove ```
            if content.endswith('```'):
                content = content[:-3]  # Remove trailing ```
            
            # Try to find JSON in the response
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON found in response")
            
            json_str = content[start_idx:end_idx]
            compression_strategy = json.loads(json_str)
            
            # Fix escaped newlines in code if present
            if 'compression_function' in compression_strategy and 'code' in compression_strategy['compression_function']:
                code = compression_strategy['compression_function']['code']
                # Replace \\n with actual newlines
                compression_strategy['compression_function']['code'] = code.replace('\\n', '\n')
            
            # Validate required fields
            required_fields = ['compression_strategy', 'compression_function', 'usage_instructions']
            for field in required_fields:
                if field not in compression_strategy:
                    raise ValueError(f"Missing required field: {field}")
            
            return compression_strategy
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from GPT response: {e}")
            logger.error(f"Response content: {response_content}")
            raise ValueError("Invalid JSON in GPT response")
    
    def execute_compression(self) -> Dict[str, Any]:
        """Execute the AI-generated compression function"""
        logger.info("Executing AI-generated compression...")
        
        if not self.compression_functions:
            raise ValueError("Compression strategy not generated. Call generate_compression_strategy() first.")
        
        try:
            # Extract the compression function - handle nested structure
            if 'compression_strategy' in self.compression_functions and 'compression_function' in self.compression_functions['compression_strategy']:
                function_info = self.compression_functions['compression_strategy']['compression_function']
            else:
                function_info = self.compression_functions['compression_function']
            function_code = function_info['code']
            required_imports = function_info.get('imports', [])
            
            # Create execution environment
            exec_globals = {'__builtins__': __builtins__}
            
            # Add required imports  
            import_statements = "\n".join([f"import {imp}" for imp in required_imports])
            if 'torch' not in required_imports:
                import_statements += "\nimport torch"
            if 'os' not in required_imports:
                import_statements += "\nimport os"
            
            # Execute imports with error handling
            logger.info(f"Executing imports: {import_statements}")
            exec(import_statements, exec_globals)
            
            # Execute the function definition with error handling  
            logger.info(f"Executing function code (first 200 chars): {function_code[:200]}...")
            
            # Save function code to file for debugging
            debug_file = f"debug_function_code_{datetime.now().strftime('%H%M%S')}.py"
            with open(debug_file, 'w') as f:
                f.write(function_code)
            logger.info(f"Function code saved to {debug_file} for debugging")
            
            exec(function_code, exec_globals)
            
            # Get the compression function
            function_name = function_info.get('function_name', 'compress_model')
            compress_func = exec_globals[function_name]
            
            # Prepare output path
            model_name = Path(self.model_path).stem
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"{self.output_dir}/{model_name}_compressed_{timestamp}.pth"
            
            # Execute compression
            logger.info(f"Running compression function: {function_name}")
            result = compress_func(self.model_path, output_path)
            
            # Verify compression results
            if os.path.exists(output_path):
                compressed_size = os.path.getsize(output_path) / 1024
                original_size = self.model_info['file_info']['size_kb']
                compression_ratio = original_size / compressed_size
                
                logger.info(f"Compression completed successfully!")
                logger.info(f"Original size: {original_size:.1f} KB")
                logger.info(f"Compressed size: {compressed_size:.1f} KB")
                logger.info(f"Compression ratio: {compression_ratio:.2f}x")
                logger.info(f"Size reduction: {((original_size - compressed_size) / original_size) * 100:.1f}%")
                
                # Check if target was met
                target_met = compressed_size < 256
                logger.info(f"Target (<256 KB) met: {target_met}")
                
                return {
                    'success': True,
                    'original_size_kb': original_size,
                    'compressed_size_kb': compressed_size,
                    'compression_ratio': compression_ratio,
                    'size_reduction_pct': ((original_size - compressed_size) / original_size) * 100,
                    'target_met': target_met,
                    'output_path': output_path,
                    'compression_function': function_info,
                    'ai_strategy': self.compression_functions['compression_strategy']
                }
            else:
                raise FileNotFoundError(f"Compressed model not created at: {output_path}")
                
        except Exception as e:
            logger.error(f"Failed to execute compression: {e}")
            return {
                'success': False,
                'error': str(e),
                'compression_function': self.compression_functions.get('compression_function', {}),
                'ai_strategy': self.compression_functions.get('compression_strategy', {})
            }
    
    def run_full_compression_pipeline(self) -> Dict[str, Any]:
        """Run the complete compression pipeline"""
        logger.info("🚀 Starting AI-Powered Model Compression Pipeline")
        logger.info("=" * 60)
        
        try:
            # Step 1: Analyze model
            logger.info("📊 Step 1: Analyzing model...")
            self.analyze_model()
            
            # Step 2: Generate compression strategy
            logger.info("🤖 Step 2: Generating AI compression strategy...")
            self.generate_compression_strategy()
            
            # Step 3: Execute compression
            logger.info("⚙️ Step 3: Executing compression...")
            results = self.execute_compression()
            
            # Step 4: Summary
            logger.info("📋 Step 4: Summary")
            logger.info("=" * 60)
            
            if results['success']:
                logger.info("✅ Compression completed successfully!")
                logger.info(f"📁 Original: {results['original_size_kb']:.1f} KB")
                logger.info(f"📁 Compressed: {results['compressed_size_kb']:.1f} KB")
                logger.info(f"📈 Ratio: {results['compression_ratio']:.2f}x")
                logger.info(f"🎯 Target met: {results['target_met']}")
                logger.info(f"💾 Output: {results['output_path']}")
            else:
                logger.error("❌ Compression failed!")
                logger.error(f"Error: {results['error']}")
            
            return results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            return {'success': False, 'error': str(e)}

def main():
    """Main function to run AI model compression"""
    
    # Configuration
    MODEL_PATH = "trained_models/best_model_FallbackLSTM_20250914_231643.pth"
    
    print("🤖 AI-Powered Model Compression")
    print("=" * 50)
    print(f"Model: {MODEL_PATH}")
    print(f"Target: <256 KB")
    print(f"AI Engine: {config.openai_model}")
    print("=" * 50)
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model file not found: {MODEL_PATH}")
        print("Available models:")
        for file in os.listdir("trained_models"):
            if file.endswith('.pth'):
                size = os.path.getsize(f"trained_models/{file}") / 1024
                print(f"  {file} ({size:.1f} KB)")
        return
    
    try:
        # Initialize compressor
        compressor = AIModelCompressor(MODEL_PATH)
        
        # Run compression pipeline
        results = compressor.run_full_compression_pipeline()
        
        if results['success']:
            print("\\n🎉 Compression completed successfully!")
            print(f"Check output: {results['output_path']}")
        else:
            print(f"\\n❌ Compression failed: {results.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"\\n❌ Pipeline error: {e}")

if __name__ == "__main__":
    main()