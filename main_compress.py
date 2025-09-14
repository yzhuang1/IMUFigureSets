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
        
        try:
            response = self.client.chat.completions.create(
                model=config.openai_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert in neural network compression and optimization. You specialize in creating efficient compression functions that maintain model accuracy while achieving aggressive size reductions."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                max_completion_tokens=3000
            )
            
            # Parse the response
            logger.info(f"GPT Response structure: {response}")
            logger.info(f"Response choices: {len(response.choices) if hasattr(response, 'choices') else 'No choices'}")
            
            if response.choices and len(response.choices) > 0:
                response_content = response.choices[0].message.content
                logger.info(f"Response content length: {len(response_content) if response_content else 0}")
                logger.info("GPT-5 compression strategy generated successfully")
            else:
                raise ValueError("No response choices available")
            
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
                raise
            
        except Exception as e:
            logger.error(f"Failed to generate compression strategy: {e}")
            raise
    
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

RESPONSE FORMAT:
Please respond with a JSON object containing:

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
    
    def _parse_gpt_response(self, response_content: str) -> Dict[str, Any]:
        """Parse GPT-5 response and extract JSON"""
        try:
            # Try to find JSON in the response
            start_idx = response_content.find('{')
            end_idx = response_content.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON found in response")
            
            json_str = response_content[start_idx:end_idx]
            compression_strategy = json.loads(json_str)
            
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
            # Extract the compression function
            function_info = self.compression_functions['compression_function']
            function_code = function_info['code']
            required_imports = function_info.get('imports', [])
            
            # Create execution environment
            exec_globals = {'__builtins__': __builtins__}
            
            # Add required imports
            import_statements = "\\n".join([f"import {imp}" for imp in required_imports])
            if 'torch' not in required_imports:
                import_statements += "\\nimport torch"
            if 'os' not in required_imports:
                import_statements += "\\nimport os"
            
            # Execute imports
            exec(import_statements, exec_globals)
            
            # Execute the function definition
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