"""
AI Code Generator
Uses GPT to generate complete training functions as executable code
"""

import json
import logging
import torch
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from openai import OpenAI
from config import config
import os
from pathlib import Path

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
    
    def _create_prompt(self, data_profile: Dict[str, Any], input_shape: tuple, num_classes: int) -> str:
        """Create prompt for code generation"""
        
        prompt = f"""You are a machine learning expert. Generate a complete PyTorch training function for this data.

DATA CHARACTERISTICS:
{json.dumps(data_profile, indent=2)}

INPUT SHAPE: {input_shape}
NUM CLASSES: {num_classes}

TASK: Generate a complete training function that takes PyTorch tensors as input.

REQUIREMENTS:
1. Function signature: train_model(X_train, y_train, X_val, y_val, device, **hyperparams)
2. Function should build the model architecture from scratch
3. Include training loop with validation
4. Return trained model and training metrics
5. Use appropriate model architecture for the data type
6. Include proper error handling
7. Use hyperparameters as function parameters with defaults

RESPONSE FORMAT (JSON only, no other text):
{{
    "model_name": "CustomLSTMClassifier",
    "training_code": "def train_model(X_train, y_train, X_val, y_val, device='cpu', lr=0.001, epochs=10, batch_size=64, hidden_size=128, dropout=0.2, num_layers=2):\\n    import torch\\n    import torch.nn as nn\\n    import torch.optim as optim\\n    from torch.utils.data import TensorDataset, DataLoader\\n    \\n    # Model definition\\n    class CustomModel(nn.Module):\\n        def __init__(self, input_size, hidden_size, num_classes, num_layers, dropout):\\n            super().__init__()\\n            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)\\n            self.fc = nn.Linear(hidden_size, num_classes)\\n            self.dropout = nn.Dropout(dropout)\\n        \\n        def forward(self, x):\\n            lstm_out, _ = self.lstm(x)\\n            last_output = lstm_out[:, -1, :]\\n            output = self.dropout(last_output)\\n            return self.fc(output)\\n    \\n    # Data preparation\\n    train_dataset = TensorDataset(X_train, y_train)\\n    val_dataset = TensorDataset(X_val, y_val)\\n    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)\\n    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)\\n    \\n    # Model initialization\\n    input_size = X_train.shape[-1]\\n    model = CustomModel(input_size, hidden_size, {num_classes}, num_layers, dropout)\\n    model.to(device)\\n    \\n    # Training setup\\n    criterion = nn.CrossEntropyLoss()\\n    optimizer = optim.Adam(model.parameters(), lr=lr)\\n    \\n    # Training loop\\n    model.train()\\n    train_losses = []\\n    val_accuracies = []\\n    \\n    for epoch in range(epochs):\\n        epoch_loss = 0\\n        for batch_X, batch_y in train_loader:\\n            batch_X, batch_y = batch_X.to(device), batch_y.to(device)\\n            \\n            optimizer.zero_grad()\\n            outputs = model(batch_X)\\n            loss = criterion(outputs, batch_y)\\n            loss.backward()\\n            optimizer.step()\\n            \\n            epoch_loss += loss.item()\\n        \\n        # Validation\\n        model.eval()\\n        correct = 0\\n        total = 0\\n        with torch.no_grad():\\n            for batch_X, batch_y in val_loader:\\n                batch_X, batch_y = batch_X.to(device), batch_y.to(device)\\n                outputs = model(batch_X)\\n                _, predicted = torch.max(outputs.data, 1)\\n                total += batch_y.size(0)\\n                correct += (predicted == batch_y).sum().item()\\n        \\n        val_acc = correct / total\\n        train_losses.append(epoch_loss / len(train_loader))\\n        val_accuracies.append(val_acc)\\n        model.train()\\n    \\n    # Final metrics\\n    model.eval()\\n    final_metrics = {{'val_accuracy': val_accuracies[-1], 'final_loss': train_losses[-1]}}\\n    \\n    return model, final_metrics",
    "hyperparameters": {{
        "lr": 0.001,
        "epochs": 10,
        "batch_size": 64,
        "hidden_size": 128,
        "dropout": 0.2,
        "num_layers": 2
    }},
    "reasoning": "LSTM architecture chosen for sequence data with temporal dependencies. Custom implementation allows for flexible hyperparameter tuning.",
    "confidence": 0.9,
    "bo_parameters": ["lr", "epochs", "batch_size", "hidden_size", "dropout", "num_layers"]
}}

IMPORTANT GUIDELINES:
- Generate complete, executable Python code
- Use proper PyTorch conventions
- Include validation during training
- Handle different data types (sequence, tabular, image)
- Use appropriate model architectures for data characteristics
- Include proper tensor shapes and device handling
- All hyperparameters should be tunable via function parameters
- Code should be production-ready and error-free
- Return both model and metrics dictionary
- **CRITICAL: Keep model size small - final compressed model must be under 256K. Choose lightweight architectures with fewer parameters, smaller hidden sizes, and fewer layers.**
"""

        return prompt
    
    def _call_openai_api(self, prompt: str) -> str:
        """Call OpenAI API"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a machine learning expert. Generate complete, executable PyTorch training code. Respond only with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=3000
        )
        return response.choices[0].message.content
    
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
            data = json.loads(json_str)
            
            # Validate required fields
            required_fields = ["model_name", "training_code", "hyperparameters", "reasoning", "confidence", "bo_parameters"]
            for field in required_fields:
                if field not in data:
                    raise ValueError(f"Missing required field: {field}")
            
            return CodeRecommendation(
                model_name=data["model_name"],
                training_code=data["training_code"],
                hyperparameters=data["hyperparameters"],
                reasoning=data["reasoning"],
                confidence=float(data["confidence"]),
                bo_parameters=data["bo_parameters"]
            )
        
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.error(f"Failed to parse code recommendation: {e}")
            logger.error(f"Original response: {response}")
            raise ValueError(f"Failed to parse AI code recommendation: {e}")
    
    def generate_training_function(self, data_profile: Dict[str, Any], input_shape: tuple, num_classes: int) -> CodeRecommendation:
        """Generate training function code based on data characteristics"""
        prompt = self._create_prompt(data_profile, input_shape, num_classes)
        response = self._call_openai_api(prompt)
        recommendation = self._parse_recommendation(response)
        
        logger.info(f"AI generated training function: {recommendation.model_name}")
        logger.info(f"Confidence: {recommendation.confidence:.2f}")
        logger.info(f"Reasoning: {recommendation.reasoning}")
        
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
    
    def load_training_function(self, filepath: str) -> Dict[str, Any]:
        """Load training function from JSON file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data

class CodeValidator:
    """Validates generated training code"""
    
    def validate_code(self, code: str) -> bool:
        """Validate that code is syntactically correct"""
        try:
            compile(code, '<string>', 'exec')
            return True
        except SyntaxError as e:
            logger.error(f"Syntax error in generated code: {e}")
            return False
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return False
    
    def test_code_execution(self, code: str, X_sample: torch.Tensor, y_sample: torch.Tensor) -> bool:
        """Test code execution with sample data"""
        try:
            # Create small test datasets
            X_train = X_sample[:10]
            y_train = y_sample[:10]
            X_val = X_sample[10:15] if len(X_sample) > 15 else X_sample[:5]
            y_val = y_sample[10:15] if len(y_sample) > 15 else y_sample[:5]
            
            # Execute the function definition
            namespace = {}
            exec(code, namespace)
            
            # Get the train_model function
            if 'train_model' not in namespace:
                logger.error("train_model function not found in generated code")
                return False
            
            train_model = namespace['train_model']
            
            # Test with minimal parameters
            model, metrics = train_model(
                X_train, y_train, X_val, y_val, 
                device='cpu', epochs=1, batch_size=min(4, len(X_train))
            )
            
            # Validate outputs
            if not hasattr(model, 'eval'):
                logger.error("Returned object is not a PyTorch model")
                return False
            
            if not isinstance(metrics, dict):
                logger.error("Metrics should be a dictionary")
                return False
            
            logger.info("Code validation successful")
            return True
            
        except Exception as e:
            logger.error(f"Code execution test failed: {e}")
            return False

class FallbackCodeGenerator:
    """Rule-based fallback code generator"""
    
    def generate_fallback_code(self, data_profile: Dict[str, Any], input_shape: tuple, num_classes: int) -> CodeRecommendation:
        """Generate fallback training code when AI fails"""
        
        # Determine architecture based on data type
        if data_profile.get('is_sequence', False) or len(input_shape) > 1:
            # Sequence data - use LSTM
            model_name = "FallbackLSTM"
            training_code = self._generate_lstm_code(input_shape, num_classes)
            hyperparams = {"lr": 0.001, "epochs": 10, "batch_size": 64, "hidden_size": 128, "dropout": 0.2, "num_layers": 2}
            bo_params = ["lr", "epochs", "batch_size", "hidden_size", "dropout", "num_layers"]
            reasoning = "Fallback: LSTM selected for sequence data"
            
        else:
            # Tabular data - use MLP
            model_name = "FallbackMLP"
            training_code = self._generate_mlp_code(input_shape, num_classes)
            hyperparams = {"lr": 0.001, "epochs": 10, "batch_size": 64, "hidden_size": 256, "dropout": 0.2}
            bo_params = ["lr", "epochs", "batch_size", "hidden_size", "dropout"]
            reasoning = "Fallback: MLP selected for tabular data"
        
        return CodeRecommendation(
            model_name=model_name,
            training_code=training_code,
            hyperparameters=hyperparams,
            reasoning=reasoning,
            confidence=0.7,
            bo_parameters=bo_params
        )
    
    def _generate_lstm_code(self, input_shape: tuple, num_classes: int) -> str:
        """Generate LSTM training code"""
        return f"""def train_model(X_train, y_train, X_val, y_val, device='cpu', lr=0.001, epochs=10, batch_size=64, hidden_size=128, dropout=0.2, num_layers=2):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    
    # Model definition
    class FallbackLSTM(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes, num_layers, dropout):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout if num_layers > 1 else 0, batch_first=True)
            self.fc = nn.Linear(hidden_size, num_classes)
            self.dropout = nn.Dropout(dropout)
        
        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            last_output = lstm_out[:, -1, :]
            output = self.dropout(last_output)
            return self.fc(output)
    
    # Data preparation
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Model initialization
    input_size = X_train.shape[-1]
    model = FallbackLSTM(input_size, hidden_size, {num_classes}, num_layers, dropout)
    model.to(device)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    model.train()
    train_losses = []
    val_accuracies = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        val_acc = correct / total
        train_losses.append(epoch_loss / len(train_loader))
        val_accuracies.append(val_acc)
        model.train()
    
    # Final metrics
    model.eval()
    final_metrics = {{'val_accuracy': val_accuracies[-1], 'final_loss': train_losses[-1], 'macro_f1': val_accuracies[-1]}}
    
    return model, final_metrics"""
    
    def _generate_mlp_code(self, input_shape: tuple, num_classes: int) -> str:
        """Generate MLP training code"""
        return f"""def train_model(X_train, y_train, X_val, y_val, device='cpu', lr=0.001, epochs=10, batch_size=64, hidden_size=256, dropout=0.2):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    
    # Model definition
    class FallbackMLP(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes, dropout):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, num_classes)
            )
        
        def forward(self, x):
            if x.dim() > 2:
                x = x.view(x.size(0), -1)
            return self.network(x)
    
    # Data preparation
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Model initialization
    input_size = X_train.view(X_train.size(0), -1).shape[-1]
    model = FallbackMLP(input_size, hidden_size, {num_classes}, dropout)
    model.to(device)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    model.train()
    train_losses = []
    val_accuracies = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        val_acc = correct / total
        train_losses.append(epoch_loss / len(train_loader))
        val_accuracies.append(val_acc)
        model.train()
    
    # Final metrics
    model.eval()
    final_metrics = {{'val_accuracy': val_accuracies[-1], 'final_loss': train_losses[-1], 'macro_f1': val_accuracies[-1]}}
    
    return model, final_metrics"""

# Global instances
ai_code_generator = AICodeGenerator()
code_validator = CodeValidator()
fallback_code_generator = FallbackCodeGenerator()

def generate_training_code_for_data(data_profile: Dict[str, Any], input_shape: tuple, num_classes: int) -> CodeRecommendation:
    """
    Convenience function: Generate training code for data with fallback
    """
    try:
        # Try AI generation
        recommendation = ai_code_generator.generate_training_function(data_profile, input_shape, num_classes)
        
        # Validate the generated code
        if code_validator.validate_code(recommendation.training_code):
            return recommendation
        else:
            logger.warning("AI generated code failed validation, using fallback")
            raise ValueError("AI generated code failed validation")
        
    except Exception as e:
        logger.warning(f"AI code generation failed: {e}, using fallback")
        # Use rule-based fallback
        fallback_recommendation = fallback_code_generator.generate_fallback_code(data_profile, input_shape, num_classes)
        return fallback_recommendation