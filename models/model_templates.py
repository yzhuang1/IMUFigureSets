"""
Pre-built Model Templates for AI Selection
Safe, validated PyTorch model architectures that GPT can configure
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Type
import logging

logger = logging.getLogger(__name__)

class LSTMTemplate(nn.Module):
    """LSTM model for sequence data"""
    
    def __init__(self, input_size: int, hidden_size: int, num_classes: int, 
                 num_layers: int = 1, dropout: float = 0.2, bidirectional: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size, 
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        # Adjust output size for bidirectional
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_output_size, num_classes)
        
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        # Take last timestep output
        last_output = lstm_out[:, -1, :]
        output = self.dropout(last_output)
        output = self.fc(output)
        return output

class GRUTemplate(nn.Module):
    """GRU model for sequence data (faster than LSTM)"""
    
    def __init__(self, input_size: int, hidden_size: int, num_classes: int,
                 num_layers: int = 1, dropout: float = 0.2, bidirectional: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers, 
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        gru_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(gru_output_size, num_classes)
        
    def forward(self, x):
        gru_out, _ = self.gru(x)
        last_output = gru_out[:, -1, :]
        output = self.dropout(last_output)
        output = self.fc(output)
        return output

class CNN1DTemplate(nn.Module):
    """1D CNN for sequence pattern recognition"""
    
    def __init__(self, input_size: int, num_classes: int, num_filters: int = 64,
                 kernel_sizes: list = None, dropout: float = 0.2, pool_size: int = 2):
        super().__init__()
        
        if kernel_sizes is None:
            kernel_sizes = [3, 5, 7]
            
        self.conv_layers = nn.ModuleList()
        
        # Multiple conv layers with different kernel sizes
        for kernel_size in kernel_sizes:
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(input_size, num_filters, kernel_size, padding=kernel_size//2),
                    nn.ReLU(),
                    nn.MaxPool1d(pool_size),
                    nn.Dropout(dropout)
                )
            )
        
        # Calculate output size (approximate)
        # This will be adjusted dynamically in forward pass
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)
        
    def forward(self, x):
        # x shape: (batch, seq_len, features) -> (batch, features, seq_len)
        x = x.transpose(1, 2)
        
        conv_outputs = []
        for conv_layer in self.conv_layers:
            conv_out = conv_layer(x)
            # Global average pooling
            pooled = self.adaptive_pool(conv_out).squeeze(-1)
            conv_outputs.append(pooled)
        
        # Concatenate all conv outputs
        combined = torch.cat(conv_outputs, dim=1)
        output = self.fc(combined)
        return output

class TransformerTemplate(nn.Module):
    """Transformer model for complex sequence relationships"""
    
    def __init__(self, input_size: int, num_classes: int, d_model: int = 128,
                 nhead: int = 8, num_layers: int = 2, dropout: float = 0.1, max_seq_len: int = 1000):
        super().__init__()
        
        self.d_model = d_model
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(max_seq_len, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)
        
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        seq_len = x.size(1)
        
        # Project input to d_model dimension
        x = self.input_projection(x)
        
        # Add positional encoding
        if seq_len <= self.pos_encoding.size(0):
            x = x + self.pos_encoding[:seq_len].unsqueeze(0)
        
        # Transformer encoding
        transformer_out = self.transformer(x)
        
        # Global average pooling over sequence dimension
        pooled = transformer_out.mean(dim=1)
        output = self.dropout(pooled)
        output = self.fc(output)
        return output

class MLPTemplate(nn.Module):
    """Multi-layer perceptron for tabular data"""
    
    def __init__(self, input_size: int, num_classes: int, hidden_sizes: list = None,
                 dropout: float = 0.2, activation: str = 'relu'):
        super().__init__()
        
        if hidden_sizes is None:
            hidden_sizes = [256, 128, 64]
            
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU() if activation == 'relu' else nn.Tanh(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        # Flatten if needed
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.network(x)

class HybridCNNLSTMTemplate(nn.Module):
    """Hybrid CNN-LSTM for complex sequence patterns"""
    
    def __init__(self, input_size: int, num_classes: int, cnn_filters: int = 64,
                 lstm_hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        
        # CNN feature extraction
        self.conv1 = nn.Conv1d(input_size, cnn_filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(cnn_filters, cnn_filters, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(cnn_filters, lstm_hidden, batch_first=True)
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden, num_classes)
        
    def forward(self, x):
        # x shape: (batch, seq_len, features) -> (batch, features, seq_len)
        x = x.transpose(1, 2)
        
        # CNN feature extraction
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        # Back to (batch, seq_len, features) for LSTM
        x = x.transpose(1, 2)
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        
        output = self.dropout(last_output)
        output = self.fc(output)
        return output

# Template registry
MODEL_TEMPLATES = {
    "LSTM": LSTMTemplate,
    "GRU": GRUTemplate, 
    "CNN1D": CNN1DTemplate,
    "Transformer": TransformerTemplate,
    "MLP": MLPTemplate,
    "HybridCNNLSTM": HybridCNNLSTMTemplate
}

def get_template_class(template_name: str) -> Type[nn.Module]:
    """Get template class by name"""
    if template_name not in MODEL_TEMPLATES:
        available = list(MODEL_TEMPLATES.keys())
        raise ValueError(f"Unknown template '{template_name}'. Available: {available}")
    
    return MODEL_TEMPLATES[template_name]

def create_model_from_template(template_name: str, config: Dict[str, Any]) -> nn.Module:
    """Create model instance from template and configuration"""
    template_class = get_template_class(template_name)
    
    try:
        model = template_class(**config)
        logger.info(f"Created {template_name} model with config: {config}")
        return model
    except Exception as e:
        logger.error(f"Failed to create {template_name} model: {e}")
        raise ValueError(f"Template instantiation failed: {e}")

def get_template_info():
    """Get information about available templates"""
    return {
        "LSTM": {
            "description": "Long Short-Term Memory network for sequence data",
            "best_for": "Sequential data, time series, ECG signals",
            "required_params": ["input_size", "hidden_size", "num_classes"],
            "optional_params": ["num_layers", "dropout", "bidirectional"]
        },
        "GRU": {
            "description": "Gated Recurrent Unit network (faster than LSTM)",
            "best_for": "Sequential data where speed is important",
            "required_params": ["input_size", "hidden_size", "num_classes"],
            "optional_params": ["num_layers", "dropout", "bidirectional"]
        },
        "CNN1D": {
            "description": "1D Convolutional Neural Network for pattern recognition",
            "best_for": "Sequential data with local patterns",
            "required_params": ["input_size", "num_classes"],
            "optional_params": ["num_filters", "kernel_sizes", "dropout", "pool_size"]
        },
        "Transformer": {
            "description": "Transformer encoder for complex sequence relationships",
            "best_for": "Complex sequential dependencies, attention mechanisms",
            "required_params": ["input_size", "num_classes"],
            "optional_params": ["d_model", "nhead", "num_layers", "dropout", "max_seq_len"]
        },
        "MLP": {
            "description": "Multi-layer perceptron for tabular data",
            "best_for": "Tabular/structured data, flattened features",
            "required_params": ["input_size", "num_classes"],
            "optional_params": ["hidden_sizes", "dropout", "activation"]
        },
        "HybridCNNLSTM": {
            "description": "CNN feature extraction + LSTM temporal modeling",
            "best_for": "Complex sequential patterns with local and temporal features",
            "required_params": ["input_size", "num_classes"],
            "optional_params": ["cnn_filters", "lstm_hidden", "dropout"]
        }
    }