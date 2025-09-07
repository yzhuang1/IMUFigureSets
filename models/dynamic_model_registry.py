"""
Dynamic Model Registry
Supports dynamic registration and creation of neural network models
"""

import logging
from typing import Dict, Any, Type, Optional, Callable, List
from torch import nn
import importlib
import inspect

logger = logging.getLogger(__name__)

class ModelRegistry:
    """Dynamic model registry"""
    
    def __init__(self):
        self.models: Dict[str, Type[nn.Module]] = {}
        self.model_factories: Dict[str, Callable] = {}
        self.model_metadata: Dict[str, Dict[str, Any]] = {}
        self._register_default_models()
    
    def _register_default_models(self):
        """Register default models"""
        # Import default models
        from .tabular_mlp import TabMLP
        from .image_cnn import SmallCNN
        from .tiny_cnn_1d import TinyCNN1D
        
        # Register models
        self.register_model("TabMLP", TabMLP, {
            "type": "tabular",
            "description": "Multi-layer perceptron for tabular data",
            "input_types": ["tabular"],
            "parameters": ["in_dim", "num_classes", "hidden"]
        })
        
        self.register_model("SmallCNN", SmallCNN, {
            "type": "image",
            "description": "Small convolutional neural network for image data",
            "input_types": ["image"],
            "parameters": ["in_channels", "num_classes", "hidden"]
        })
        
        self.register_model("TinyCNN1D", TinyCNN1D, {
            "type": "sequence",
            "description": "1D convolutional neural network for sequence data",
            "input_types": ["sequence", "timeseries"],
            "parameters": ["in_channels", "num_classes", "hidden"]
        })
    
    def register_model(
        self, 
        name: str, 
        model_class: Type[nn.Module], 
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Register model class
        
        Args:
            name: Model name
            model_class: Model class
            metadata: Model metadata
        """
        self.models[name] = model_class
        self.model_metadata[name] = metadata or {}
        logger.info(f"Registered model: {name}")
    
    def register_model_factory(
        self, 
        name: str, 
        factory_func: Callable, 
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Register model factory function
        
        Args:
            name: Model name
            factory_func: Factory function
            metadata: Model metadata
        """
        self.model_factories[name] = factory_func
        self.model_metadata[name] = metadata or {}
        logger.info(f"Registered model factory: {name}")
    
    def load_model_from_module(self, module_path: str, class_name: str, model_name: str):
        """
        Dynamically load model from module
        
        Args:
            module_path: Module path
            class_name: Class name
            model_name: Registered model name
        """
        try:
            module = importlib.import_module(module_path)
            model_class = getattr(module, class_name)
            self.register_model(model_name, model_class)
            logger.info(f"Loaded model {class_name} from module {module_path} as {model_name}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def create_model(
        self, 
        model_name: str, 
        **kwargs
    ) -> nn.Module:
        """
        Create model instance
        
        Args:
            model_name: Model name
            **kwargs: Model parameters
        
        Returns:
            nn.Module: Model instance
        """
        if model_name in self.models:
            model_class = self.models[model_name]
            return model_class(**kwargs)
        elif model_name in self.model_factories:
            factory_func = self.model_factories[model_name]
            return factory_func(**kwargs)
        else:
            raise ValueError(f"Model not found: {model_name}")
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get model information"""
        if model_name not in self.model_metadata:
            raise ValueError(f"Model not found: {model_name}")
        
        info = self.model_metadata[model_name].copy()
        info["name"] = model_name
        info["is_factory"] = model_name in self.model_factories
        return info
    
    def list_models(self, model_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all models
        
        Args:
            model_type: Filter model type
        
        Returns:
            List[Dict]: List of model information
        """
        models = []
        for name in self.models.keys():
            info = self.get_model_info(name)
            if model_type is None or info.get("type") == model_type:
                models.append(info)
        
        for name in self.model_factories.keys():
            if name not in self.models:  # Avoid duplicates
                info = self.get_model_info(name)
                if model_type is None or info.get("type") == model_type:
                    models.append(info)
        
        return models
    
    def get_model_parameters(self, model_name: str) -> List[str]:
        """Get model parameter list"""
        if model_name in self.models:
            model_class = self.models[model_name]
            sig = inspect.signature(model_class.__init__)
            return list(sig.parameters.keys())[1:]  # Exclude self
        elif model_name in self.model_factories:
            factory_func = self.model_factories[model_name]
            sig = inspect.signature(factory_func)
            return list(sig.parameters.keys())
        else:
            raise ValueError(f"Model not found: {model_name}")
    
    def validate_model_parameters(self, model_name: str, **kwargs) -> bool:
        """Validate model parameters"""
        try:
            required_params = self.get_model_parameters(model_name)
            for param in required_params:
                if param not in kwargs:
                    logger.warning(f"Missing required parameter: {param}")
                    return False
            return True
        except Exception as e:
            logger.error(f"Parameter validation failed: {e}")
            return False

class ModelBuilder:
    """Model builder"""
    
    def __init__(self, registry: Optional[ModelRegistry] = None):
        self.registry = registry or ModelRegistry()
    
    def build_model(
        self, 
        model_name: str, 
        input_shape: tuple, 
        num_classes: int,
        **kwargs
    ) -> nn.Module:
        """
        Build model
        
        Args:
            model_name: Model name
            input_shape: Input shape
            num_classes: Number of classes
            **kwargs: Other parameters
        
        Returns:
            nn.Module: Built model
        """
        # Validate parameters
        if not self.registry.validate_model_parameters(model_name, **kwargs):
            logger.warning(f"Model parameter validation failed: {model_name}")
        
        # Set default parameters based on model type
        model_info = self.registry.get_model_info(model_name)
        model_type = model_info.get("type", "general")
        
        # Set default parameters
        default_params = {
            "num_classes": num_classes,
            "hidden": kwargs.get("hidden", 64)
        }
        
        if model_type == "tabular":
            default_params["in_dim"] = input_shape[0]
        elif model_type == "image":
            if len(input_shape) == 3:
                default_params["in_channels"] = input_shape[0]
            else:
                default_params["in_channels"] = 1
        elif model_type in ["sequence", "timeseries"]:
            default_params["in_channels"] = input_shape[-1]
        
        # Merge parameters
        final_params = {**default_params, **kwargs}
        
        # Create model
        model = self.registry.create_model(model_name, **final_params)
        logger.info(f"Built model: {model_name}, parameters: {final_params}")
        
        return model
    
    def build_model_from_recommendation(
        self, 
        recommendation, 
        input_shape: tuple, 
        num_classes: int
    ) -> nn.Module:
        """
        Build model based on AI recommendation
        
        Args:
            recommendation: AI model recommendation
            input_shape: Input shape
            num_classes: Number of classes
        
        Returns:
            nn.Module: Built model
        """
        # Use recommended hyperparameters
        hyperparams = recommendation.hyperparameters.copy()
        
        # Map common AI parameter names to model parameter names
        param_mapping = {
            "hidden_size": "hidden",
            "num_layers": "layers",
            "learning_rate": "lr",
            "dropout_rate": "dropout"
        }
        
        # Apply parameter mapping
        mapped_params = {}
        for key, value in hyperparams.items():
            mapped_key = param_mapping.get(key, key)
            mapped_params[mapped_key] = value
        
        # Get valid parameters for this model to filter out unsupported ones
        try:
            valid_params = self.registry.get_model_parameters(recommendation.model_name)
            filtered_params = {k: v for k, v in mapped_params.items() if k in valid_params}
            logger.info(f"Filtered parameters for {recommendation.model_name}: {filtered_params}")
        except Exception:
            # If we can't get valid parameters, use all mapped parameters
            filtered_params = mapped_params
            logger.warning(f"Could not validate parameters for {recommendation.model_name}, using all: {filtered_params}")
        
        # Build model
        model = self.build_model(
            model_name=recommendation.model_name,
            input_shape=input_shape,
            num_classes=num_classes,
            **filtered_params
        )
        
        logger.info(f"Built model based on AI recommendation: {recommendation.model_name}")
        logger.info(f"Recommendation reason: {recommendation.reasoning}")
        logger.info(f"Mapped parameters: {mapped_params}")
        
        return model

# Global registry and builder
model_registry = ModelRegistry()
model_builder = ModelBuilder(model_registry)

def register_model(name: str, model_class: Type[nn.Module], metadata: Optional[Dict[str, Any]] = None):
    """Convenience function: Register model"""
    model_registry.register_model(name, model_class, metadata)

def build_model(model_name: str, input_shape: tuple, num_classes: int, **kwargs) -> nn.Module:
    """Convenience function: Build model"""
    return model_builder.build_model(model_name, input_shape, num_classes, **kwargs)

def build_model_from_recommendation(recommendation, input_shape: tuple, num_classes: int) -> nn.Module:
    """Convenience function: Build model from AI recommendation"""
    return model_builder.build_model_from_recommendation(recommendation, input_shape, num_classes)
