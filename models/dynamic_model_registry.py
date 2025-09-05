"""
Dynamic Model Registry
支持动态注册和创建神经网络模型
"""

import logging
from typing import Dict, Any, Type, Optional, Callable, List
from torch import nn
import importlib
import inspect

logger = logging.getLogger(__name__)

class ModelRegistry:
    """动态模型注册表"""
    
    def __init__(self):
        self.models: Dict[str, Type[nn.Module]] = {}
        self.model_factories: Dict[str, Callable] = {}
        self.model_metadata: Dict[str, Dict[str, Any]] = {}
        self._register_default_models()
    
    def _register_default_models(self):
        """注册默认模型"""
        # 导入默认模型
        from .tabular_mlp import TabMLP
        from .image_cnn import SmallCNN
        from .tiny_cnn_1d import TinyCNN1D
        
        # 注册模型
        self.register_model("TabMLP", TabMLP, {
            "type": "tabular",
            "description": "多层感知机，适用于表格数据",
            "input_types": ["tabular"],
            "parameters": ["in_dim", "num_classes", "hidden"]
        })
        
        self.register_model("SmallCNN", SmallCNN, {
            "type": "image",
            "description": "小型卷积神经网络，适用于图像数据",
            "input_types": ["image"],
            "parameters": ["in_channels", "num_classes", "hidden"]
        })
        
        self.register_model("TinyCNN1D", TinyCNN1D, {
            "type": "sequence",
            "description": "一维卷积神经网络，适用于序列数据",
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
        注册模型类
        
        Args:
            name: 模型名称
            model_class: 模型类
            metadata: 模型元数据
        """
        self.models[name] = model_class
        self.model_metadata[name] = metadata or {}
        logger.info(f"注册模型: {name}")
    
    def register_model_factory(
        self, 
        name: str, 
        factory_func: Callable, 
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        注册模型工厂函数
        
        Args:
            name: 模型名称
            factory_func: 工厂函数
            metadata: 模型元数据
        """
        self.model_factories[name] = factory_func
        self.model_metadata[name] = metadata or {}
        logger.info(f"注册模型工厂: {name}")
    
    def load_model_from_module(self, module_path: str, class_name: str, model_name: str):
        """
        从模块动态加载模型
        
        Args:
            module_path: 模块路径
            class_name: 类名
            model_name: 注册的模型名称
        """
        try:
            module = importlib.import_module(module_path)
            model_class = getattr(module, class_name)
            self.register_model(model_name, model_class)
            logger.info(f"从模块 {module_path} 加载模型 {class_name} 为 {model_name}")
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            raise
    
    def create_model(
        self, 
        model_name: str, 
        **kwargs
    ) -> nn.Module:
        """
        创建模型实例
        
        Args:
            model_name: 模型名称
            **kwargs: 模型参数
        
        Returns:
            nn.Module: 模型实例
        """
        if model_name in self.models:
            model_class = self.models[model_name]
            return model_class(**kwargs)
        elif model_name in self.model_factories:
            factory_func = self.model_factories[model_name]
            return factory_func(**kwargs)
        else:
            raise ValueError(f"未找到模型: {model_name}")
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """获取模型信息"""
        if model_name not in self.model_metadata:
            raise ValueError(f"未找到模型: {model_name}")
        
        info = self.model_metadata[model_name].copy()
        info["name"] = model_name
        info["is_factory"] = model_name in self.model_factories
        return info
    
    def list_models(self, model_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        列出所有模型
        
        Args:
            model_type: 过滤模型类型
        
        Returns:
            List[Dict]: 模型信息列表
        """
        models = []
        for name in self.models.keys():
            info = self.get_model_info(name)
            if model_type is None or info.get("type") == model_type:
                models.append(info)
        
        for name in self.model_factories.keys():
            if name not in self.models:  # 避免重复
                info = self.get_model_info(name)
                if model_type is None or info.get("type") == model_type:
                    models.append(info)
        
        return models
    
    def get_model_parameters(self, model_name: str) -> List[str]:
        """获取模型参数列表"""
        if model_name in self.models:
            model_class = self.models[model_name]
            sig = inspect.signature(model_class.__init__)
            return list(sig.parameters.keys())[1:]  # 排除self
        elif model_name in self.model_factories:
            factory_func = self.model_factories[model_name]
            sig = inspect.signature(factory_func)
            return list(sig.parameters.keys())
        else:
            raise ValueError(f"未找到模型: {model_name}")
    
    def validate_model_parameters(self, model_name: str, **kwargs) -> bool:
        """验证模型参数"""
        try:
            required_params = self.get_model_parameters(model_name)
            for param in required_params:
                if param not in kwargs:
                    logger.warning(f"缺少必需参数: {param}")
                    return False
            return True
        except Exception as e:
            logger.error(f"验证参数失败: {e}")
            return False

class ModelBuilder:
    """模型构建器"""
    
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
        构建模型
        
        Args:
            model_name: 模型名称
            input_shape: 输入形状
            num_classes: 类别数量
            **kwargs: 其他参数
        
        Returns:
            nn.Module: 构建的模型
        """
        # 验证参数
        if not self.registry.validate_model_parameters(model_name, **kwargs):
            logger.warning(f"模型参数验证失败: {model_name}")
        
        # 根据模型类型设置默认参数
        model_info = self.registry.get_model_info(model_name)
        model_type = model_info.get("type", "general")
        
        # 设置默认参数
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
        
        # 合并参数
        final_params = {**default_params, **kwargs}
        
        # 创建模型
        model = self.registry.create_model(model_name, **final_params)
        logger.info(f"构建模型: {model_name}, 参数: {final_params}")
        
        return model
    
    def build_model_from_recommendation(
        self, 
        recommendation, 
        input_shape: tuple, 
        num_classes: int
    ) -> nn.Module:
        """
        根据AI推荐构建模型
        
        Args:
            recommendation: AI模型推荐
            input_shape: 输入形状
            num_classes: 类别数量
        
        Returns:
            nn.Module: 构建的模型
        """
        # 使用推荐的超参数
        hyperparams = recommendation.hyperparameters.copy()
        
        # 构建模型
        model = self.build_model(
            model_name=recommendation.model_name,
            input_shape=input_shape,
            num_classes=num_classes,
            **hyperparams
        )
        
        logger.info(f"根据AI推荐构建模型: {recommendation.model_name}")
        logger.info(f"推荐理由: {recommendation.reasoning}")
        
        return model

# 全局注册表和构建器
model_registry = ModelRegistry()
model_builder = ModelBuilder(model_registry)

def register_model(name: str, model_class: Type[nn.Module], metadata: Optional[Dict[str, Any]] = None):
    """便捷函数：注册模型"""
    model_registry.register_model(name, model_class, metadata)

def build_model(model_name: str, input_shape: tuple, num_classes: int, **kwargs) -> nn.Module:
    """便捷函数：构建模型"""
    return model_builder.build_model(model_name, input_shape, num_classes, **kwargs)
