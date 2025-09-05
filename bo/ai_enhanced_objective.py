"""
AI增强的目标函数
集成通用数据转换器和AI模型选择器
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader
import logging

from adapters.universal_converter import convert_to_torch_dataset
from models.ai_model_selector import select_model_for_data
from models.dynamic_model_registry import build_model_from_recommendation
from train import train_one_model
from evaluation.evaluate import evaluate_model

logger = logging.getLogger(__name__)

class AIEnhancedObjective:
    """AI增强的目标函数类"""
    
    def __init__(self, data, labels=None, device="cpu", **kwargs):
        """
        初始化目标函数
        
        Args:
            data: 输入数据
            labels: 标签数据
            device: 设备
            **kwargs: 其他参数
        """
        self.data = data
        self.labels = labels
        self.device = device
        self.kwargs = kwargs
        
        # 预处理数据
        self._preprocess_data()
        
        # 获取AI推荐
        self._get_ai_recommendation()
    
    def _preprocess_data(self):
        """预处理数据"""
        logger.info("预处理数据...")
        
        # 转换数据
        self.dataset, self.collate_fn, self.data_profile = convert_to_torch_dataset(
            self.data, self.labels, **self.kwargs
        )
        
        logger.info(f"数据预处理完成: {self.data_profile}")
    
    def _get_ai_recommendation(self):
        """获取AI模型推荐"""
        logger.info("获取AI模型推荐...")
        
        self.recommendation = select_model_for_data(self.data_profile.to_dict())
        
        logger.info(f"AI推荐模型: {self.recommendation.model_name}")
        logger.info(f"推荐理由: {self.recommendation.reasoning}")
    
    def _determine_input_shape(self) -> tuple:
        """确定输入形状"""
        if self.data_profile.is_sequence:
            return (self.data_profile.feature_count,)
        elif self.data_profile.is_image:
            if (self.data_profile.channels and 
                self.data_profile.height and 
                self.data_profile.width):
                return (self.data_profile.channels, 
                       self.data_profile.height, 
                       self.data_profile.width)
            else:
                return (3, 32, 32)  # 默认图像尺寸
        elif self.data_profile.is_tabular:
            return (self.data_profile.feature_count,)
        else:
            return (self.data_profile.feature_count,)
    
    def _create_model(self, hparams: Dict[str, Any]) -> torch.nn.Module:
        """创建模型"""
        # 确定输入形状和类别数
        input_shape = self._determine_input_shape()
        num_classes = self.data_profile.label_count if self.data_profile.has_labels else 2
        
        # 合并超参数
        model_params = self.recommendation.hyperparameters.copy()
        model_params.update(hparams)
        
        # 构建模型
        model = build_model_from_recommendation(
            self.recommendation, input_shape, num_classes
        )
        
        return model
    
    def __call__(self, hparams: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """
        目标函数调用
        
        Args:
            hparams: 超参数
        
        Returns:
            Tuple[float, Dict]: (目标值, 详细指标)
        """
        try:
            # 创建模型
            model = self._create_model(hparams)
            model.to(self.device)
            
            # 创建数据加载器
            batch_size = hparams.get('batch_size', 64)
            loader = DataLoader(
                self.dataset, 
                batch_size=batch_size, 
                shuffle=True, 
                collate_fn=self.collate_fn
            )
            
            # 训练模型
            epochs = hparams.get('epochs', 3)
            lr = hparams.get('lr', 1e-3)
            
            trained_model = train_one_model(
                model, loader, device=self.device, epochs=epochs, lr=lr
            )
            
            # 评估模型
            metrics = evaluate_model(trained_model, loader, device=self.device)
            
            # 返回目标值（使用macro_f1作为主要指标）
            objective_value = metrics.get("macro_f1", 0.0)
            
            # 添加额外信息
            metrics.update({
                "model_name": self.recommendation.model_name,
                "confidence": self.recommendation.confidence,
                "data_type": self.data_profile.data_type,
                "sample_count": self.data_profile.sample_count,
                "feature_count": self.data_profile.feature_count
            })
            
            logger.info(f"目标函数评估完成: {objective_value:.4f}")
            
            return float(objective_value), metrics
        
        except Exception as e:
            logger.error(f"目标函数评估失败: {e}")
            return 0.0, {"error": str(e)}

def create_ai_enhanced_objective(data, labels=None, device="cpu", **kwargs):
    """
    创建AI增强的目标函数
    
    Args:
        data: 输入数据
        labels: 标签数据
        device: 设备
        **kwargs: 其他参数
    
    Returns:
        AIEnhancedObjective: 目标函数实例
    """
    return AIEnhancedObjective(data, labels, device, **kwargs)

def objective_for_dataset_ai_enhanced(
    data, 
    labels=None, 
    device="cpu", 
    hparams: Optional[Dict[str, Any]] = None
) -> Tuple[float, Dict[str, Any]]:
    """
    便捷函数：为数据集创建AI增强的目标函数并评估
    
    Args:
        data: 输入数据
        labels: 标签数据
        device: 设备
        hparams: 超参数
    
    Returns:
        Tuple[float, Dict]: (目标值, 详细指标)
    """
    hparams = hparams or {"lr": 1e-3, "epochs": 3, "hidden": 64}
    
    objective = create_ai_enhanced_objective(data, labels, device)
    return objective(hparams)

# 向后兼容的别名
objective_for_dataset = objective_for_dataset_ai_enhanced
