"""
AI Model Selector
使用ChatGPT API根据数据特征自动选择最适合的神经网络架构
"""

import json
import logging
from typing import Dict, Any, List, Optional, Tuple
import requests
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ModelRecommendation:
    """模型推荐结果"""
    model_name: str
    model_type: str
    architecture: str
    input_shape: Tuple[int, ...]
    reasoning: str
    confidence: float
    hyperparameters: Dict[str, Any]

class AIModelSelector:
    """AI模型选择器"""
    
    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://api.openai.com/v1"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url
        self.model = "gpt-4"  # 使用GPT-4获得更好的推荐
        
        if not self.api_key:
            logger.warning("未设置OPENAI_API_KEY，将使用默认模型选择")
    
    def _create_prompt(self, data_profile: Dict[str, Any]) -> str:
        """创建用于模型选择的提示词"""
        prompt = f"""
你是一个机器学习专家，需要根据数据特征推荐最适合的神经网络架构。

数据特征信息：
{json.dumps(data_profile, indent=2, ensure_ascii=False)}

请根据以下信息推荐最适合的神经网络架构：

1. 数据特征分析：
   - 数据类型：{data_profile.get('data_type', 'unknown')}
   - 数据形状：{data_profile.get('shape', 'unknown')}
   - 样本数量：{data_profile.get('sample_count', 0)}
   - 特征数量：{data_profile.get('feature_count', 0)}
   - 是否有标签：{data_profile.get('has_labels', False)}
   - 标签数量：{data_profile.get('label_count', 0)}

2. 数据特性：
   - 是否为序列数据：{data_profile.get('is_sequence', False)}
   - 是否为图像数据：{data_profile.get('is_image', False)}
   - 是否为表格数据：{data_profile.get('is_tabular', False)}

请从以下预定义的模型类型中选择最合适的：

**表格数据模型：**
- TabMLP: 多层感知机，适用于表格数据
- TabTransformer: 基于Transformer的表格数据模型
- TabNet: 可解释的表格数据模型

**图像数据模型：**
- SmallCNN: 小型卷积神经网络
- ResNet: 残差网络
- EfficientNet: 高效的卷积神经网络
- VisionTransformer: 基于Transformer的图像模型

**序列数据模型：**
- TinyCNN1D: 一维卷积神经网络
- LSTM: 长短期记忆网络
- GRU: 门控循环单元
- Transformer: 基于注意力机制的序列模型

**通用模型：**
- MLP: 通用多层感知机
- AutoEncoder: 自编码器

请以JSON格式返回推荐结果，格式如下：
{{
    "model_name": "推荐的模型名称",
    "model_type": "模型类型（如：tabular, image, sequence, general）",
    "architecture": "具体的架构描述",
    "input_shape": [输入形状的数组],
    "reasoning": "推荐理由的详细说明",
    "confidence": 0.95,
    "hyperparameters": {{
        "hidden_size": 128,
        "num_layers": 3,
        "dropout": 0.2,
        "learning_rate": 0.001
    }}
}}

请确保推荐结果基于数据特征，并给出合理的推荐理由。
"""
        return prompt
    
    def _call_openai_api(self, prompt: str) -> str:
        """调用OpenAI API"""
        if not self.api_key:
            raise ValueError("未设置OPENAI_API_KEY")
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "你是一个专业的机器学习专家，擅长根据数据特征推荐最适合的神经网络架构。"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.3,
            "max_tokens": 1000
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
        
        except requests.exceptions.RequestException as e:
            logger.error(f"OpenAI API调用失败: {e}")
            raise
        except KeyError as e:
            logger.error(f"API响应格式错误: {e}")
            raise
    
    def _parse_recommendation(self, response: str) -> ModelRecommendation:
        """解析API响应为模型推荐"""
        try:
            # 尝试提取JSON部分
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                raise ValueError("响应中未找到JSON格式")
            
            json_str = response[start_idx:end_idx]
            data = json.loads(json_str)
            
            return ModelRecommendation(
                model_name=data.get("model_name", "Unknown"),
                model_type=data.get("model_type", "general"),
                architecture=data.get("architecture", "Unknown"),
                input_shape=tuple(data.get("input_shape", [])),
                reasoning=data.get("reasoning", "No reasoning provided"),
                confidence=float(data.get("confidence", 0.5)),
                hyperparameters=data.get("hyperparameters", {})
            )
        
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.error(f"解析推荐结果失败: {e}")
            logger.error(f"原始响应: {response}")
            
            # 返回默认推荐
            return ModelRecommendation(
                model_name="MLP",
                model_type="general",
                architecture="Multi-layer Perceptron",
                input_shape=(data_profile.get("feature_count", 10),),
                reasoning="解析失败，使用默认MLP模型",
                confidence=0.1,
                hyperparameters={"hidden_size": 64, "num_layers": 2}
            )
    
    def select_model(self, data_profile: Dict[str, Any]) -> ModelRecommendation:
        """
        根据数据特征选择最适合的模型
        
        Args:
            data_profile: 数据特征档案
        
        Returns:
            ModelRecommendation: 模型推荐结果
        """
        try:
            if not self.api_key:
                logger.warning("未设置API密钥，使用默认模型选择")
                return self._get_default_recommendation(data_profile)
            
            prompt = self._create_prompt(data_profile)
            response = self._call_openai_api(prompt)
            recommendation = self._parse_recommendation(response)
            
            logger.info(f"AI推荐模型: {recommendation.model_name} (置信度: {recommendation.confidence:.2f})")
            logger.info(f"推荐理由: {recommendation.reasoning}")
            
            return recommendation
        
        except Exception as e:
            logger.error(f"模型选择失败: {e}")
            return self._get_default_recommendation(data_profile)
    
    def _get_default_recommendation(self, data_profile: Dict[str, Any]) -> ModelRecommendation:
        """获取默认模型推荐（当API不可用时）"""
        data_type = data_profile.get("data_type", "unknown")
        is_image = data_profile.get("is_image", False)
        is_sequence = data_profile.get("is_sequence", False)
        is_tabular = data_profile.get("is_tabular", False)
        
        if is_image:
            return ModelRecommendation(
                model_name="SmallCNN",
                model_type="image",
                architecture="Small Convolutional Neural Network",
                input_shape=(3, 32, 32),  # 默认图像尺寸
                reasoning="检测到图像数据，推荐使用卷积神经网络",
                confidence=0.8,
                hyperparameters={"hidden_size": 64, "num_layers": 2}
            )
        elif is_sequence:
            return ModelRecommendation(
                model_name="TinyCNN1D",
                model_type="sequence",
                architecture="1D Convolutional Neural Network",
                input_shape=(data_profile.get("feature_count", 10),),
                reasoning="检测到序列数据，推荐使用一维卷积神经网络",
                confidence=0.8,
                hyperparameters={"hidden_size": 64, "num_layers": 2}
            )
        elif is_tabular:
            return ModelRecommendation(
                model_name="TabMLP",
                model_type="tabular",
                architecture="Multi-layer Perceptron for Tabular Data",
                input_shape=(data_profile.get("feature_count", 10),),
                reasoning="检测到表格数据，推荐使用多层感知机",
                confidence=0.8,
                hyperparameters={"hidden_size": 64, "num_layers": 2}
            )
        else:
            return ModelRecommendation(
                model_name="MLP",
                model_type="general",
                architecture="Multi-layer Perceptron",
                input_shape=(data_profile.get("feature_count", 10),),
                reasoning="未知数据类型，使用通用多层感知机",
                confidence=0.5,
                hyperparameters={"hidden_size": 64, "num_layers": 2}
            )

# 全局模型选择器实例
ai_model_selector = AIModelSelector()

def select_model_for_data(data_profile: Dict[str, Any]) -> ModelRecommendation:
    """
    便捷函数：为数据选择最适合的模型
    
    Args:
        data_profile: 数据特征档案
    
    Returns:
        ModelRecommendation: 模型推荐结果
    """
    return ai_model_selector.select_model(data_profile)
