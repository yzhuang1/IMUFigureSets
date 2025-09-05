"""
AI增强的贝叶斯优化运行脚本
集成通用数据转换器和AI模型选择器
"""

import logging
import numpy as np
import torch
from typing import Dict, Any, List, Tuple
import json
import time

from adapters.universal_converter import convert_to_torch_dataset
from models.ai_model_selector import select_model_for_data
from bo.ai_enhanced_objective import create_ai_enhanced_objective
from bo.run_bo import suggest, observe

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIEnhancedBO:
    """AI增强的贝叶斯优化类"""
    
    def __init__(self, data, labels=None, device="cpu", n_trials=20, **kwargs):
        """
        初始化AI增强的BO
        
        Args:
            data: 输入数据
            labels: 标签数据
            device: 设备
            n_trials: 优化轮数
            **kwargs: 其他参数
        """
        self.data = data
        self.labels = labels
        self.device = device
        self.n_trials = n_trials
        self.kwargs = kwargs
        
        # 预处理数据
        self._preprocess_data()
        
        # 获取AI推荐
        self._get_ai_recommendation()
        
        # 创建目标函数
        self.objective = create_ai_enhanced_objective(
            data, labels, device, **kwargs
        )
        
        # 存储结果
        self.results = []
        self.best_value = -np.inf
        self.best_params = None
        self.best_metrics = None
    
    def _preprocess_data(self):
        """预处理数据"""
        logger.info("预处理数据...")
        
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
        logger.info(f"置信度: {self.recommendation.confidence:.2f}")
    
    def run_optimization(self) -> Dict[str, Any]:
        """
        运行贝叶斯优化
        
        Returns:
            Dict: 优化结果
        """
        logger.info(f"开始AI增强的贝叶斯优化，共{n_trials}轮")
        
        start_time = time.time()
        
        for trial in range(self.n_trials):
            logger.info(f"第 {trial + 1}/{self.n_trials} 轮优化")
            
            # 建议超参数
            hparams = suggest()
            
            # 评估目标函数
            try:
                value, metrics = self.objective(hparams)
                
                # 记录结果
                result = {
                    'trial': trial + 1,
                    'hparams': hparams,
                    'value': value,
                    'metrics': metrics,
                    'timestamp': time.time()
                }
                self.results.append(result)
                
                # 更新最佳结果
                if value > self.best_value:
                    self.best_value = value
                    self.best_params = hparams.copy()
                    self.best_metrics = metrics.copy()
                    logger.info(f"新的最佳结果: {value:.4f}")
                
                # 观察结果（用于BO算法）
                observe(hparams, value)
                
                logger.info(f"第 {trial + 1} 轮完成: 值={value:.4f}, 模型={metrics.get('model_name', 'Unknown')}")
                
            except Exception as e:
                logger.error(f"第 {trial + 1} 轮评估失败: {e}")
                continue
        
        end_time = time.time()
        
        # 汇总结果
        summary = {
            'best_value': self.best_value,
            'best_params': self.best_params,
            'best_metrics': self.best_metrics,
            'total_trials': len(self.results),
            'successful_trials': len([r for r in self.results if 'error' not in r.get('metrics', {})]),
            'total_time': end_time - start_time,
            'data_profile': self.data_profile.to_dict(),
            'ai_recommendation': {
                'model_name': self.recommendation.model_name,
                'model_type': self.recommendation.model_type,
                'architecture': self.recommendation.architecture,
                'reasoning': self.recommendation.reasoning,
                'confidence': self.recommendation.confidence
            },
            'all_results': self.results
        }
        
        logger.info("优化完成!")
        logger.info(f"最佳值: {self.best_value:.4f}")
        logger.info(f"最佳参数: {self.best_params}")
        logger.info(f"总时间: {end_time - start_time:.2f}秒")
        
        return summary
    
    def save_results(self, filename: str):
        """保存结果到文件"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"结果已保存到: {filename}")

def run_ai_enhanced_bo(data, labels=None, device="cpu", n_trials=20, **kwargs):
    """
    运行AI增强的贝叶斯优化
    
    Args:
        data: 输入数据
        labels: 标签数据
        device: 设备
        n_trials: 优化轮数
        **kwargs: 其他参数
    
    Returns:
        Dict: 优化结果
    """
    bo = AIEnhancedBO(data, labels, device, n_trials, **kwargs)
    return bo.run_optimization()

def demo_ai_enhanced_bo():
    """演示AI增强的贝叶斯优化"""
    
    print("=" * 80)
    print("AI增强的贝叶斯优化演示")
    print("=" * 80)
    
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 设置OpenAI API密钥（如果可用）
    import os
    if not os.getenv("OPENAI_API_KEY"):
        print("警告: 未设置OPENAI_API_KEY，将使用默认模型选择")
        print("要使用AI模型选择功能，请设置环境变量: export OPENAI_API_KEY='your-api-key'")
    
    print("\n" + "=" * 60)
    print("演示1: 表格数据优化")
    print("=" * 60)
    
    # 表格数据
    X_tabular = np.random.randn(500, 15).astype("float32")
    y_tabular = np.random.choice(["A", "B", "C"], size=500)
    
    result1 = run_ai_enhanced_bo(X_tabular, y_tabular, device=device, n_trials=5)
    print(f"表格数据优化结果:")
    print(f"  最佳值: {result1['best_value']:.4f}")
    print(f"  最佳参数: {result1['best_params']}")
    print(f"  AI推荐模型: {result1['ai_recommendation']['model_name']}")
    print(f"  推荐理由: {result1['ai_recommendation']['reasoning']}")
    
    print("\n" + "=" * 60)
    print("演示2: 图像数据优化")
    print("=" * 60)
    
    # 图像数据
    X_image = np.random.randn(300, 3, 28, 28).astype("float32")
    y_image = np.random.choice([0, 1], size=300)
    
    result2 = run_ai_enhanced_bo(X_image, y_image, device=device, n_trials=5)
    print(f"图像数据优化结果:")
    print(f"  最佳值: {result2['best_value']:.4f}")
    print(f"  最佳参数: {result2['best_params']}")
    print(f"  AI推荐模型: {result2['ai_recommendation']['model_name']}")
    print(f"  推荐理由: {result2['ai_recommendation']['reasoning']}")
    
    print("\n" + "=" * 60)
    print("演示3: 序列数据优化")
    print("=" * 60)
    
    # 序列数据
    X_sequence = np.random.randn(400, 30, 8).astype("float32")
    y_sequence = np.random.choice([0, 1, 2], size=400)
    
    result3 = run_ai_enhanced_bo(X_sequence, y_sequence, device=device, n_trials=5)
    print(f"序列数据优化结果:")
    print(f"  最佳值: {result3['best_value']:.4f}")
    print(f"  最佳参数: {result3['best_params']}")
    print(f"  AI推荐模型: {result3['ai_recommendation']['model_name']}")
    print(f"  推荐理由: {result3['ai_recommendation']['reasoning']}")

if __name__ == "__main__":
    demo_ai_enhanced_bo()
