"""
新的主流程文件
集成通用数据转换器和AI模型选择器
"""

import logging
import numpy as np
import torch
from torch.utils.data import DataLoader

from adapters.universal_converter import convert_to_torch_dataset, DataProfile
from models.ai_model_selector import select_model_for_data
from models.dynamic_model_registry import build_model_from_recommendation
from train import train_one_model
from evaluation.evaluate import evaluate_model

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_data_with_ai(data, labels=None, **kwargs):
    """
    使用AI自动处理数据和选择模型
    
    Args:
        data: 输入数据（任意格式）
        labels: 标签数据
        **kwargs: 其他参数
    
    Returns:
        dict: 包含数据集、模型、推荐信息等的字典
    """
    logger.info("开始处理数据...")
    
    # 1. 转换数据为PyTorch格式
    logger.info("步骤1: 转换数据为PyTorch格式")
    dataset, collate_fn, data_profile = convert_to_torch_dataset(
        data, labels, **kwargs
    )
    
    logger.info(f"数据转换完成: {data_profile}")
    
    # 2. 使用AI选择最适合的模型
    logger.info("步骤2: 使用AI选择最适合的模型")
    recommendation = select_model_for_data(data_profile.to_dict())
    
    logger.info(f"AI推荐模型: {recommendation.model_name}")
    logger.info(f"推荐理由: {recommendation.reasoning}")
    logger.info(f"置信度: {recommendation.confidence:.2f}")
    
    # 3. 根据推荐构建模型
    logger.info("步骤3: 构建推荐模型")
    
    # 确定输入形状
    if data_profile.is_sequence:
        input_shape = (data_profile.feature_count,)
    elif data_profile.is_image:
        if data_profile.channels and data_profile.height and data_profile.width:
            input_shape = (data_profile.channels, data_profile.height, data_profile.width)
        else:
            input_shape = (3, 32, 32)  # 默认图像尺寸
    elif data_profile.is_tabular:
        input_shape = (data_profile.feature_count,)
    else:
        input_shape = (data_profile.feature_count,)
    
    num_classes = data_profile.label_count if data_profile.has_labels else 2
    
    model = build_model_from_recommendation(
        recommendation, input_shape, num_classes
    )
    
    # 4. 准备数据加载器
    logger.info("步骤4: 准备数据加载器")
    loader = DataLoader(
        dataset, 
        batch_size=kwargs.get('batch_size', 64), 
        shuffle=True, 
        collate_fn=collate_fn
    )
    
    return {
        'dataset': dataset,
        'data_loader': loader,
        'model': model,
        'data_profile': data_profile,
        'recommendation': recommendation,
        'collate_fn': collate_fn
    }

def train_and_evaluate(data, labels=None, device="cpu", epochs=5, **kwargs):
    """
    训练和评估模型
    
    Args:
        data: 输入数据
        labels: 标签数据
        device: 设备
        epochs: 训练轮数
        **kwargs: 其他参数
    
    Returns:
        dict: 训练结果
    """
    # 处理数据
    result = process_data_with_ai(data, labels, **kwargs)
    
    model = result['model']
    loader = result['data_loader']
    
    # 移动到设备
    device = torch.device(device)
    model.to(device)
    
    # 训练模型
    logger.info(f"开始训练模型: {result['recommendation'].model_name}")
    trained_model = train_one_model(
        model, loader, device=device, epochs=epochs
    )
    
    # 评估模型
    logger.info("开始评估模型")
    metrics = evaluate_model(trained_model, loader, device=device)
    
    return {
        'model': trained_model,
        'metrics': metrics,
        'data_profile': result['data_profile'],
        'recommendation': result['recommendation']
    }

def demo_with_different_data_types():
    """演示不同数据类型的处理"""
    
    print("=" * 60)
    print("演示1: 表格数据")
    print("=" * 60)
    
    # 表格数据
    X_tabular = np.random.randn(1000, 20).astype("float32")
    y_tabular = np.random.choice(["A", "B", "C", "D"], size=1000)
    
    result1 = train_and_evaluate(X_tabular, y_tabular, epochs=3)
    print(f"表格数据结果: {result1['metrics']}")
    print(f"推荐模型: {result1['recommendation'].model_name}")
    print(f"推荐理由: {result1['recommendation'].reasoning}")
    
    print("\n" + "=" * 60)
    print("演示2: 图像数据")
    print("=" * 60)
    
    # 图像数据
    X_image = np.random.randn(500, 3, 32, 32).astype("float32")
    y_image = np.random.choice([0, 1], size=500)
    
    result2 = train_and_evaluate(X_image, y_image, epochs=3)
    print(f"图像数据结果: {result2['metrics']}")
    print(f"推荐模型: {result2['recommendation'].model_name}")
    print(f"推荐理由: {result2['recommendation'].reasoning}")
    
    print("\n" + "=" * 60)
    print("演示3: 序列数据")
    print("=" * 60)
    
    # 序列数据
    X_sequence = np.random.randn(300, 50, 10).astype("float32")  # (N, T, C)
    y_sequence = np.random.choice([0, 1, 2], size=300)
    
    result3 = train_and_evaluate(X_sequence, y_sequence, epochs=3)
    print(f"序列数据结果: {result3['metrics']}")
    print(f"推荐模型: {result3['recommendation'].model_name}")
    print(f"推荐理由: {result3['recommendation'].reasoning}")
    
    print("\n" + "=" * 60)
    print("演示4: 不规则序列数据")
    print("=" * 60)
    
    # 不规则序列数据
    X_irregular = [
        np.random.randn(np.random.randint(10, 50), 5) for _ in range(200)
    ]
    y_irregular = np.random.choice([0, 1], size=200)
    
    result4 = train_and_evaluate(X_irregular, y_irregular, epochs=3)
    print(f"不规则序列数据结果: {result4['metrics']}")
    print(f"推荐模型: {result4['recommendation'].model_name}")
    print(f"推荐理由: {result4['recommendation'].reasoning}")

if __name__ == "__main__":
    # 设置OpenAI API密钥（如果可用）
    import os
    if not os.getenv("OPENAI_API_KEY"):
        print("警告: 未设置OPENAI_API_KEY，将使用默认模型选择")
        print("要使用AI模型选择功能，请设置环境变量: export OPENAI_API_KEY='your-api-key'")
    
    # 运行演示
    demo_with_different_data_types()
