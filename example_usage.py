"""
使用示例：展示如何使用新的AI增强机器学习管道
"""

import numpy as np
import torch
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def example_1_simple_usage():
    """示例1: 简单使用"""
    print("=" * 60)
    print("示例1: 简单使用")
    print("=" * 60)
    
    from main_new import process_data_with_ai, train_and_evaluate
    
    # 准备数据
    X = np.random.randn(1000, 20).astype("float32")
    y = np.random.choice(["A", "B", "C"], size=1000)
    
    # 使用AI自动处理数据和选择模型
    result = process_data_with_ai(X, y)
    
    print(f"数据特征: {result['data_profile']}")
    print(f"AI推荐模型: {result['recommendation'].model_name}")
    print(f"推荐理由: {result['recommendation'].reasoning}")
    
    # 训练和评估
    training_result = train_and_evaluate(X, y, epochs=3)
    print(f"训练结果: {training_result['metrics']}")

def example_2_different_data_types():
    """示例2: 不同数据类型"""
    print("\n" + "=" * 60)
    print("示例2: 不同数据类型")
    print("=" * 60)
    
    from main_new import train_and_evaluate
    
    # 表格数据
    print("处理表格数据...")
    X_tabular = np.random.randn(500, 15).astype("float32")
    y_tabular = np.random.choice([0, 1, 2], size=500)
    result1 = train_and_evaluate(X_tabular, y_tabular, epochs=2)
    print(f"表格数据 - 推荐模型: {result1['recommendation'].model_name}")
    
    # 图像数据
    print("处理图像数据...")
    X_image = np.random.randn(200, 3, 32, 32).astype("float32")
    y_image = np.random.choice([0, 1], size=200)
    result2 = train_and_evaluate(X_image, y_image, epochs=2)
    print(f"图像数据 - 推荐模型: {result2['recommendation'].model_name}")
    
    # 序列数据
    print("处理序列数据...")
    X_sequence = np.random.randn(300, 50, 10).astype("float32")
    y_sequence = np.random.choice([0, 1, 2], size=300)
    result3 = train_and_evaluate(X_sequence, y_sequence, epochs=2)
    print(f"序列数据 - 推荐模型: {result3['recommendation'].model_name}")

def example_3_bo_optimization():
    """示例3: 贝叶斯优化"""
    print("\n" + "=" * 60)
    print("示例3: 贝叶斯优化")
    print("=" * 60)
    
    from bo.run_ai_enhanced_bo import run_ai_enhanced_bo
    
    # 准备数据
    X = np.random.randn(800, 12).astype("float32")
    y = np.random.choice(["X", "Y", "Z"], size=800)
    
    # 运行BO优化
    result = run_ai_enhanced_bo(X, y, n_trials=3)
    
    print(f"BO优化结果:")
    print(f"  最佳值: {result['best_value']:.4f}")
    print(f"  最佳参数: {result['best_params']}")
    print(f"  AI推荐模型: {result['ai_recommendation']['model_name']}")

def example_4_custom_data_conversion():
    """示例4: 自定义数据转换"""
    print("\n" + "=" * 60)
    print("示例4: 自定义数据转换")
    print("=" * 60)
    
    from adapters.universal_converter import convert_to_torch_dataset
    
    # 不规则序列数据
    X_irregular = [
        np.random.randn(np.random.randint(10, 50), 5) for _ in range(100)
    ]
    y_irregular = np.random.choice([0, 1], size=100)
    
    # 转换数据
    dataset, collate_fn, profile = convert_to_torch_dataset(X_irregular, y_irregular)
    
    print(f"数据特征: {profile}")
    print(f"数据集大小: {len(dataset)}")
    print(f"需要collate函数: {collate_fn is not None}")

def example_5_model_registry():
    """示例5: 模型注册系统"""
    print("\n" + "=" * 60)
    print("示例5: 模型注册系统")
    print("=" * 60)
    
    from models.dynamic_model_registry import model_registry, build_model
    
    # 查看已注册的模型
    models = model_registry.list_models()
    print("已注册的模型:")
    for model_info in models:
        print(f"  - {model_info['name']}: {model_info['description']}")
    
    # 构建模型
    model = build_model("TabMLP", input_shape=(20,), num_classes=3, hidden=128)
    print(f"\n构建的模型: {model}")

def example_6_ai_model_selection():
    """示例6: AI模型选择"""
    print("\n" + "=" * 60)
    print("示例6: AI模型选择")
    print("=" * 60)
    
    from models.ai_model_selector import select_model_for_data
    from adapters.universal_converter import analyze_data_profile
    
    # 分析数据特征
    X = np.random.randn(500, 25).astype("float32")
    y = np.random.choice([0, 1, 2, 3], size=500)
    
    profile = analyze_data_profile(X, y)
    print(f"数据特征: {profile}")
    
    # AI选择模型
    recommendation = select_model_for_data(profile.to_dict())
    print(f"AI推荐: {recommendation.model_name}")
    print(f"推荐理由: {recommendation.reasoning}")
    print(f"置信度: {recommendation.confidence:.2f}")

if __name__ == "__main__":
    print("AI增强机器学习管道使用示例")
    print("=" * 80)
    
    # 设置OpenAI API密钥（如果可用）
    import os
    if not os.getenv("OPENAI_API_KEY"):
        print("注意: 未设置OPENAI_API_KEY，将使用默认模型选择")
        print("要使用AI模型选择功能，请设置环境变量: export OPENAI_API_KEY='your-api-key'")
    
    try:
        example_1_simple_usage()
        example_2_different_data_types()
        example_3_bo_optimization()
        example_4_custom_data_conversion()
        example_5_model_registry()
        example_6_ai_model_selection()
        
        print("\n" + "=" * 80)
        print("所有示例运行完成!")
        
    except Exception as e:
        print(f"运行示例时出错: {e}")
        import traceback
        traceback.print_exc()
