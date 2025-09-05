# AI-Enhanced ML Pipeline

一个灵活的AI增强机器学习管道，支持多种数据类型和自动模型选择。

## 新架构特性

### 🚀 核心功能
- **通用数据转换器**: 自动将各种数据格式转换为PyTorch tensor
- **AI模型选择器**: 使用ChatGPT API根据数据特征自动推荐最适合的神经网络
- **动态模型注册系统**: 支持动态添加新的神经网络架构
- **智能贝叶斯优化**: 集成AI推荐的BO流程
- **自动数据特征分析**: 智能分析数据特征并生成详细档案

### 📊 支持的数据类型
- **表格数据**: NumPy数组、Pandas DataFrame
- **图像数据**: 2D/3D/4D数组，支持不同通道格式
- **序列数据**: 规则和不规则时间序列
- **自定义数据**: 通过注册器支持任意数据格式

### 🤖 AI模型推荐
- 基于数据特征的智能模型选择
- 支持多种预定义模型类型
- 提供详细的推荐理由和置信度
- 可扩展的模型注册系统

## 快速开始

### 1. 安装依赖
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 设置OpenAI API密钥（可选）
```bash
export OPENAI_API_KEY='your-api-key'
```

### 3. 基本使用
```python
from main_new import train_and_evaluate
import numpy as np

# 准备数据
X = np.random.randn(1000, 20).astype("float32")
y = np.random.choice(["A", "B", "C"], size=1000)

# AI自动处理数据和选择模型
result = train_and_evaluate(X, y, epochs=5)
print(f"推荐模型: {result['recommendation'].model_name}")
print(f"训练结果: {result['metrics']}")
```

### 4. 运行演示
```python
python example_usage.py
```

## 详细使用指南

### 数据转换
```python
from adapters.universal_converter import convert_to_torch_dataset

# 自动检测数据类型并转换
dataset, collate_fn, profile = convert_to_torch_dataset(data, labels)
print(f"数据特征: {profile}")
```

### AI模型选择
```python
from models.ai_model_selector import select_model_for_data

# 根据数据特征选择模型
recommendation = select_model_for_data(profile.to_dict())
print(f"推荐模型: {recommendation.model_name}")
print(f"推荐理由: {recommendation.reasoning}")
```

### 贝叶斯优化
```python
from bo.run_ai_enhanced_bo import run_ai_enhanced_bo

# 运行AI增强的BO
result = run_ai_enhanced_bo(data, labels, n_trials=20)
print(f"最佳参数: {result['best_params']}")
```

### 自定义模型注册
```python
from models.dynamic_model_registry import register_model

# 注册自定义模型
register_model("MyModel", MyModelClass, {
    "type": "custom",
    "description": "我的自定义模型"
})
```

## 项目结构

```
ml_pipeline/
├── adapters/
│   ├── universal_converter.py    # 通用数据转换器
│   └── unified_adapter.py        # 原有适配器（向后兼容）
├── models/
│   ├── ai_model_selector.py      # AI模型选择器
│   ├── dynamic_model_registry.py # 动态模型注册系统
│   ├── model_picker.py           # 原有模型选择器（向后兼容）
│   └── ...                       # 各种模型实现
├── bo/
│   ├── ai_enhanced_objective.py  # AI增强的目标函数
│   ├── run_ai_enhanced_bo.py     # AI增强的BO运行器
│   └── ...                       # 原有BO代码
├── main_new.py                   # 新的主流程
├── example_usage.py              # 使用示例
└── requirements.txt              # 依赖列表
```

## 依赖要求

- torch
- numpy
- scikit-learn
- tqdm
- requests (for OpenAI API)
- pandas (optional)
- opencv-python (optional, for image processing)
- pillow (optional, for image processing)

## 向后兼容性

新架构完全向后兼容，原有的代码仍然可以正常工作：
- `main.py` - 原有主流程
- `adapters/unified_adapter.py` - 原有数据转换器
- `models/model_picker.py` - 原有模型选择器
- `bo/objective.py` - 原有目标函数

## 贡献指南

1. 添加新的数据转换器：在 `adapters/universal_converter.py` 中注册
2. 添加新的模型：在 `models/dynamic_model_registry.py` 中注册
3. 扩展AI推荐：在 `models/ai_model_selector.py` 中添加新的模型类型

## 许可证

MIT License

