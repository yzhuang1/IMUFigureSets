# AI增强机器学习管道架构说明

## 整体架构

```
输入数据 (任意格式)
    ↓
通用数据转换器 (Universal Converter)
    ↓
数据特征分析 (Data Profile)
    ↓
AI模型选择器 (AI Model Selector)
    ↓
动态模型注册系统 (Dynamic Model Registry)
    ↓
模型构建 (Model Builder)
    ↓
训练和评估 (Training & Evaluation)
    ↓
贝叶斯优化 (Bayesian Optimization)
    ↓
输出结果 (Results)
```

## 核心组件

### 1. 通用数据转换器 (`adapters/universal_converter.py`)

**功能**: 将各种数据格式自动转换为PyTorch tensor格式

**特性**:
- 自动检测数据类型（表格、图像、序列等）
- 支持多种输入格式（NumPy、Pandas、列表等）
- 智能数据预处理和标准化
- 生成详细的数据特征档案

**主要类**:
- `DataProfile`: 数据特征描述类
- `UniversalDataset`: 通用数据集类
- `UniversalConverter`: 数据转换器主类

### 2. AI模型选择器 (`models/ai_model_selector.py`)

**功能**: 使用ChatGPT API根据数据特征自动推荐最适合的神经网络

**特性**:
- 基于数据特征的智能推荐
- 支持多种预定义模型类型
- 提供详细的推荐理由和置信度
- 可配置的API调用

**主要类**:
- `ModelRecommendation`: 模型推荐结果类
- `AIModelSelector`: AI模型选择器主类

### 3. 动态模型注册系统 (`models/dynamic_model_registry.py`)

**功能**: 支持动态注册和创建神经网络模型

**特性**:
- 动态模型注册和发现
- 模型工厂模式
- 参数验证和自动构建
- 元数据管理

**主要类**:
- `ModelRegistry`: 模型注册表
- `ModelBuilder`: 模型构建器

### 4. AI增强的目标函数 (`bo/ai_enhanced_objective.py`)

**功能**: 集成AI推荐的贝叶斯优化目标函数

**特性**:
- 自动数据预处理
- AI模型推荐集成
- 智能超参数设置
- 详细的评估指标

**主要类**:
- `AIEnhancedObjective`: AI增强的目标函数类

### 5. 主流程 (`main_new.py`)

**功能**: 集成所有组件的完整机器学习流程

**特性**:
- 一键式数据处理和模型选择
- 自动训练和评估
- 支持多种数据类型
- 详细的日志和结果输出

## 数据流

### 1. 数据输入阶段
```
原始数据 → 数据类型检测 → 数据特征分析 → 数据转换 → PyTorch Dataset
```

### 2. 模型选择阶段
```
数据特征 → AI分析 → 模型推荐 → 模型构建 → 模型实例
```

### 3. 训练优化阶段
```
模型 + 数据 → 超参数优化 → 模型训练 → 性能评估 → 结果输出
```

## 扩展性设计

### 1. 数据转换器扩展
- 在 `UniversalConverter` 中注册新的转换器
- 实现自定义的 `_convert_*` 方法
- 支持新的数据格式和预处理逻辑

### 2. 模型注册扩展
- 使用 `register_model()` 注册新模型
- 实现模型工厂函数
- 添加模型元数据

### 3. AI推荐扩展
- 在 `AIModelSelector` 中添加新的模型类型
- 更新推荐提示词模板
- 扩展模型参数映射

## 向后兼容性

新架构完全向后兼容原有代码：
- 保留所有原有的API接口
- 原有文件继续可用
- 渐进式迁移支持

## 使用示例

### 基本使用
```python
from main_new import train_and_evaluate

# 自动处理数据和选择模型
result = train_and_evaluate(data, labels)
```

### 高级使用
```python
from adapters.universal_converter import convert_to_torch_dataset
from models.ai_model_selector import select_model_for_data
from models.dynamic_model_registry import build_model_from_recommendation

# 1. 转换数据
dataset, collate_fn, profile = convert_to_torch_dataset(data, labels)

# 2. AI选择模型
recommendation = select_model_for_data(profile.to_dict())

# 3. 构建模型
model = build_model_from_recommendation(recommendation, input_shape, num_classes)
```

### 贝叶斯优化
```python
from bo.run_ai_enhanced_bo import run_ai_enhanced_bo

# 运行AI增强的BO
result = run_ai_enhanced_bo(data, labels, n_trials=20)
```

## 配置选项

### 环境变量
- `OPENAI_API_KEY`: OpenAI API密钥（用于AI模型选择）

### 参数配置
- 数据预处理参数（标准化、归一化等）
- 模型超参数（学习率、隐藏层大小等）
- BO优化参数（试验次数、搜索空间等）

## 性能考虑

### 1. 数据转换优化
- 延迟加载和内存优化
- 批量处理支持
- 并行数据预处理

### 2. 模型选择优化
- API调用缓存
- 本地模型推荐备选
- 异步处理支持

### 3. 训练优化
- GPU加速支持
- 混合精度训练
- 分布式训练准备

## 错误处理

### 1. 数据转换错误
- 自动降级到默认转换器
- 详细的错误日志
- 数据验证和修复

### 2. AI推荐错误
- 本地默认推荐备选
- API调用重试机制
- 推荐结果验证

### 3. 模型构建错误
- 参数验证和修复
- 模型兼容性检查
- 自动参数调整

## 监控和日志

### 1. 日志系统
- 分级日志记录
- 结构化日志输出
- 性能指标跟踪

### 2. 监控指标
- 数据转换性能
- AI推荐准确性
- 模型训练效果
- BO优化进度

## 未来扩展

### 1. 计划功能
- 更多数据格式支持
- 更丰富的模型类型
- 自动化超参数调优
- 模型解释性分析

### 2. 技术改进
- 更智能的AI推荐
- 更高效的BO算法
- 更好的错误恢复
- 更强的扩展性
