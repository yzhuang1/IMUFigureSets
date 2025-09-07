# API Call Limits Configuration

To prevent infinite loops and control OpenAI API costs, the system has configurable limits for AI operations.

## 🎛️ **Configurable Limits**

### **Environment Variables**

Set these in your `.env` file or environment:

```bash
# Maximum number of different model architectures to try
MAX_MODEL_ATTEMPTS=4

# Maximum number of Bayesian Optimization trials
MAX_BO_TRIALS=10

# Maximum retries for evaluation failures (future use)
MAX_EVAL_RETRIES=2
```

### **Default Values**

If not configured, the system uses these defaults:

- `MAX_MODEL_ATTEMPTS=3` - Try up to 3 different model architectures
- `MAX_BO_TRIALS=10` - Run up to 10 BO optimization trials
- `MAX_EVAL_RETRIES=2` - Retry evaluation up to 2 times on failures

## 📊 **API Call Breakdown**

### **Per Model Selection Iteration:**
- **1 API call** for model selection (GPT-4)
- **1 API call** for performance evaluation (GPT-4)
- **Total per iteration: 2 API calls**

### **Full Pipeline Costs:**

**Iterative Model Selection:**
- Worst case: `MAX_MODEL_ATTEMPTS × 2 = 6 API calls`
- Typical case: `1-2 iterations × 2 = 2-4 API calls`

**Bayesian Optimization:**
- Model selection: `2 API calls` (one-time)
- BO iterations: `MAX_BO_TRIALS = 10 API calls` (evaluation only)
- **Total: ~12 API calls**

## 💰 **Cost Estimates (GPT-4)**

Assuming ~500 tokens per API call:

**Model Selection Pipeline:**
- Best case: 2 calls × $0.015 = **$0.03**
- Worst case: 6 calls × $0.015 = **$0.09**

**Bayesian Optimization:**
- Full run: 12 calls × $0.015 = **$0.18**

**Total per experiment: $0.03 - $0.27**

## 🎯 **Recommended Settings**

### **For Development/Testing:**
```bash
MAX_MODEL_ATTEMPTS=2    # Quick iterations
MAX_BO_TRIALS=5         # Fast feedback
```

### **For Production/Research:**
```bash
MAX_MODEL_ATTEMPTS=3    # Thorough search
MAX_BO_TRIALS=20        # Better optimization
```

### **For Cost-Conscious Usage:**
```bash
MAX_MODEL_ATTEMPTS=1    # Single model attempt
MAX_BO_TRIALS=5         # Minimal BO
```

## ⚙️ **Usage Examples**

### **Using Config Defaults:**
```python
from main_ai_enhanced import process_data_with_ai_enhanced_evaluation

# Uses MAX_MODEL_ATTEMPTS from config (default: 3)
result = process_data_with_ai_enhanced_evaluation(data, labels)
```

### **Overriding Limits:**
```python
# Override for specific use case
result = process_data_with_ai_enhanced_evaluation(
    data, labels, 
    max_model_attempts=5  # Override default
)
```

### **Bayesian Optimization:**
```python
from bo.run_ai_enhanced_bo import run_ai_enhanced_bo

# Uses MAX_BO_TRIALS from config (default: 10)
result = run_ai_enhanced_bo(data, labels)

# Or override
result = run_ai_enhanced_bo(data, labels, n_trials=15)
```

## 🚫 **Preventing Runaway Costs**

The system will **never exceed** the configured limits:

1. **Hard Stops**: All loops have maximum iteration counts
2. **Clear Logging**: Shows exactly how many API calls are made
3. **Fail Fast**: Errors stop execution immediately
4. **No Retries**: Failed API calls don't retry indefinitely

## 📈 **Monitoring API Usage**

The system logs all API activities:

```
INFO - IterativeModelSelector initialized with max_iterations=3
INFO - AIEnhancedBO initialized with n_trials=10
INFO - Model selection iteration 1/3
INFO - AI recommended model: TabMLP (confidence: 0.95)
INFO - AI evaluation completed: accept (confidence: 0.88)
```

Monitor these logs to track your API usage in real-time.