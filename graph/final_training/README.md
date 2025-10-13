# Final Training Resumer

Standalone pipeline to resume final training for runs that completed BO but failed during final training.

## Features

- Automatically detects incomplete runs from logs
- Extracts best BO parameters from log files
- Finds corresponding training functions
- Executes final training with memory cleanup
- Saves trained models and results

## Usage

### List incomplete runs
```bash
python graph/final_training/resume_final_training.py --list
```

### Resume specific run
```bash
python graph/final_training/resume_final_training.py --log 2025-10-13_10-57-10.log
```

### Resume all incomplete runs
```bash
python graph/final_training/resume_final_training.py --all
```

## How it works

1. **Scans log files** in `logs/` directory
2. **Identifies incomplete runs**:
   - Has "STEP 4: Final Training Execution"
   - Missing "STEP 5: Performance Analysis"
3. **Extracts information**:
   - Best BO parameters
   - Model name
   - Training function path
4. **Executes final training**:
   - Loads training function from `generated_training_functions/`
   - Uses centralized data splits
   - Applies best BO parameters
   - Saves to `trained_models/{timestamp}_{model_name}_resumed/`

## Output

Final training results are saved to:
```
trained_models/{timestamp}_{model_name}_resumed/
├── model.pt                 # Trained model weights
├── X_test.pt               # Test data
├── y_test.pt               # Test labels
└── training_results.json   # Full results and metrics
```

## Memory Management

The resumer includes aggressive memory cleanup:
- Python garbage collection before training
- GPU cache clearing with synchronization
- Prevents OOM issues after long BO runs

## Example

```bash
# Check what needs to be resumed
python graph/final_training/resume_final_training.py --list

# Output:
# Found 4 incomplete run(s):
#
# 1. Log: 2025-10-13_10-57-10.log
#    Model: MixSleepTinyGCN (MixSleepNet-inspired)
#    Timestamp: 2025-10-13_10-57-10
#
# 2. Log: 2025-10-13_03-43-26.log
#    Model: MR-CNN + BiGRU (BiMamba-lite) for ISRUC
#    Timestamp: 2025-10-13_03-43-26

# Resume the first one
python graph/final_training/resume_final_training.py --log 2025-10-13_10-57-10.log
```

## Troubleshooting

**Log file not found or cannot be parsed:**
- Ensure the log file contains "Best params:" line
- Check that the training function exists in `generated_training_functions/`

**OOM during final training:**
- The resumer cleans GPU memory before starting
- If still fails, check your GPU memory availability
- Consider reducing batch_size in the best params manually

**Model name mismatch:**
- The resumer searches for training functions by model name
- Ensure the model name in the log matches the JSON file
