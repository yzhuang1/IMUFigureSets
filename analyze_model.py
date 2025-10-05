"""
Model Analysis Script - Simple version for quantized models
Manually calculates accuracy and inference latency without complex dependencies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import time
import argparse
from pathlib import Path
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import inspect
import ast


def load_quantized_model(model_path, training_json_path):
    """Load quantized model by reconstructing architecture from training JSON."""
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint['model_state_dict']
    hyperparams = checkpoint['best_hyperparameters']

    # Calculate actual checkpoint size (what was saved)
    checkpoint_size_bytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor in state_dict.values()
        if isinstance(tensor, torch.Tensor)
    )

    print(f"Model: {checkpoint['model_name']}")

    # Load training code and data profile
    with open(training_json_path, 'r') as f:
        training_data = json.load(f)

    training_code = training_data['training_code']
    data_profile = training_data.get('data_profile', {})

    # Extract data characteristics from profile
    input_shape = data_profile.get('shape', [None, 1000, 2])  # [N, seq_len, channels]
    num_classes = data_profile.get('label_count', 5)

    # Determine in_ch and seq_len from shape
    if len(input_shape) >= 3:
        seq_len = input_shape[1]
        in_ch = input_shape[2]
    else:
        # Fallback to defaults
        seq_len = 1000
        in_ch = 2

    print(f"Data profile: seq_len={seq_len}, in_ch={in_ch}, num_classes={num_classes}")

    # Extract model class name - find the one that's actually instantiated in the training code
    class_pattern = r'class\s+(\w+)\s*\([^)]*nn\.Module[^)]*\):'
    class_matches = re.findall(class_pattern, training_code)
    model_class_candidates = [c for c in class_matches if 'Dataset' not in c and 'Loss' not in c]

    if not model_class_candidates:
        raise ValueError("No model class found")

    # Find which class is actually instantiated by looking for pattern: model = ClassName(...)
    model_class_name = None
    for candidate in model_class_candidates:
        # Look specifically for: model = ClassName(...) - this is the main model
        # The pattern must have 'model =' before it
        instantiation_pattern = rf'\bmodel\s*=\s*{candidate}\s*\('
        if re.search(instantiation_pattern, training_code):
            model_class_name = candidate
            break

    # Fallback: use the last candidate (usually the main model class comes after helper classes)
    if not model_class_name:
        model_class_name = model_class_candidates[-1]

    print(f"Found model class: {model_class_name} (from candidates: {model_class_candidates})")

    # Extract ALL class definitions and helper functions
    # The main model class might depend on helper classes (SEBlock, FocalLoss) and functions (conv1d_out_len)
    all_code_blocks = []

    # Extract all classes
    for class_name in model_class_candidates:
        class_start = training_code.find(f'class {class_name}')
        if class_start == -1:
            continue

        lines = training_code[class_start:].split('\n')
        class_code_lines = []
        in_class = False
        base_indent = None

        for line in lines:
            if line.strip().startswith('class ') and in_class:
                break
            if line.strip().startswith(f'class {class_name}'):
                in_class = True
                class_code_lines.append(line)
                continue
            if in_class:
                if line.strip() and not line[0].isspace():
                    break
                if line.strip():
                    if base_indent is None:
                        base_indent = len(line) - len(line.lstrip())
                    current_indent = len(line) - len(line.lstrip())
                    if current_indent < base_indent and line.strip():
                        break
                class_code_lines.append(line)

        if class_code_lines:
            all_code_blocks.append('\n'.join(class_code_lines))

    # Extract specific known helper functions that models might use
    # Look for common helper functions like conv1d_out_len, conv2d_out_len, etc.
    helper_func_names = ['conv1d_out_len', 'conv2d_out_len', 'calculate_output_size']
    lines = training_code.split('\n')

    for func_name in helper_func_names:
        for i, line in enumerate(lines):
            if f'def {func_name}' in line:
                # Found the function, extract it
                func_lines = []
                base_indent = len(line) - len(line.lstrip())

                # Add the function definition line
                func_lines.append(line)

                # Collect the function body
                j = i + 1
                while j < len(lines):
                    next_line = lines[j]
                    if not next_line.strip():
                        # Empty line - include it
                        func_lines.append(next_line)
                        j += 1
                        continue

                    current_indent = len(next_line) - len(next_line.lstrip())
                    # Stop if we're back at or before the function's indent level (except for blank lines)
                    if current_indent <= base_indent:
                        break

                    func_lines.append(next_line)
                    j += 1

                # Dedent the function to make it module-level
                if func_lines:
                    min_indent = min((len(l) - len(l.lstrip()) for l in func_lines if l.strip()), default=0)
                    dedented = [l[min_indent:] if len(l) > min_indent else l for l in func_lines]
                    all_code_blocks.append('\n'.join(dedented))
                break  # Found this function, move to next

    # Execute all code blocks together
    exec_globals = {'torch': torch, 'nn': nn, 'F': F, 'math': __import__('math')}
    combined_code = '\n\n'.join(all_code_blocks)
    exec(combined_code, exec_globals)
    ModelClass = exec_globals[model_class_name]

    # Extract model instantiation from training code using AST parsing
    print("Extracting model instantiation pattern from training code...")

    # Build execution context with values from training code
    # For tabular data, num_features is the feature dimension
    num_features = data_profile.get('feature_count', input_shape[-1] if input_shape else 561)

    code_context = {
        'seq_len': seq_len,
        'num_classes': num_classes,
        'in_ch': in_ch,
        'num_features': num_features,
        'int': int,
        'float': float,
        'bool': bool,
        'max': max,
        'hyperparams': hyperparams,
    }

    # Pre-populate context with all hyperparameters directly
    # This ensures that variables like head_dim, n_heads are available
    # even if they're used directly (not via hyperparams.get())
    for key, value in hyperparams.items():
        if key not in code_context:
            code_context[key] = value

    print(f"Context initialized with hyperparams: {list(hyperparams.keys())}")

    # Parse variable assignments from training code to build context
    # We need to do this in multiple passes to handle dependencies (e.g., d_model = head_dim * n_heads)
    lines = training_code.split('\n')

    # First pass: collect ALL variable assignments (not just hyperparams)
    # This will catch things like: d_model = int(head_dim) * int(n_heads)
    for line in lines:
        line_stripped = line.strip()
        # Skip comments, imports, class/function definitions
        if line_stripped.startswith('#') or line_stripped.startswith('import') or line_stripped.startswith('def ') or line_stripped.startswith('class '):
            continue

        if '=' in line_stripped and not line_stripped.startswith('if ') and not line_stripped.startswith('for ') and not line_stripped.startswith('while '):
            # Extract variable assignment
            var_match = re.match(r'(\w+)\s*=\s*(.+)', line_stripped)
            if var_match:
                var_name = var_match.group(1)
                var_expr = var_match.group(2).strip()

                # Skip if it's part of a data structure or multiline
                if var_expr.endswith(',') or var_expr.endswith('\\'):
                    continue

                # Skip if already in context
                if var_name in code_context:
                    continue

                try:
                    # Safely evaluate the expression in our context
                    value = eval(var_expr, {}, code_context)
                    code_context[var_name] = value
                    # Print relevant model parameters
                    if var_name in ['patch_size', 'n_heads', 'head_dim', 'num_layers', 'mlp_ratio', 'dropout', 'attn_dropout', 'd_model', 'nhead', 'd_model_factor', 'ff_factor', 'stem_channels']:
                        print(f"  Found: {var_name} = {var_expr} → {value}")
                except Exception as e:
                    # If evaluation fails, it might be because dependencies aren't resolved yet
                    # We'll try again in the next pass
                    pass

    # Second pass: try again for any that failed (dependency resolution)
    for line in lines:
        line_stripped = line.strip()
        if line_stripped.startswith('#') or line_stripped.startswith('import') or line_stripped.startswith('def ') or line_stripped.startswith('class '):
            continue

        if '=' in line_stripped and not line_stripped.startswith('if ') and not line_stripped.startswith('for ') and not line_stripped.startswith('while '):
            var_match = re.match(r'(\w+)\s*=\s*(.+)', line_stripped)
            if var_match:
                var_name = var_match.group(1)
                var_expr = var_match.group(2).strip()

                # Skip if already in context or irrelevant
                if var_name in code_context or var_expr.endswith(',') or var_expr.endswith('\\'):
                    continue

                # Only process model-relevant variables
                if var_name in ['d_model', 'stem_channels', 'nhead', 'n_heads', 'head_dim', 'd_model_factor', 'ff_factor', 'mlp_ratio', 'patch_size', 'dropout', 'attn_dropout', 'num_layers']:
                    try:
                        value = eval(var_expr, {}, code_context)
                        code_context[var_name] = value
                        print(f"  Derived (2nd pass): {var_name} = {var_expr} → {value}")
                    except:
                        pass

    # Find model instantiation using AST to handle nested parentheses properly
    model_call_args = {}
    model_call_positional = []
    try:
        # Parse the entire training code
        tree = ast.parse(training_code)

        # Walk through AST to find model class instantiation
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                # Check if this is a call to our model class
                if isinstance(node.func, ast.Name) and node.func.id == model_class_name:
                    # Extract positional arguments
                    for arg in node.args:
                        param_expr = ast.unparse(arg)
                        model_call_positional.append(param_expr)
                    # Extract keyword arguments
                    for keyword in node.keywords:
                        param_name = keyword.arg
                        # Convert AST node back to source code
                        param_expr = ast.unparse(keyword.value)
                        model_call_args[param_name] = param_expr
                    print(f"Found model instantiation via AST: positional={model_call_positional}, keyword={model_call_args}")
                    break
    except Exception as e:
        print(f"AST parsing failed: {e}, trying regex fallback...")
        # Fallback to regex with better handling of nested parentheses
        # Find lines containing model instantiation
        pattern = rf'model\s*=\s*{model_class_name}\s*\('
        for i, line in enumerate(training_code.split('\n')):
            if re.search(pattern, line):
                # Extract the complete call by balancing parentheses
                call_start = i
                paren_count = 0
                call_lines = []
                for j in range(i, min(i + 20, len(training_code.split('\n')))):
                    line = training_code.split('\n')[j]
                    call_lines.append(line)
                    paren_count += line.count('(') - line.count(')')
                    if paren_count == 0 and '(' in ''.join(call_lines):
                        break

                call_text = ' '.join(call_lines)
                # Extract arguments with simple parsing
                args_start = call_text.find('(')
                args_end = call_text.rfind(')')
                if args_start != -1 and args_end != -1:
                    args_text = call_text[args_start+1:args_end]
                    # Split by comma, but respect nested parentheses
                    args = []
                    current_arg = ''
                    paren_depth = 0
                    for char in args_text:
                        if char == '(':
                            paren_depth += 1
                        elif char == ')':
                            paren_depth -= 1
                        elif char == ',' and paren_depth == 0:
                            args.append(current_arg.strip())
                            current_arg = ''
                            continue
                        current_arg += char
                    if current_arg.strip():
                        args.append(current_arg.strip())

                    # Parse each argument
                    for arg in args:
                        if '=' in arg:
                            parts = arg.split('=', 1)
                            param_name = parts[0].strip()
                            param_expr = parts[1].strip()
                            model_call_args[param_name] = param_expr

                print(f"Found model instantiation via regex: {model_call_args}")
                break

    # Get the __init__ signature
    sig = inspect.signature(ModelClass.__init__)
    model_params = {}
    positional_args = []

    print(f"Model __init__ parameters: {list(sig.parameters.keys())}")

    # If we have positional arguments, map them to parameter names
    if model_call_positional:
        param_names = [p for p in sig.parameters.keys() if p != 'self']
        for i, expr in enumerate(model_call_positional):
            if i < len(param_names):
                param_name = param_names[i]
                try:
                    value = eval(expr, {}, code_context)
                    positional_args.append(value)
                    print(f"  {param_name} (positional) = {expr} → {value}")
                except Exception as e:
                    print(f"  Warning: Could not evaluate positional arg {i} ({expr}): {e}")

    # Build model parameters by evaluating expressions from training code
    # Only process keyword arguments if we don't have positional args
    if not positional_args:
        for param_name in sig.parameters:
            if param_name == 'self':
                continue

            # If we found how this param is set in training code, use that
            if param_name in model_call_args:
                expr = model_call_args[param_name]
                try:
                    # Evaluate the expression with our context
                    value = eval(expr, {}, code_context)
                    model_params[param_name] = value
                    print(f"  {param_name} = {expr} → {value}")
                except Exception as e:
                    print(f"  Warning: Could not evaluate {param_name}={expr}: {e}")
                    # Fallback: try to get from code_context or hyperparams
                    if param_name in code_context:
                        model_params[param_name] = code_context[param_name]
                    elif param_name in hyperparams:
                        model_params[param_name] = hyperparams[param_name]
            # Check if it's in our code context (from parsed variable assignments)
            elif param_name in code_context:
                model_params[param_name] = code_context[param_name]
                print(f"  {param_name} = {code_context[param_name]} (from context)")
            # Try hyperparams directly
            elif param_name in hyperparams:
                model_params[param_name] = hyperparams[param_name]
                print(f"  {param_name} = {hyperparams[param_name]} (from hyperparams)")
            else:
                print(f"  Warning: Could not find value for {param_name}, will use default if available")

    # Instantiate model with extracted parameters
    if positional_args:
        print(f"Final model instantiation with positional args: {positional_args}")
        model = ModelClass(*positional_args)
    else:
        print(f"Final model instantiation parameters: {model_params}")
        model = ModelClass(**model_params)

    # Check if quantized
    is_quantized = any('_packed_params' in key for key in state_dict.keys())

    if is_quantized:
        print("Detected quantized model - applying quantization to model first, then loading state dict...")
        try:
            from torch.ao.quantization import quantize_dynamic
        except ImportError:
            from torch.quantization import quantize_dynamic

        # First, apply quantization to create the quantized model structure
        print("Creating quantized model structure...")
        model = quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)

        # Now load the quantized state dict directly
        print(f"Loading quantized state dict ({len(state_dict)} keys)...")
        model.load_state_dict(state_dict, strict=False)
    else:
        # Not quantized - load normally
        # Try strict first, if it fails due to missing buffers (like pos.pe), use strict=False
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            error_msg = str(e)
            # If only buffers are missing (registered_buffer items), we can safely use strict=False
            if 'Missing key(s)' in error_msg and ('pe' in error_msg or 'buffer' in error_msg.lower()):
                print(f"Warning: Some buffers missing in checkpoint, loading with strict=False")
                print(f"  Missing: {error_msg.split('Missing key(s) in state_dict:')[1].split('Unexpected')[0] if 'Missing key(s) in state_dict:' in error_msg else 'unknown'}")
                model.load_state_dict(state_dict, strict=False)
            else:
                raise
    model.eval()

    return model, is_quantized, checkpoint_size_bytes


def calculate_accuracy(model, X_test, y_test, batch_size=128):
    """Calculate accuracy manually and return predictions."""
    print(f"Calculating accuracy on {len(X_test)} samples...")

    model.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []

    num_batches = (len(X_test) + batch_size - 1) // batch_size

    with torch.no_grad():
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(X_test))

            batch_X = X_test[start_idx:end_idx]
            batch_y = y_test[start_idx:end_idx]

            # Ensure shape is (batch, 2, 1000)
            if batch_X.shape[1] == 1000 and batch_X.shape[2] == 2:
                batch_X = batch_X.transpose(1, 2)

            outputs = model(batch_X)
            predictions = outputs.argmax(dim=1)

            correct += (predictions == batch_y).sum().item()
            total += batch_y.size(0)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    accuracy = correct / total
    return accuracy, np.array(all_predictions), np.array(all_labels)


def measure_latency(model, X_test, batch_size=128, num_runs=100):
    """Measure inference latency."""
    print(f"Measuring latency with {num_runs} runs...")

    model.eval()

    # Get test batch
    test_batch = X_test[:batch_size]
    if test_batch.shape[1] == 1000 and test_batch.shape[2] == 2:
        test_batch = test_batch.transpose(1, 2)

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(test_batch)

    # Measure
    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model(test_batch)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # ms

    latencies = np.array(latencies)

    return {
        'mean_ms': float(np.mean(latencies)),
        'std_ms': float(np.std(latencies)),
        'min_ms': float(np.min(latencies)),
        'max_ms': float(np.max(latencies)),
        'median_ms': float(np.median(latencies)),
        'p95_ms': float(np.percentile(latencies, 95)),
        'p99_ms': float(np.percentile(latencies, 99))
    }


def calculate_model_size(model):
    """Calculate model parameters and size."""
    total_params = sum(p.numel() for p in model.parameters())

    size_bytes = 0
    for param in model.parameters():
        size_bytes += param.numel() * param.element_size()
    for buffer in model.buffers():
        size_bytes += buffer.numel() * buffer.element_size()

    size_kb = size_bytes / 1024
    size_mb = size_bytes / (1024 * 1024)

    return total_params, size_kb, size_mb


def plot_confusion_matrix(y_true, y_pred, output_path, class_names=None):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)

    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Default class names if not provided
    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(cm))]

    # Plot counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    ax1.set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('True Label', fontsize=12)
    ax1.set_xlabel('Predicted Label', fontsize=12)

    # Plot percentages
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues', ax=ax2,
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Percentage (%)'})
    ax2.set_title('Confusion Matrix (Percentages)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('True Label', fontsize=12)
    ax2.set_xlabel('Predicted Label', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Confusion matrix saved to {output_path}")

    # Calculate per-class metrics
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)

    return {
        'confusion_matrix': cm.tolist(),
        'confusion_matrix_percent': cm_percent.tolist(),
        'per_class_accuracy': per_class_accuracy.tolist()
    }


def main():
    parser = argparse.ArgumentParser(description='Analyze trained model')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--x_val', type=str, required=True)
    parser.add_argument('--y_val', type=str, required=True)
    parser.add_argument('--training_json', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='graph/data')
    parser.add_argument('--batch_size', type=int, default=128)

    args = parser.parse_args()

    print("="*60)
    print("MODEL ANALYSIS")
    print("="*60)

    # Create output directory first
    output_path = Path(args.output_dir)
    model_name = Path(args.model_path).stem
    cm_output_dir = output_path / model_name
    cm_output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"\nLoading validation data...")
    X_test = torch.load(args.x_val, weights_only=False)
    y_test = torch.load(args.y_val, weights_only=False)
    print(f"X shape: {X_test.shape}, y shape: {y_test.shape}")

    # Load model
    print(f"\nLoading model...")
    model, is_quantized, checkpoint_size_bytes = load_quantized_model(args.model_path, args.training_json)
    print(f"✓ Model loaded (quantized={is_quantized})")

    # Model size
    print(f"\n{'='*60}")
    print("MODEL SIZE")
    print("="*60)
    total_params, _, _ = calculate_model_size(model)
    # Use checkpoint size instead of calculated size to avoid counting uninitialized buffers
    size_kb = checkpoint_size_bytes / 1024
    size_mb = checkpoint_size_bytes / (1024 * 1024)
    print(f"Total parameters: {total_params:,}")
    print(f"Model size: {size_kb:.2f} KB")

    # Accuracy and confusion matrix
    print(f"\n{'='*60}")
    print("ACCURACY")
    print("="*60)
    accuracy, predictions, labels = calculate_accuracy(model, X_test, y_test, args.batch_size)
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    # Generate confusion matrix
    print(f"\n{'='*60}")
    print("CONFUSION MATRIX")
    print("="*60)
    cm_output_path = cm_output_dir / 'confusion_matrix.png'

    # MIT-BIH Arrhythmia 5-class labels
    class_names = ['N', 'S', 'V', 'F', 'Q']  # Normal, Supraventricular, Ventricular, Fusion, Unknown

    cm_metrics = plot_confusion_matrix(labels, predictions, cm_output_path, class_names)

    print("Per-class accuracy:")
    for i, class_name in enumerate(class_names):
        print(f"  {class_name}: {cm_metrics['per_class_accuracy'][i]*100:.2f}%")

    # Latency
    print(f"\n{'='*60}")
    print("INFERENCE LATENCY")
    print("="*60)
    latency = measure_latency(model, X_test, args.batch_size, num_runs=100)
    print(f"Mean: {latency['mean_ms']:.3f} ± {latency['std_ms']:.3f} ms")
    print(f"Median: {latency['median_ms']:.3f} ms")
    print(f"P95: {latency['p95_ms']:.3f} ms")
    print(f"P99: {latency['p99_ms']:.3f} ms")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print("="*60)
    print(f"Parameters: {total_params:,}")
    print(f"Size: {size_mb:.2f} MB")
    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"Latency: {latency['mean_ms']:.3f} ms")
    print(f"Throughput: {1000 / latency['mean_ms'] * args.batch_size:.0f} samples/sec")
    print("="*60)

    # Save results
    output_file = cm_output_dir / 'analysis_results.json'

    results = {
        'model_name': model_name,
        'model_path': str(args.model_path),
        'is_quantized': is_quantized,
        'model_size': {
            'total_parameters': int(total_params),
            'size_kb': float(size_kb),
        },
        'accuracy': float(accuracy),
        'confusion_matrix': cm_metrics,
        'inference_latency': latency
    }

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to {output_file}")
    print(f"✓ All outputs saved to: {cm_output_dir}")


if __name__ == '__main__':
    main()
