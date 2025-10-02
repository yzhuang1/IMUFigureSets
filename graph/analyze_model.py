"""
Model Analysis Script
Analyzes trained models and generates metrics for paper figures/tables.

Usage:
    python graph/analyze_model.py --model_path trained_models/model.pth --x_val <path> --y_val <path>
"""

import torch
import torch.nn as nn
import numpy as np
import json
import time
import argparse
from pathlib import Path
from sklearn.metrics import confusion_matrix, accuracy_score
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def count_parameters(model):
    """Count total and trainable parameters in the model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def get_model_size_mb(model):
    """Calculate model size in MB."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


def measure_inference_latency(model, x_val, device, num_runs=100, warmup_runs=10):
    """Measure inference latency with warmup runs."""
    model.eval()

    # Warmup runs
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(x_val)

    # Actual measurement
    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device.type == 'cuda':
                torch.cuda.synchronize()

            start_time = time.perf_counter()
            _ = model(x_val)

            if device.type == 'cuda':
                torch.cuda.synchronize()

            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)  # Convert to ms

    return {
        'mean_ms': np.mean(latencies),
        'std_ms': np.std(latencies),
        'min_ms': np.min(latencies),
        'max_ms': np.max(latencies),
        'median_ms': np.median(latencies),
        'p95_ms': np.percentile(latencies, 95),
        'p99_ms': np.percentile(latencies, 99),
        'all_latencies': latencies
    }


def compute_confusion_matrix(model, x_val, y_val, device):
    """Compute confusion matrix and accuracy."""
    model.eval()

    with torch.no_grad():
        outputs = model(x_val)

        # Handle different output formats
        if isinstance(outputs, torch.Tensor):
            if outputs.dim() == 1:  # Binary classification or regression
                predictions = (outputs > 0.5).cpu().numpy().astype(int)
            else:  # Multi-class classification
                predictions = torch.argmax(outputs, dim=1).cpu().numpy()
        else:
            raise ValueError(f"Unexpected model output type: {type(outputs)}")

    y_true = y_val.cpu().numpy()
    if y_true.dim() > 1:
        y_true = y_true.squeeze()

    # Compute metrics
    accuracy = accuracy_score(y_true, predictions)
    cm = confusion_matrix(y_true, predictions)

    return {
        'accuracy': float(accuracy),
        'confusion_matrix': cm.tolist(),
        'predictions': predictions.tolist(),
        'true_labels': y_true.tolist()
    }


def analyze_model(model_path, x_val, y_val, output_dir='graph/data'):
    """
    Main analysis function.

    Args:
        model_path: Path to the .pth model file
        x_val: Validation input tensor
        y_val: Validation labels tensor
        output_dir: Directory to save analysis results
    """
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model_state = checkpoint['model_state_dict']
            model_class = checkpoint.get('model_class', None)
            model_code = checkpoint.get('model_code', None)
        elif 'state_dict' in checkpoint:
            model_state = checkpoint['state_dict']
            model_class = checkpoint.get('model_class', None)
            model_code = checkpoint.get('model_code', None)
        else:
            model_state = checkpoint
            model_class = None
            model_code = None
    else:
        # Checkpoint is the state dict itself
        model_state = checkpoint
        model_class = None
        model_code = None

    # Try to reconstruct model from code if available
    if model_code and model_class:
        print("Reconstructing model from saved code...")
        exec_globals = {'torch': torch, 'nn': nn}
        exec(model_code, exec_globals)
        model = exec_globals[model_class]()
        model.load_state_dict(model_state)
    else:
        print("ERROR: Model architecture not found in checkpoint.")
        print("Please ensure your checkpoint includes 'model_code' and 'model_class'.")
        return None

    model.to(device)
    x_val = x_val.to(device)
    y_val = y_val.to(device)

    # Extract model name from path
    model_name = Path(model_path).stem

    print(f"\nAnalyzing model: {model_name}")
    print("=" * 60)

    # 1. Model Size Analysis
    print("\n1. Computing model size metrics...")
    total_params, trainable_params = count_parameters(model)
    model_size_mb = get_model_size_mb(model)

    # 2. Inference Latency
    print("2. Measuring inference latency...")
    latency_stats = measure_inference_latency(model, x_val, device)

    # 3. Confusion Matrix and Accuracy
    print("3. Computing confusion matrix and accuracy...")
    classification_metrics = compute_confusion_matrix(model, x_val, y_val, device)

    # Compile results
    results = {
        'model_name': model_name,
        'model_path': str(model_path),
        'device': str(device),
        'model_size': {
            'total_parameters': int(total_params),
            'trainable_parameters': int(trainable_params),
            'size_mb': float(model_size_mb)
        },
        'accuracy': classification_metrics['accuracy'],
        'inference_latency': {
            'mean_ms': latency_stats['mean_ms'],
            'std_ms': latency_stats['std_ms'],
            'min_ms': latency_stats['min_ms'],
            'max_ms': latency_stats['max_ms'],
            'median_ms': latency_stats['median_ms'],
            'p95_ms': latency_stats['p95_ms'],
            'p99_ms': latency_stats['p99_ms']
        },
        'confusion_matrix': classification_metrics['confusion_matrix']
    }

    # Save results
    output_file = output_path / f'{model_name}_analysis.json'
    print(f"\nSaving results to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Save detailed latency data
    latency_file = output_path / f'{model_name}_latencies.json'
    with open(latency_file, 'w') as f:
        json.dump(latency_stats['all_latencies'], f)

    # Save predictions for further analysis
    predictions_file = output_path / f'{model_name}_predictions.json'
    with open(predictions_file, 'w') as f:
        json.dump({
            'predictions': classification_metrics['predictions'],
            'true_labels': classification_metrics['true_labels']
        }, f)

    # Print summary
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Model Size: {model_size_mb:.2f} MB")
    print(f"Accuracy: {classification_metrics['accuracy']:.4f}")
    print(f"Mean Inference Latency: {latency_stats['mean_ms']:.3f} ± {latency_stats['std_ms']:.3f} ms")
    print(f"Median Latency: {latency_stats['median_ms']:.3f} ms")
    print(f"P95 Latency: {latency_stats['p95_ms']:.3f} ms")
    print("=" * 60)

    return results


def main():
    parser = argparse.ArgumentParser(description='Analyze trained model and generate metrics')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the .pth model file')
    parser.add_argument('--x_val', type=str, required=True,
                       help='Path to validation input tensor (.pt or .pth)')
    parser.add_argument('--y_val', type=str, required=True,
                       help='Path to validation labels tensor (.pt or .pth)')
    parser.add_argument('--output_dir', type=str, default='graph/data',
                       help='Directory to save analysis results (default: graph/data)')

    args = parser.parse_args()

    # Load validation data
    print(f"Loading validation data...")
    x_val = torch.load(args.x_val)
    y_val = torch.load(args.y_val)

    # Run analysis
    analyze_model(args.model_path, x_val, y_val, args.output_dir)


if __name__ == '__main__':
    main()
