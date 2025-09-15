"""
Metric Validation Module
Validates training metrics to detect common bugs and issues
"""

import numpy as np
import torch
from typing import Dict, Any, List, Tuple, Optional
import logging
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

logger = logging.getLogger(__name__)

def validate_metrics(metrics: Dict[str, Any], expected_metrics: List[str] = None) -> Dict[str, Any]:
    """
    Validate training metrics for common issues
    
    Args:
        metrics: Dictionary of computed metrics
        expected_metrics: List of expected metric names
        
    Returns:
        Dictionary with validation results and warnings
    """
    validation_results = {
        'valid': True,
        'warnings': [],
        'errors': [],
        'recommendations': []
    }
    
    expected_metrics = expected_metrics or ['val_accuracy', 'macro_f1', 'final_loss']
    
    # Check for missing metrics
    missing_metrics = set(expected_metrics) - set(metrics.keys())
    if missing_metrics:
        validation_results['errors'].append(f"Missing metrics: {missing_metrics}")
        validation_results['valid'] = False
    
    # Check for identical accuracy and F1 (common bug indicator)
    if 'val_accuracy' in metrics and 'macro_f1' in metrics:
        acc = metrics['val_accuracy']
        f1 = metrics['macro_f1']
        
        if isinstance(acc, (int, float)) and isinstance(f1, (int, float)):
            if abs(acc - f1) < 1e-10:  # Essentially identical
                validation_results['errors'].append(
                    f"Identical val_accuracy ({acc:.6f}) and macro_f1 ({f1:.6f}) - likely bug in F1 calculation"
                )
                validation_results['valid'] = False
                validation_results['recommendations'].append(
                    "Fix macro_f1 calculation to use sklearn.metrics.f1_score instead of accuracy"
                )
    
    # Check for suspicious metric values
    for metric_name, value in metrics.items():
        if isinstance(value, (int, float)):
            # Check for out-of-range values
            if metric_name in ['val_accuracy', 'macro_f1'] and not (0 <= value <= 1):
                validation_results['warnings'].append(
                    f"{metric_name} = {value} is outside valid range [0, 1]"
                )
            
            # Check for suspiciously perfect scores
            if metric_name in ['val_accuracy', 'macro_f1'] and value > 0.99:
                validation_results['warnings'].append(
                    f"{metric_name} = {value} is suspiciously high - check for data leakage"
                )
            
            # Check for suspiciously low scores (might indicate training failure)
            if metric_name in ['val_accuracy', 'macro_f1'] and value < 0.1:
                validation_results['warnings'].append(
                    f"{metric_name} = {value} is suspiciously low - check training process"
                )
    
    return validation_results

def compute_comprehensive_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    class_names: List[str] = None
) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Optional class names for detailed reporting
        
    Returns:
        Dictionary with comprehensive metrics
    """
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    # Class distribution
    unique_true, counts_true = np.unique(y_true, return_counts=True)
    unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
    
    metrics = {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'weighted_f1': weighted_f1,
        'per_class_precision': precision.tolist(),
        'per_class_recall': recall.tolist(),
        'per_class_f1': f1.tolist(),
        'per_class_support': support.tolist(),
        'true_class_distribution': dict(zip(unique_true, counts_true)),
        'pred_class_distribution': dict(zip(unique_pred, counts_pred))
    }
    
    if class_names:
        # Add named metrics for easier interpretation
        for i, name in enumerate(class_names):
            if i < len(precision):
                metrics[f'{name}_precision'] = precision[i]
                metrics[f'{name}_recall'] = recall[i] 
                metrics[f'{name}_f1'] = f1[i]
                metrics[f'{name}_support'] = int(support[i])
    
    return metrics

def detect_metric_anomalies(
    metric_history: List[Dict[str, Any]], 
    metric_name: str = 'val_accuracy'
) -> Dict[str, Any]:
    """
    Detect anomalies in metric progression across BO trials
    
    Args:
        metric_history: List of metric dictionaries from multiple trials
        metric_name: Metric to analyze
        
    Returns:
        Dictionary with anomaly detection results
    """
    if not metric_history or metric_name not in metric_history[0]:
        return {'anomalies_detected': False, 'reason': 'Insufficient data'}
    
    values = [metrics[metric_name] for metrics in metric_history if metric_name in metrics]
    
    if len(values) < 2:
        return {'anomalies_detected': False, 'reason': 'Need at least 2 data points'}
    
    values = np.array(values)
    
    # Check for identical values (major red flag)
    unique_values = np.unique(values)
    if len(unique_values) == 1:
        return {
            'anomalies_detected': True,
            'anomaly_type': 'identical_values',
            'message': f'All {len(values)} trials have identical {metric_name}: {values[0]:.6f}',
            'severity': 'critical'
        }
    
    # Check for very low variance (suspicious)
    variance = np.var(values)
    mean_value = np.mean(values)
    coefficient_of_variation = np.sqrt(variance) / mean_value if mean_value != 0 else 0
    
    if coefficient_of_variation < 0.001:  # Less than 0.1% variation
        return {
            'anomalies_detected': True,
            'anomaly_type': 'low_variance',
            'message': f'{metric_name} has suspiciously low variance: std={np.std(values):.6f}, mean={mean_value:.6f}',
            'severity': 'warning'
        }
    
    # Check for lack of improvement
    if len(values) >= 5:
        recent_improvement = max(values[-3:]) - max(values[:3])
        if recent_improvement < 0.001:  # Less than 0.1% improvement
            return {
                'anomalies_detected': True,
                'anomaly_type': 'no_improvement',
                'message': f'No improvement in {metric_name} over {len(values)} trials (improvement: {recent_improvement:.6f})',
                'severity': 'warning'
            }
    
    return {'anomalies_detected': False, 'reason': 'No anomalies detected'}

def validate_bo_results(bo_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate Bayesian Optimization results for common issues
    
    Args:
        bo_results: BO results dictionary
        
    Returns:
        Validation results
    """
    validation = {
        'valid': True,
        'issues': [],
        'recommendations': []
    }
    
    # Check for trial history
    if 'trial_history' not in bo_results:
        validation['issues'].append("Missing trial history in BO results")
        validation['valid'] = False
        return validation
    
    trial_history = bo_results['trial_history']
    
    if len(trial_history) == 0:
        validation['issues'].append("Empty trial history")
        validation['valid'] = False
        return validation
    
    # Analyze metric progression
    anomalies = detect_metric_anomalies(trial_history, 'val_accuracy')
    if anomalies['anomalies_detected']:
        validation['issues'].append(f"Metric anomaly detected: {anomalies['message']}")
        if anomalies['severity'] == 'critical':
            validation['valid'] = False
    
    # Check hyperparameter diversity
    if len(trial_history) > 1:
        # Extract hyperparameters from trials
        hparam_keys = set()
        for trial in trial_history:
            if 'hyperparameters' in trial:
                hparam_keys.update(trial['hyperparameters'].keys())
        
        # Check if hyperparameters are actually varying
        for key in hparam_keys:
            values = []
            for trial in trial_history:
                if 'hyperparameters' in trial and key in trial['hyperparameters']:
                    values.append(trial['hyperparameters'][key])
            
            if len(set(values)) == 1:
                validation['issues'].append(f"Hyperparameter '{key}' is not varying across trials")
                validation['recommendations'].append(f"Check BO search space for '{key}'")
    
    return validation

# Example usage for ECG classes
ECG_CLASS_NAMES = ['Normal', 'Bundle Branch Block', 'Atrial Premature', 'Ventricular', 'Other']

def validate_ecg_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Validate metrics specifically for ECG arrhythmia classification"""
    comprehensive_metrics = compute_comprehensive_metrics(y_true, y_pred, ECG_CLASS_NAMES)
    
    # ECG-specific validations
    validation = {
        'comprehensive_metrics': comprehensive_metrics,
        'ecg_specific_warnings': []
    }
    
    # Check for class imbalance handling
    true_dist = comprehensive_metrics['true_class_distribution']
    pred_dist = comprehensive_metrics['pred_class_distribution']
    
    # Check if minority classes are being predicted
    minority_classes = [2, 3]  # Atrial Premature, Ventricular
    for cls in minority_classes:
        if cls in true_dist and cls not in pred_dist:
            validation['ecg_specific_warnings'].append(
                f"Class {cls} ({ECG_CLASS_NAMES[cls]}) never predicted - model may be ignoring minority class"
            )
    
    # Check for reasonable performance on critical arrhythmias
    critical_classes = [2, 3]  # Atrial Premature, Ventricular
    for cls in critical_classes:
        if cls < len(comprehensive_metrics['per_class_f1']):
            f1 = comprehensive_metrics['per_class_f1'][cls]
            if f1 < 0.1:
                validation['ecg_specific_warnings'].append(
                    f"Very low F1 score ({f1:.3f}) for critical class {ECG_CLASS_NAMES[cls]} - clinical concern"
                )
    
    return validation