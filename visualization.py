"""
Integrated Visualization Module for BO Results
Generates charts during pipeline execution and saves to charts folder
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import json
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def create_charts_folder():
    """Create charts folder if it doesn't exist"""
    charts_dir = Path("charts")
    charts_dir.mkdir(exist_ok=True)
    return charts_dir

def generate_bo_charts(bo_results: Dict[str, Any], save_folder: str):
    """
    Generate BO visualization charts from results
    
    Args:
        bo_results: BO results dictionary containing 'all_results'
        save_folder: Folder to save charts in (must be a specific BO run subfolder)
    """
    if not bo_results or 'all_results' not in bo_results:
        logger.warning("No BO results available for visualization")
        return
    
    trial_results = bo_results['all_results']
    if not trial_results:
        logger.warning("No trial results found for visualization")
        return
    
    # Create charts folder
    charts_dir = Path(save_folder)
    charts_dir.mkdir(exist_ok=True)
    
    # Extract data for plotting
    trials = []
    accuracies = []
    best_so_far = []
    learning_rates = []
    epochs_list = []
    hidden_sizes = []
    
    current_best = -np.inf
    
    for result in trial_results:
        trial = result.get('trial', len(trials) + 1)
        
        # Get accuracy/value
        accuracy = result.get('value', 0)
        if accuracy is None:
            continue
            
        trials.append(trial)
        accuracies.append(accuracy)
        
        # Track best so far
        current_best = max(current_best, accuracy)
        best_so_far.append(current_best)
        
        # Extract hyperparameters
        hparams = result.get('hparams', {})
        learning_rates.append(hparams.get('lr', 0.001))
        epochs_list.append(hparams.get('epochs', 5))
        hidden_sizes.append(hparams.get('hidden', 64))
    
    if not trials:
        logger.warning("No valid trial data for visualization")
        return
    
    # Generate charts
    logger.info(f"Generating BO visualization charts with {len(trials)} trials...")
    
    # Chart 1: Accuracy Progression
    _plot_accuracy_progression(trials, accuracies, best_so_far, charts_dir)
    
    # Chart 2: Hyperparameter Analysis
    _plot_hyperparameter_analysis(trials, accuracies, learning_rates, epochs_list, hidden_sizes, charts_dir)
    
    # Chart 3: Convergence Analysis
    _plot_convergence_analysis(trials, accuracies, best_so_far, charts_dir)
    
    # Generate summary stats
    _generate_summary_stats(bo_results, accuracies, learning_rates, epochs_list, hidden_sizes, charts_dir)

    # Save raw data
    _save_raw_data(bo_results, trials, accuracies, best_so_far, learning_rates, epochs_list, hidden_sizes, charts_dir)

    logger.info(f"BO charts saved to: {charts_dir}")

def _plot_accuracy_progression(trials, accuracies, best_so_far, charts_dir):
    """Plot accuracy progression over BO trials"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Individual trial accuracies and best so far
    ax1.plot(trials, accuracies, 'o-', label='Trial Accuracy', 
            color='lightblue', alpha=0.7, markersize=6)
    ax1.plot(trials, best_so_far, 'r-', label='Best So Far', 
            linewidth=3, alpha=0.8)
    
    ax1.set_xlabel('Trial Number')
    ax1.set_ylabel('Accuracy/F1 Score')
    ax1.set_title('Bayesian Optimization: Accuracy Progression')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim([min(accuracies) - 0.05, max(accuracies) + 0.05])
    
    # Add annotation for best trial
    best_idx = np.argmax(accuracies)
    ax1.annotate(f'Best: {accuracies[best_idx]:.4f}',
                xy=(trials[best_idx], accuracies[best_idx]),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # Plot 2: Improvement over time
    improvements = np.diff([best_so_far[0]] + best_so_far)
    ax2.bar(trials, improvements, alpha=0.6, color='green')
    ax2.set_xlabel('Trial Number')
    ax2.set_ylabel('Improvement')
    ax2.set_title('Accuracy Improvements by Trial')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(charts_dir / "accuracy_progression.png", dpi=300, bbox_inches='tight')
    plt.close()

def _plot_hyperparameter_analysis(trials, accuracies, learning_rates, epochs_list, hidden_sizes, charts_dir):
    """Plot hyperparameter vs accuracy analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Learning Rate vs Accuracy
    scatter1 = axes[0, 0].scatter(learning_rates, accuracies, 
                                 c=trials, cmap='viridis', alpha=0.7, s=60)
    axes[0, 0].set_xlabel('Learning Rate (log scale)')
    axes[0, 0].set_ylabel('Accuracy/F1 Score')
    axes[0, 0].set_title('Learning Rate vs Accuracy')
    axes[0, 0].set_xscale('log')
    axes[0, 0].grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=axes[0, 0], label='Trial Number')
    
    # Epochs vs Accuracy
    scatter2 = axes[0, 1].scatter(epochs_list, accuracies, 
                                 c=trials, cmap='viridis', alpha=0.7, s=60)
    axes[0, 1].set_xlabel('Epochs')
    axes[0, 1].set_ylabel('Accuracy/F1 Score')
    axes[0, 1].set_title('Epochs vs Accuracy')
    axes[0, 1].grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=axes[0, 1], label='Trial Number')
    
    # Hidden Size vs Accuracy
    scatter3 = axes[1, 0].scatter(hidden_sizes, accuracies, 
                                 c=trials, cmap='viridis', alpha=0.7, s=60)
    axes[1, 0].set_xlabel('Hidden Size')
    axes[1, 0].set_ylabel('Accuracy/F1 Score')
    axes[1, 0].set_title('Hidden Size vs Accuracy')
    axes[1, 0].grid(True, alpha=0.3)
    plt.colorbar(scatter3, ax=axes[1, 0], label='Trial Number')
    
    # Trial progression (accuracy over time)
    axes[1, 1].plot(trials, accuracies, 'bo-', alpha=0.7, label='Trial Accuracy')
    axes[1, 1].set_xlabel('Trial Number')
    axes[1, 1].set_ylabel('Accuracy/F1 Score')
    axes[1, 1].set_title('Trial Progression')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(charts_dir / "hyperparameter_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()

def _plot_convergence_analysis(trials, accuracies, best_so_far, charts_dir):
    """Plot BO convergence analysis"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Regret plot
    theoretical_best = max(accuracies)
    regret = [max(0.001, theoretical_best - acc) for acc in best_so_far]
    
    ax1.semilogy(trials, regret, 'b-', linewidth=2, marker='o', markersize=4)
    ax1.set_xlabel('Trial Number')
    ax1.set_ylabel('Simple Regret (log scale)')
    ax1.set_title('BO Convergence: Simple Regret')
    ax1.grid(True, alpha=0.3)
    
    # Running average
    window_size = min(5, len(accuracies) // 2)
    if window_size >= 2:
        running_avg = []
        for i in range(len(accuracies)):
            start_idx = max(0, i - window_size + 1)
            avg = np.mean(accuracies[start_idx:i+1])
            running_avg.append(avg)
        
        ax2.plot(trials, accuracies, 'o', alpha=0.5, label='Individual Trials')
        ax2.plot(trials, running_avg, 'r-', linewidth=2, 
                label=f'Running Average (window={window_size})')
        ax2.plot(trials, best_so_far, 'g-', linewidth=2, label='Best So Far')
    else:
        ax2.plot(trials, accuracies, 'bo-', alpha=0.7, label='Accuracies')
        ax2.plot(trials, best_so_far, 'r-', linewidth=2, label='Best So Far')
    
    ax2.set_xlabel('Trial Number')
    ax2.set_ylabel('Accuracy/F1 Score')
    ax2.set_title('BO Convergence: Running Statistics')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(charts_dir / "convergence_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()

def _generate_summary_stats(bo_results, accuracies, learning_rates, epochs_list, hidden_sizes, charts_dir):
    """Generate and save summary statistics"""
    if not accuracies:
        return
    
    stats = {
        'Total Trials': len(accuracies),
        'Best Accuracy': max(accuracies),
        'Final Accuracy': accuracies[-1],
        'Mean Accuracy': np.mean(accuracies),
        'Std Accuracy': np.std(accuracies),
        'Improvement': max(accuracies) - accuracies[0],
        'Best Trial': np.argmax(accuracies) + 1
    }
    
    # Best hyperparameters
    best_idx = np.argmax(accuracies)
    best_hparams = {
        'Learning Rate': learning_rates[best_idx],
        'Epochs': epochs_list[best_idx],
        'Hidden Size': hidden_sizes[best_idx]
    }
    
    # Save to text file
    summary_file = charts_dir / "bo_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("BAYESIAN OPTIMIZATION SUMMARY\n")
        f.write("=" * 60 + "\n")
        
        for key, value in stats.items():
            if isinstance(value, float):
                f.write(f"{key:<20}: {value:.4f}\n")
            else:
                f.write(f"{key:<20}: {value}\n")
        
        f.write("\nBEST HYPERPARAMETERS:\n")
        f.write("-" * 30 + "\n")
        for key, value in best_hparams.items():
            if isinstance(value, float):
                f.write(f"{key:<15}: {value:.6f}\n")
            else:
                f.write(f"{key:<15}: {value}\n")
        
        # Additional BO info
        if 'ai_recommendation' in bo_results:
            ai_rec = bo_results['ai_recommendation']
            f.write(f"\nAI RECOMMENDATION:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Model: {ai_rec.get('model_name', 'Unknown')}\n")
            f.write(f"Reasoning: {ai_rec.get('reasoning', 'Not available')}\n")
            f.write(f"Confidence: {ai_rec.get('confidence', 0):.2f}\n")

    logger.info(f"BO summary saved to: {summary_file}")

def _save_raw_data(bo_results, trials, accuracies, best_so_far, learning_rates, epochs_list, hidden_sizes, charts_dir):
    """Save all raw data used to generate charts in JSON and numpy formats"""

    # Prepare data dictionary with all plot data
    raw_data = {
        'trials': trials,
        'accuracies': accuracies,
        'best_so_far': best_so_far,
        'hyperparameters': {
            'learning_rates': learning_rates,
            'epochs': epochs_list,
            'hidden_sizes': hidden_sizes
        },
        'statistics': {
            'total_trials': len(accuracies),
            'best_accuracy': float(max(accuracies)) if accuracies else 0.0,
            'final_accuracy': float(accuracies[-1]) if accuracies else 0.0,
            'mean_accuracy': float(np.mean(accuracies)) if accuracies else 0.0,
            'std_accuracy': float(np.std(accuracies)) if accuracies else 0.0,
            'improvement': float(max(accuracies) - accuracies[0]) if len(accuracies) >= 2 else 0.0,
            'best_trial': int(np.argmax(accuracies) + 1) if accuracies else 0
        },
        'full_bo_results': bo_results  # Include complete BO results with all hparams
    }

    # Save as JSON
    json_file = charts_dir / "bo_raw_data.json"
    with open(json_file, 'w') as f:
        json.dump(raw_data, f, indent=2)
    logger.info(f"Raw data saved to: {json_file}")

    # Also save as numpy arrays for easy loading in Python
    npz_file = charts_dir / "bo_raw_data.npz"
    np.savez(
        npz_file,
        trials=np.array(trials),
        accuracies=np.array(accuracies),
        best_so_far=np.array(best_so_far),
        learning_rates=np.array(learning_rates),
        epochs=np.array(epochs_list),
        hidden_sizes=np.array(hidden_sizes)
    )
    logger.info(f"Numpy arrays saved to: {npz_file}")