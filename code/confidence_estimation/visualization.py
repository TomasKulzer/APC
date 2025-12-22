"""Visualization utilities for temperature scaling calibration."""

import numpy as np
import matplotlib.pyplot as plt
from .temperature_scaling import evaluate_calibration


def plot_reliability_diagram(probs_before, probs_after, labels, model_name, output_path, n_bins=20):
    """
    Plot reliability diagram showing calibration before and after temperature scaling.
    
    Args:
        probs_before: Probabilities before calibration
        probs_after: Probabilities after calibration
        labels: True labels
        model_name: Name of the model
        output_path: Where to save the plot
        n_bins: Number of bins
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for idx, (probs, title) in enumerate([
        (probs_before, 'Before Temperature Scaling'),
        (probs_after, 'After Temperature Scaling')
    ]):
        ax = axes[idx]
        
        confidences = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1)
        accuracies = (predictions == labels)
        
        # Bin statistics
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            bin_center = (bin_lower + bin_upper) / 2
            if np.sum(in_bin) > 0:
                bin_accuracies.append(np.mean(accuracies[in_bin]))
                bin_confidences.append(bin_center)
                bin_counts.append(np.sum(in_bin))
            else:
                bin_accuracies.append(0)
                bin_confidences.append(bin_center)
                bin_counts.append(0)
        
        # Plot
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2)
        ax.bar(bin_confidences, bin_accuracies, width=1.0/n_bins, 
               alpha=0.7, edgecolor='black', label='Model Output')
        
        # Calculate ECE
        ece = evaluate_calibration(probs, labels, n_bins)
        
        ax.set_xlabel('Confidence', fontsize=11)
        ax.set_ylabel('Accuracy', fontsize=11)
        ax.set_title(f'{title}\nECE: {ece:.4f}', fontsize=12)
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
    
    plt.suptitle(f'Reliability Diagram - {model_name}', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
