"""
Compare Calibration Methods

This script creates comparison visualizations showing reliability diagrams
for all calibration methods: Uncalibrated Softmax, Platt Scaling, 
Temperature Scaling, and Isotonic Regression.
"""

import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import joblib
from sklearn.metrics import log_loss, accuracy_score
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.special import softmax
import json

from confidence_estimation import (
    get_logits_from_model,
    evaluate_calibration,
    temperature_scaling
)


def plot_comparison_reliability_diagrams(model, model_name, X_test, y_test, 
                                         platt_scaler, temperature, iso_calibrator,
                                         output_path, n_bins=15):
    """
    Plot 2x2 comparison of all calibration methods.
    
    Args:
        model: Trained model
        model_name: Name of the model
        X_test: Test features
        y_test: Test labels
        platt_scaler: Fitted Platt scaling calibrator
        temperature: Optimal temperature value
        iso_calibrator: Fitted isotonic calibrator
        output_path: Where to save the plot
        n_bins: Number of bins for calibration
    """
    # Get predictions for all methods
    logits = get_logits_from_model(model, X_test)
    
    # 1. Uncalibrated (softmax)
    probs_softmax = softmax(logits, axis=1)
    
    # 2. Platt Scaling
    probs_platt = platt_scaler.predict_proba(logits)
    
    # 3. Temperature Scaling
    probs_temp = softmax(temperature_scaling(logits, temperature), axis=1)
    
    # 4. Isotonic Regression
    probs_isotonic = iso_calibrator.predict_proba(probs_softmax)
    
    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    methods = [
        ('Uncalibrated (Softmax)', probs_softmax),
        ('Platt Scaling', probs_platt),
        ('Temperature Scaling', probs_temp),
        ('Isotonic Regression', probs_isotonic)
    ]
    
    for idx, (method_name, probs) in enumerate(methods):
        ax = axes[idx]
        
        # Calculate metrics
        confidences = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1)
        accuracies = (predictions == y_test)
        
        accuracy = accuracy_score(y_test, predictions)
        nll = log_loss(y_test, probs)
        ece = evaluate_calibration(probs, y_test, n_bins)
        
        # Calculate MCE
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        bin_accuracies = []
        bin_confidences = []
        mce = 0.0
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            bin_center = (bin_lower + bin_upper) / 2
            
            if np.sum(in_bin) > 0:
                accuracy_in_bin = np.mean(accuracies[in_bin])
                avg_confidence_in_bin = np.mean(confidences[in_bin])
                bin_accuracies.append(accuracy_in_bin)
                bin_confidences.append(bin_center)
                mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
            else:
                bin_accuracies.append(0)
                bin_confidences.append(bin_center)
        
        # Plot
        # Plot the actual model output bars first
        ax.bar(bin_confidences, bin_accuracies, width=1.0/n_bins, 
               alpha=1.0, edgecolor='black', linewidth=2, label='Model Output', color='#0000ff')
        
        # Plot the gap between actual and perfect calibration ON TOP
        for i, (conf, acc) in enumerate(zip(bin_confidences, bin_accuracies)):
            # Only plot if this bin has samples (acc > 0 or explicitly tracked)
            # Check if bin had samples by looking at the calculation
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            
            if np.sum(in_bin) > 0:
                gap_height = abs(conf - acc)
                gap_bottom = min(conf, acc)
                
                # Choose color based on whether actual is below or above ideal
                if acc < conf:  # Overconfident (actual below ideal)
                    color = '#FF6347'  # Red-orange (tomato)
                else:  # Underconfident (actual above ideal)
                    color = '#8235b2'  # Purple
                
                if gap_height > 0:
                    ax.bar(conf, gap_height, width=1.0/n_bins, 
                           bottom=gap_bottom, alpha=0.7, edgecolor='black', 
                           color=color, hatch='/', linewidth=0.5, zorder=3)
        
        # Plot perfect calibration line on top of everything
        ax.plot([0, 1], [0, 1], 'k--', label='Expected', linewidth=2, zorder=4)
        
        # Create custom legend with Gap always shown as red-orange
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#0000ff', edgecolor='black', linewidth=2, label='Accuracy'),
            Patch(facecolor='#FF6347', edgecolor='black', hatch='/', alpha=0.7, label='Gap (Calibration Error)'),
            plt.Line2D([0], [0], color='k', linestyle='--', linewidth=2, label='Expected')
        ]
        
        ax.set_xlabel('Confidence', fontsize=16)
        ax.set_ylabel('Accuracy', fontsize=16)
        ax.set_title(f'{method_name}', fontsize=18, fontweight='bold')
        ax.legend(handles=legend_elements, loc='upper left', fontsize=16)
        ax.grid(alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.tick_params(axis='both', which='major', labelsize=14)
        
        # Add ECE text box (bottom right)
        ece_text = f'ECE: {ece:.4f}'
        ax.text(0.95, 0.05, ece_text, transform=ax.transAxes,
                fontsize=16, verticalalignment='bottom', horizontalalignment='right',
                fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
    
    #plt.suptitle(f'Calibration Methods Comparison - {model_name}', fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Return ECE values for summary
    return {
        'Uncalibrated': evaluate_calibration(probs_softmax, y_test, n_bins),
        'Platt': evaluate_calibration(probs_platt, y_test, n_bins),
        'Temperature': evaluate_calibration(probs_temp, y_test, n_bins),
        'Isotonic': evaluate_calibration(probs_isotonic, y_test, n_bins)
    }


def main():
    print("="*60)
    print("CALIBRATION METHODS COMPARISON")
    print("="*60)
    
    # Load test data
    print("\nLoading test data...")
    X_test = joblib.load('features/combined/combined_test.joblib')
    y_test = joblib.load('features/combined/labels_test.joblib')
    
    X_test = np.asarray(X_test.get('features', X_test))
    y_test = np.asarray(y_test.get('labels', y_test))
    
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Load calibrators
    print("\nLoading calibrators...")
    platt_scalers = joblib.load('evaluation_results/platt_scalers.joblib')
    
    with open('evaluation_results/temperature_scaling_results.json', 'r') as f:
        temp_results = json.load(f)
    
    iso_calibrators = joblib.load('evaluation_results/isotonic_calibrators.joblib')
    
    # Models to evaluate
    models_to_evaluate = [
        ('features/model_rf.joblib', 'RF'),
        ('features/model_svm.joblib', 'SVM'),
        ('features/model_gb.joblib', 'GB'),
    ]
    
    viz_dir = Path('visualizations/calibration/comparison')
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    all_eces = {}
    
    for model_path, model_name in models_to_evaluate:
        if not Path(model_path).exists():
            print(f"\n{'='*60}")
            print(f"Model not found: {model_name}")
            print('='*60)
            continue
        
        print(f"\n{'='*60}")
        print(f"Comparing Calibration Methods: {model_name}")
        print('='*60)
        
        # Load model
        model_data = joblib.load(model_path)
        model = model_data.best_estimator_ if hasattr(model_data, 'best_estimator_') else model_data
        
        # Get calibrators for this model
        platt_scaler = platt_scalers[model_name]
        temperature = temp_results[model_name]['temperature']
        iso_calibrator = iso_calibrators[model_name]
        
        # Create comparison plot
        safe_name = model_name.lower().replace(' ', '_')
        plot_path = viz_dir / f'{safe_name}_calibration_comparison.png'
        
        eces = plot_comparison_reliability_diagrams(
            model, model_name, X_test, y_test,
            platt_scaler, temperature, iso_calibrator,
            plot_path, n_bins=15
        )
        
        all_eces[model_name] = eces
        
        print(f"\nECE Comparison for {model_name}:")
        print(f"  Uncalibrated:        {eces['Uncalibrated']:.4f}")
        print(f"  Platt Scaling:       {eces['Platt']:.4f}")
        print(f"  Temperature Scaling: {eces['Temperature']:.4f}")
        print(f"  Isotonic Regression: {eces['Isotonic']:.4f}")
        print(f"\nSaved comparison plot to: {plot_path}")
    
    # Summary table
    print("\n" + "="*60)
    print("CALIBRATION METHODS SUMMARY (ECE)")
    print("="*60)
    
    print(f"\n{'Model':<15} {'Uncalibrated':<15} {'Platt':<15} {'Temperature':<15} {'Isotonic':<15}")
    print("-" * 90)
    
    for model_name, eces in all_eces.items():
        print(f"{model_name:<15} {eces['Uncalibrated']:<15.4f} {eces['Platt']:<15.4f} "
              f"{eces['Temperature']:<15.4f} {eces['Isotonic']:<15.4f}")
    
    print("\n" + "="*60)
    print("COMPARISON COMPLETED")
    print("="*60)


if __name__ == '__main__':
    main()
