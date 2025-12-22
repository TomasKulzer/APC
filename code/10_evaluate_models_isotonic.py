"""
Evaluate Model Performance After Isotonic Regression Calibration

Computes comprehensive metrics for all trained models after applying 
isotonic regression calibration including:
- Accuracy, precision, recall, F1-score
- Confusion matrices
- Per-class metrics
- Comparison with uncalibrated performance
"""

import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score, mean_absolute_error, mean_squared_error,
    confusion_matrix, log_loss
)
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.special import softmax

from confidence_estimation import (
    get_logits_from_model,
    evaluate_calibration
)


def load_model(model_path):
    """Load model, handling both Pipeline and SimpleNamespace objects."""
    if not Path(model_path).exists():
        return None
    
    model_data = joblib.load(model_path)
    
    # Handle SimpleNamespace from training scripts
    if hasattr(model_data, 'best_estimator_'):
        return model_data.best_estimator_
    return model_data


def evaluate_model_isotonic(model, iso_calibrator, X, y, class_names, model_name):
    """
    Evaluate a model after isotonic calibration and return comprehensive metrics.
    
    Returns:
        dict with metrics and predictions
    """
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name} (Isotonic Calibrated)")
    print('='*60)
    
    # Get predictions with progress bar
    print("Making predictions...")
    
    # Get logits for calibration
    print("  Computing logits...")
    logits = get_logits_from_model(model, X)
    
    # Get uncalibrated probabilities
    probs_uncalibrated = softmax(logits, axis=1)
    
    # Apply isotonic calibration
    print("  Applying isotonic calibration...")
    probs_calibrated = iso_calibrator.predict_proba(probs_uncalibrated)
    
    # Get predictions from calibrated probabilities
    y_pred = np.argmax(probs_calibrated, axis=1)
    
    # Compute metrics
    print("Computing metrics...")
    accuracy = accuracy_score(y, y_pred)
    misclassification_rate = 1.0 - accuracy
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    
    # Calibration metrics
    nll_uncalibrated = log_loss(y, probs_uncalibrated)
    nll_calibrated = log_loss(y, probs_calibrated)
    ece_uncalibrated = evaluate_calibration(probs_uncalibrated, y)
    ece_calibrated = evaluate_calibration(probs_calibrated, y)
    
    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    
    # Print results
    print(f"\nMetrics:")
    print(f"  Accuracy:                  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Misclassification Rate:    {misclassification_rate:.4f} ({misclassification_rate*100:.2f}%)")
    print(f"  Mean Absolute Error (MAE): {mae:.4f}")
    print(f"  Mean Square Error (MSE):   {mse:.4f}")
    
    print(f"\nCalibration Metrics:")
    print(f"  NLL (Uncalibrated): {nll_uncalibrated:.4f}")
    print(f"  NLL (Calibrated):   {nll_calibrated:.4f}")
    print(f"  NLL Improvement:    {nll_uncalibrated - nll_calibrated:.4f}")
    print(f"  ECE (Uncalibrated): {ece_uncalibrated:.4f}")
    print(f"  ECE (Calibrated):   {ece_calibrated:.4f}")
    print(f"  ECE Improvement:    {ece_uncalibrated - ece_calibrated:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"{'':>12}", end='')
    for name in class_names:
        print(f"{name[:8]:>10}", end='')
    print()
    for i, name in enumerate(class_names):
        print(f"{name[:12]:>12}", end='')
        for j in range(len(class_names)):
            print(f"{cm[i,j]:>10}", end='')
        print()
    
    # Return results
    return {
        'model_name': model_name,
        'accuracy': float(accuracy),
        'misclassification_rate': float(misclassification_rate),
        'mae': float(mae),
        'mse': float(mse),
        'nll_uncalibrated': float(nll_uncalibrated),
        'nll_calibrated': float(nll_calibrated),
        'ece_uncalibrated': float(ece_uncalibrated),
        'ece_calibrated': float(ece_calibrated),
        'confusion_matrix': cm.tolist(),
        'predictions': y_pred.tolist(),
        'probabilities_calibrated': probs_calibrated.tolist(),
        'class_names': class_names
    }


def plot_confusion_matrix(cm, class_names, model_name, output_path):
    """Plot and save confusion matrix visualization."""
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    
    plt.title(f'Confusion Matrix - {model_name}\nTest Set (Isotonic Calibrated)', 
              fontsize=14, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved confusion matrix to: {output_path}")


def plot_metrics_comparison(results, output_path):
    """Plot comparison of metrics across all models."""
    if len(results) < 2:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    models = [r['model_name'] for r in results]
    metrics = {
        'Accuracy': [r['accuracy'] for r in results],
        'Misclassification Rate': [r['misclassification_rate'] for r in results],
        'Mean Absolute Error': [r['mae'] for r in results],
        'Mean Square Error': [r['mse'] for r in results]
    }
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(models)))
    
    for idx, (metric_name, values) in enumerate(metrics.items()):
        ax = axes[idx // 2, idx % 2]
        bars = ax.bar(range(len(models)), values, color=colors)
        ax.set_ylabel(metric_name, fontsize=11)
        ax.set_title(f'{metric_name} Comparison', fontsize=12)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right')
        
        # Set appropriate y-axis limits based on metric
        if 'Accuracy' in metric_name or 'Misclassification' in metric_name:
            ax.set_ylim([0, 1.0])
        else:  # MAE, MSE
            ax.set_ylim([0, max(values) * 1.2])
        
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, values)):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, 
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('Model Performance Comparison - Test Set (Isotonic Calibrated)', 
                 fontsize=14, y=1.00)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved metrics comparison to: {output_path}")


def plot_calibration_improvement(results, output_path):
    """Plot calibration improvement across models."""
    if not results:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    models = [r['model_name'] for r in results]
    nll_before = [r['nll_uncalibrated'] for r in results]
    nll_after = [r['nll_calibrated'] for r in results]
    ece_before = [r['ece_uncalibrated'] for r in results]
    ece_after = [r['ece_calibrated'] for r in results]
    
    x = np.arange(len(models))
    width = 0.35
    
    # NLL comparison
    bars1 = ax1.bar(x - width/2, nll_before, width, label='Uncalibrated', 
                    color='lightcoral', alpha=0.8)
    bars2 = ax1.bar(x + width/2, nll_after, width, label='Isotonic Calibrated', 
                    color='skyblue', alpha=0.8)
    ax1.set_ylabel('Negative Log-Likelihood', fontsize=11)
    ax1.set_title('NLL Comparison', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # ECE comparison
    bars3 = ax2.bar(x - width/2, ece_before, width, label='Uncalibrated', 
                    color='lightcoral', alpha=0.8)
    bars4 = ax2.bar(x + width/2, ece_after, width, label='Isotonic Calibrated', 
                    color='skyblue', alpha=0.8)
    ax2.set_ylabel('Expected Calibration Error', fontsize=11)
    ax2.set_title('ECE Comparison', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Calibration Improvement - Test Set (Isotonic)', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved calibration improvement plot to: {output_path}")


def main():
    print("="*60)
    print("MODEL EVALUATION AFTER ISOTONIC CALIBRATION - TEST SET")
    print("="*60)
    
    # Load test data
    print("\nLoading test data...")
    features_data = joblib.load('features/combined/combined_test.joblib')
    labels_data = joblib.load('features/combined/labels_test.joblib')
    
    X_test = np.asarray(features_data.get('features', features_data))
    y_test = np.asarray(labels_data.get('labels', labels_data))
    
    # Load class names
    encoder_info = joblib.load('features/combined/ordinal_encoder_info.joblib')
    class_names = encoder_info['class_names']
    
    print(f"Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    print(f"Classes: {class_names}")
    
    # Load isotonic calibrators
    calibrators_path = Path('evaluation_results/isotonic_calibrators.joblib')
    if not calibrators_path.exists():
        print(f"\nError: Isotonic calibrators not found at {calibrators_path}")
        print("Please run 08_isotonic_calibration.py first to generate calibrators.")
        return
    
    print(f"\nLoading isotonic calibrators from: {calibrators_path}")
    calibrators = joblib.load(calibrators_path)
    
    # Models to evaluate (must match the names in calibrators)
    models_to_evaluate = [
        ('features/model_rf_mord.joblib', 'RF'),
        ('features/model_svm_ordinal.joblib', 'SVM'),
        ('features/model_gb_lightgbm.joblib', 'GB'),
    ]
    
    results = []
    predictions_storage = {}
    
    # Create visualization output directory
    viz_dir = Path('visualizations/confusion_matrices/isotonic')
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Evaluate each model with isotonic calibration
    for model_path, model_name in tqdm(models_to_evaluate, desc='Evaluating models'):
        model = load_model(model_path)
        
        if model is None:
            print(f"\n{'='*60}")
            print(f"Model not found: {model_name}")
            print(f"Path: {model_path}")
            print('='*60)
            continue
        
        # Check if calibrator exists for this model
        if model_name not in calibrators:
            print(f"\n{'='*60}")
            print(f"Calibrator not found for: {model_name}")
            print('='*60)
            continue
        
        iso_calibrator = calibrators[model_name]
        
        result = evaluate_model_isotonic(
            model, iso_calibrator, X_test, y_test, class_names, model_name
        )
        results.append(result)
        
        # Plot confusion matrix
        cm = np.array(result['confusion_matrix'])
        safe_name = model_name.lower().replace(' ', '_').replace('(', '').replace(')', '')
        cm_path = viz_dir / f'{safe_name}_confusion_matrix_isotonic.png'
        plot_confusion_matrix(cm, class_names, model_name, cm_path)
        
        # Store predictions for further analysis
        predictions_storage[model_name] = {
            'predictions': result['predictions'],
            'probabilities': result['probabilities_calibrated'],
            'true_labels': y_test.tolist()
        }
    
    # Summary comparison
    if results:
        print("\n" + "="*60)
        print("SUMMARY - TEST SET PERFORMANCE (ISOTONIC CALIBRATED)")
        print("="*60)
        
        print(f"\n{'Model':<30} {'Accuracy':<12} {'Misclass.':<12} {'MAE':<10} {'MSE':<10}")
        print("-" * 74)
        
        # Sort by accuracy (descending - higher is better)
        sorted_results = sorted(results, key=lambda x: x['accuracy'], reverse=True)
        for r in sorted_results:
            print(f"{r['model_name']:<30} {r['accuracy']:<12.4f} "
                  f"{r['misclassification_rate']:<12.4f} {r['mae']:<10.4f} {r['mse']:<10.4f}")
        
        print("\n" + "="*60)
        print("CALIBRATION IMPROVEMENT SUMMARY")
        print("="*60)
        
        print(f"\n{'Model':<30} {'NLL Improve':<12} {'ECE Improve':<12}")
        print("-" * 60)
        
        for r in sorted_results:
            nll_improve = r['nll_uncalibrated'] - r['nll_calibrated']
            ece_improve = r['ece_uncalibrated'] - r['ece_calibrated']
            print(f"{r['model_name']:<30} {nll_improve:<12.4f} {ece_improve:<12.4f}")
        
        print("\n" + "="*60)
        print(f"Best Model (highest accuracy): {sorted_results[0]['model_name']}")
        print(f"Accuracy: {sorted_results[0]['accuracy']:.4f} ({sorted_results[0]['accuracy']*100:.2f}%)")
        print(f"Misclassification Rate: {sorted_results[0]['misclassification_rate']:.4f}")
        print(f"MAE: {sorted_results[0]['mae']:.4f}")
        print(f"MSE: {sorted_results[0]['mse']:.4f}")
        print(f"NLL (Calibrated): {sorted_results[0]['nll_calibrated']:.4f}")
        print(f"ECE (Calibrated): {sorted_results[0]['ece_calibrated']:.4f}")
        print("="*60)
        
        # Save results
        output_dir = Path('evaluation_results')
        output_dir.mkdir(exist_ok=True)
        
        # Save metrics
        metrics_path = output_dir / 'test_metrics_isotonic.json'
        print(f"\nSaving metrics to: {metrics_path}")
        with open(metrics_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save predictions
        predictions_path = output_dir / 'test_predictions_isotonic.joblib'
        print(f"Saving predictions to: {predictions_path}")
        joblib.dump(predictions_storage, predictions_path)
        
        # Plot visualizations
        print("\n" + "-"*60)
        print("Creating visualizations...")
        print("-"*60)
        
        # Metrics comparison
        comparison_path = viz_dir / 'metrics_comparison_isotonic.png'
        plot_metrics_comparison(results, comparison_path)
        
        # Calibration improvement
        calibration_path = viz_dir / 'calibration_improvement_isotonic.png'
        plot_calibration_improvement(results, calibration_path)
        
        print("\n" + "="*60)
        print("ISOTONIC CALIBRATED EVALUATION COMPLETED")
        print("="*60)
    else:
        print("\nNo models found to evaluate!")


if __name__ == '__main__':
    main()
