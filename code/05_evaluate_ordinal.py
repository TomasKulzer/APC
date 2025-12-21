"""
Evaluate Model Performance on Validation Set

Computes essential metrics for all trained models:
- Accuracy
- Misclassification Error Rate
- Mean Absolute Error (MAE)
- Mean Square Error (MSE)
- Saves predictions for later calibration
"""

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score, mean_absolute_error, mean_squared_error
)
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


def load_model(model_path):
    """Load model, handling both Pipeline and SimpleNamespace objects."""
    if not Path(model_path).exists():
        return None
    
    model_data = joblib.load(model_path)
    
    # Handle SimpleNamespace from training scripts
    if hasattr(model_data, 'best_estimator_'):
        return model_data.best_estimator_
    return model_data


def evaluate_model(model, X, y, class_names, model_name):
    """
    Evaluate a model and return comprehensive metrics.
    
    Returns:
        dict with metrics and predictions
    """
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print('='*60)
    
    # Get predictions with progress bar
    print("Making predictions...")
    # Predict in batches with progress bar
    batch_size = 50
    n_samples = X.shape[0]
    predictions = []
    
    for i in tqdm(range(0, n_samples, batch_size), desc=f'  Predicting {model_name}'):
        batch_end = min(i + batch_size, n_samples)
        batch_pred = model.predict(X[i:batch_end])
        predictions.append(batch_pred)
    
    y_pred = np.vstack(predictions) if len(predictions[0].shape) > 1 else np.concatenate(predictions)
    
    # For ordinal models, decode predictions
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        print("Decoding ordinal predictions...")
        y_pred = np.sum(y_pred, axis=1).astype(int)
    
    # Compute metrics
    print("Computing metrics...")
    accuracy = accuracy_score(y, y_pred)
    misclassification_rate = 1.0 - accuracy
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    
    # Print results
    print(f"\nMetrics:")
    print(f"  Accuracy:                  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Misclassification Rate:    {misclassification_rate:.4f} ({misclassification_rate*100:.2f}%)")
    print(f"  Mean Absolute Error (MAE): {mae:.4f}")
    print(f"  Mean Square Error (MSE):   {mse:.4f}")
    
    # Return results
    return {
        'model_name': model_name,
        'accuracy': float(accuracy),
        'misclassification_rate': float(misclassification_rate),
        'mae': float(mae),
        'mse': float(mse),
        'predictions': y_pred.tolist(),
        'class_names': class_names
    }


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
    
    plt.suptitle('Model Performance Comparison - Validation Set', fontsize=14, y=1.00)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved metrics comparison to: {output_path}")


def main():
    print("="*60)
    print("MODEL PERFORMANCE EVALUATION - VALIDATION SET")
    print("="*60)
    
    # Load validation data
    print("\nLoading validation data...")
    features_data = joblib.load('features/combined/combined_val.joblib')
    labels_data = joblib.load('features/combined/labels_val.joblib')
    
    X_val = np.asarray(features_data.get('features', features_data))
    y_val = np.asarray(labels_data.get('labels', labels_data))
    
    # Load class names
    encoder_info = joblib.load('features/combined/ordinal_encoder_info.joblib')
    class_names = encoder_info['class_names']
    
    print(f"Validation set: {X_val.shape[0]} samples, {X_val.shape[1]} features")
    print(f"Classes: {class_names}")
    
    # Models to evaluate
    models_to_evaluate = [
        ('features/model_rf_mord.joblib', 'RF'),
        ('features/model_svm_ordinal.joblib', 'SVM'),
        ('features/model_gb_lightgbm.joblib', 'GB'),

    ]
    
    results = []
    predictions_storage = {}
    
    # Create visualization output directory
    viz_dir = Path('visualizations/metrics_simple')
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Evaluate each model
    for model_path, model_name in tqdm(models_to_evaluate, desc='Evaluating models'):
        model = load_model(model_path)
        
        if model is None:
            print(f"\n{'='*60}")
            print(f"Model not found: {model_name}")
            print(f"Path: {model_path}")
            print('='*60)
            continue
        
        result = evaluate_model(model, X_val, y_val, class_names, model_name)
        results.append(result)
        
        # Store predictions for calibration
        predictions_storage[model_name] = {
            'predictions': result['predictions'],
            'true_labels': y_val.tolist()
        }
    
    # Summary comparison
    if results:
        print("\n" + "="*60)
        print("SUMMARY - VALIDATION SET PERFORMANCE")
        print("="*60)
        
        print(f"\n{'Model':<30} {'Accuracy':<12} {'Misclass.':<12} {'MAE':<10} {'MSE':<10}")
        print("-" * 74)
        
        # Sort by accuracy (descending - higher is better)
        sorted_results = sorted(results, key=lambda x: x['accuracy'], reverse=True)
        for r in sorted_results:
            print(f"{r['model_name']:<30} {r['accuracy']:<12.4f} "
                  f"{r['misclassification_rate']:<12.4f} {r['mae']:<10.4f} {r['mse']:<10.4f}")
        
        print("\n" + "="*60)
        print(f"Best Model (highest accuracy): {sorted_results[0]['model_name']}")
        print(f"Accuracy: {sorted_results[0]['accuracy']:.4f} ({sorted_results[0]['accuracy']*100:.2f}%)")
        print(f"Misclassification Rate: {sorted_results[0]['misclassification_rate']:.4f}")
        print(f"MAE: {sorted_results[0]['mae']:.4f}")
        print(f"MSE: {sorted_results[0]['mse']:.4f}")
        print("="*60)
        
        # Save results
        output_dir = Path('evaluation_results')
        output_dir.mkdir(exist_ok=True)
        
        # Save metrics
        metrics_path = output_dir / 'validation_metrics_simple.json'
        print(f"\nSaving metrics to: {metrics_path}")
        with open(metrics_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save predictions for calibration
        predictions_path = output_dir / 'validation_predictions_simple.joblib'
        print(f"Saving predictions to: {predictions_path}")
        joblib.dump(predictions_storage, predictions_path)
        
        # Plot metrics comparison
        print("\n" + "-"*60)
        print("Creating visualizations...")
        print("-"*60)
        comparison_path = viz_dir / 'metrics_comparison.png'
        plot_metrics_comparison(results, comparison_path)
        
        print("\n" + "="*60)
        print("EVALUATION COMPLETED")
        print("="*60)
    else:
        print("\nNo models found to evaluate!")


if __name__ == '__main__':
    main()
