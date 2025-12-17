"""
Evaluate Model Performance on Validation Set

Computes comprehensive metrics for all trained models including:
- Accuracy, precision, recall, F1-score
- Confusion matrices
- Per-class metrics
- Saves predictions for later calibration
"""

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
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
    precision = precision_score(y, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y, y_pred, average='weighted', zero_division=0)
    
    # Per-class metrics
    precision_per_class = precision_score(y, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y, y_pred, average=None, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    
    # Print results
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    print(f"\nPer-Class Metrics:")
    print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 51)
    for i, class_name in enumerate(class_names):
        print(f"{class_name:<15} {precision_per_class[i]:<12.4f} "
              f"{recall_per_class[i]:<12.4f} {f1_per_class[i]:<12.4f}")
    
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
    
    print(f"\nClassification Report:")
    print(classification_report(y, y_pred, target_names=class_names))
    
    # Return results
    return {
        'model_name': model_name,
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'precision_per_class': precision_per_class.tolist(),
        'recall_per_class': recall_per_class.tolist(),
        'f1_per_class': f1_per_class.tolist(),
        'confusion_matrix': cm.tolist(),
        'predictions': y_pred.tolist(),
        'class_names': class_names
    }


def plot_confusion_matrix(cm, class_names, model_name, output_path):
    """Plot and save confusion matrix visualization."""
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    
    plt.title(f'Confusion Matrix - {model_name}\nValidation Set', fontsize=14, pad=20)
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
        'Precision': [r['precision'] for r in results],
        'Recall': [r['recall'] for r in results],
        'F1-Score': [r['f1_score'] for r in results]
    }
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(models)))
    
    for idx, (metric_name, values) in enumerate(metrics.items()):
        ax = axes[idx // 2, idx % 2]
        bars = ax.bar(range(len(models)), values, color=colors)
        ax.set_ylabel(metric_name, fontsize=11)
        ax.set_title(f'{metric_name} Comparison', fontsize=12)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_ylim([0, 1.0])
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
    
    # Models to evaluate (include integer SVM and Random Forest for comparison)
    models_to_evaluate = [
        ('features/model_svm.joblib', 'Integer SVM'),
        ('features/model_rf.joblib', 'Integer Random Forest'),
        ('features/model_svm_ordinal.joblib', 'Ordinal SVM'),
        ('features/model_gb_ordinal.joblib', 'Ordinal Gradient Boosting'),
        ('features/model_rf_mord.joblib', 'Ordinal RF (HGB Monotonic)'),
    ]
    
    results = []
    predictions_storage = {}
    
    # Create visualization output directory
    viz_dir = Path('visualizations/confusion_matrices')
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
        
        # Plot confusion matrix
        cm = np.array(result['confusion_matrix'])
        safe_name = model_name.lower().replace(' ', '_')
        cm_path = viz_dir / f'{safe_name}_confusion_matrix.png'
        plot_confusion_matrix(cm, class_names, model_name, cm_path)
        
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
        
        print(f"\n{'Model':<30} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
        print("-" * 70)
        
        # Sort by accuracy
        sorted_results = sorted(results, key=lambda x: x['accuracy'], reverse=True)
        for r in sorted_results:
            print(f"{r['model_name']:<30} {r['accuracy']:<10.4f} "
                  f"{r['precision']:<10.4f} {r['recall']:<10.4f} {r['f1_score']:<10.4f}")
        
        print("\n" + "="*60)
        print(f"Best Model: {sorted_results[0]['model_name']}")
        print(f"Best Accuracy: {sorted_results[0]['accuracy']:.4f} ({sorted_results[0]['accuracy']*100:.2f}%)")
        print("="*60)
        
        # Save results
        output_dir = Path('evaluation_results')
        output_dir.mkdir(exist_ok=True)
        
        # Save metrics
        metrics_path = output_dir / 'validation_metrics.json'
        print(f"\nSaving metrics to: {metrics_path}")
        with open(metrics_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save predictions for calibration
        predictions_path = output_dir / 'validation_predictions.joblib'
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
