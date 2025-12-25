"""
Generate Baseline (Uncalibrated) Reliability Diagrams

This script generates reliability diagrams for uncalibrated model predictions
using softmax probabilities. This serves as a baseline for comparison with
calibrated models.
"""

import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import joblib
from sklearn.metrics import log_loss, accuracy_score
from pathlib import Path
import json
from scipy.special import softmax

from confidence_estimation import (
    get_logits_from_model,
    evaluate_calibration,
    plot_reliability_diagram
)


def main():
    print("="*60)
    print("UNCALIBRATED SOFTMAX BASELINE")
    print("="*60)
    
    # Load validation data
    print("\nLoading validation data...")
    X_val = joblib.load('features/combined/combined_val.joblib')
    y_val = joblib.load('features/combined/labels_val.joblib')
    
    X_val = np.asarray(X_val.get('features', X_val))
    y_val = np.asarray(y_val.get('labels', y_val))
    
    print(f"Validation set: {X_val.shape[0]} samples")
    
    # Load test data
    print("Loading test data...")
    X_test = joblib.load('features/combined/combined_test.joblib')
    y_test = joblib.load('features/combined/labels_test.joblib')
    
    X_test = np.asarray(X_test.get('features', X_test))
    y_test = np.asarray(y_test.get('labels', y_test))
    
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Load class names
    encoder_info = joblib.load('features/combined/ordinal_encoder_info.joblib')
    class_names = encoder_info['class_names']
    
    # Models to evaluate
    models_to_evaluate = [
        ('features/model_rf.joblib', 'RF'),
        ('features/model_svm.joblib', 'SVM'),
        ('features/model_gb.joblib', 'GB'),
    ]
    
    results = {}
    viz_dir = Path('visualizations/calibration/uncalibrated')
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    for model_path, model_name in models_to_evaluate:
        if not Path(model_path).exists():
            print(f"\n{'='*60}")
            print(f"Model not found: {model_name}")
            print('='*60)
            continue
        
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print('='*60)
        
        # Load model
        model_data = joblib.load(model_path)
        model = model_data.best_estimator_ if hasattr(model_data, 'best_estimator_') else model_data
        
        # Validation set evaluation
        print("\n--- Validation Set ---")
        val_logits = get_logits_from_model(model, X_val)
        val_probs = softmax(val_logits, axis=1)
        
        val_nll = log_loss(y_val, val_probs)
        val_ece = evaluate_calibration(val_probs, y_val)
        y_val_pred = np.argmax(val_logits, axis=1)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        
        print(f"Accuracy: {val_accuracy:.4f}")
        print(f"NLL: {val_nll:.4f}")
        print(f"ECE: {val_ece:.4f}")
        
        # Plot reliability diagram for validation (uncalibrated only)
        safe_name = model_name.lower().replace(' ', '_')
        plot_path = viz_dir / f'{safe_name}_validation_reliability.png'
        plot_reliability_diagram(
            None,
            val_probs,
            y_val,
            f"{model_name} (Validation - Uncalibrated)",
            plot_path,
            n_bins=15
        )
        print(f"Saved reliability diagram to: {plot_path}")
        
        # Test set evaluation
        print("\n--- Test Set ---")
        test_logits = get_logits_from_model(model, X_test)
        test_probs = softmax(test_logits, axis=1)
        
        test_nll = log_loss(y_test, test_probs)
        test_ece = evaluate_calibration(test_probs, y_test)
        y_test_pred = np.argmax(test_logits, axis=1)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        
        print(f"Accuracy: {test_accuracy:.4f}")
        print(f"NLL: {test_nll:.4f}")
        print(f"ECE: {test_ece:.4f}")
        
        # Plot reliability diagram for test (uncalibrated only)
        plot_path_test = viz_dir / f'{safe_name}_test_reliability.png'
        plot_reliability_diagram(
            None,
            test_probs,
            y_test,
            f"{model_name} (Test - Uncalibrated)",
            plot_path_test,
            n_bins=15
        )
        print(f"Saved test reliability diagram to: {plot_path_test}")
        
        # Store results
        results[model_name] = {
            'validation': {
                'nll': float(val_nll),
                'ece': float(val_ece),
                'accuracy': float(val_accuracy)
            },
            'test': {
                'nll': float(test_nll),
                'ece': float(test_ece),
                'accuracy': float(test_accuracy)
            }
        }
    
    # Save results
    output_dir = Path('evaluation_results')
    output_dir.mkdir(exist_ok=True)
    
    results_path = output_dir / 'uncalibrated_softmax_results.json'
    print(f"\n{'='*60}")
    print(f"Saving results to: {results_path}")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Summary
    print("\n" + "="*60)
    print("UNCALIBRATED BASELINE SUMMARY")
    print("="*60)
    
    print(f"\n{'Model':<30} {'Test NLL':<15} {'Test ECE':<15} {'Test Accuracy':<15}")
    print("-" * 75)
    
    for model_name, res in results.items():
        print(f"{model_name:<30} {res['test']['nll']:<15.4f} {res['test']['ece']:<15.4f} {res['test']['accuracy']:<15.4f}")
    
    print("\n" + "="*60)
    print("BASELINE EVALUATION COMPLETED")
    print("="*60)


if __name__ == '__main__':
    main()
