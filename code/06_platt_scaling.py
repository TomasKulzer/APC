"""
Apply Platt Scaling Calibration to Models

This script calibrates trained models using Platt scaling on the validation set
and evaluates the calibration on the test set.

Platt scaling fits a logistic regression on model outputs to calibrate probabilities.
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
from confidence_estimation.platt_scaling import (
    calibrate_model_platt,
    apply_platt_scaling
)


def main():
    print("="*60)
    print("PLATT SCALING CALIBRATION")
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
    
    # Models to calibrate
    models_to_calibrate = [
        ('features/model_rf.joblib', 'RF'),
        ('features/model_svm.joblib', 'SVM'),
        ('features/model_gb.joblib', 'GB'),
    ]
    
    results = {}
    viz_dir = Path('visualizations/calibration/platt_scaling')
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Store calibrators for later use
    calibrators = {}
    
    for model_path, model_name in models_to_calibrate:
        if not Path(model_path).exists():
            print(f"\n{'='*60}")
            print(f"Model not found: {model_name}")
            print('='*60)
            continue
        
        print(f"\n{'='*60}")
        print(f"Calibrating: {model_name}")
        print('='*60)
        
        # Load model
        model_data = joblib.load(model_path)
        model = model_data.best_estimator_ if hasattr(model_data, 'best_estimator_') else model_data
        
        # Calibrate on validation set
        print("\n--- Validation Set Calibration ---")
        val_results = calibrate_model_platt(model, X_val, y_val, get_logits_from_model)
        
        print(f"\nPlatt Scaling Parameters Fitted")
        print(f"NLL Before: {val_results['nll_before']:.4f}")
        print(f"NLL After:  {val_results['nll_after']:.4f}")
        print(f"Improvement: {val_results['nll_before'] - val_results['nll_after']:.4f}")
        print(f"Accuracy: {val_results['accuracy']:.4f}")
        
        # Compute ECE
        ece_before = evaluate_calibration(val_results['probs_before'], y_val)
        ece_after = evaluate_calibration(val_results['probs_after'], y_val)
        print(f"ECE Before: {ece_before:.4f}")
        print(f"ECE After:  {ece_after:.4f}")
        
        # Plot reliability diagram for validation
        safe_name = model_name.lower().replace(' ', '_')
        plot_path = viz_dir / f'{safe_name}_validation_reliability.png'
        plot_reliability_diagram(
            val_results['probs_before'],
            val_results['probs_after'],
            y_val,
            f"{model_name} (Validation - Platt Scaling)",
            plot_path
        )
        print(f"Saved reliability diagram to: {plot_path}")
        
        # Store calibrator
        calibrators[model_name] = val_results['platt_scaler']
        
        # Test on test set
        print("\n--- Test Set Evaluation ---")
        test_logits = get_logits_from_model(model, X_test)
        test_probs_before = softmax(test_logits, axis=1)
        test_probs_after = val_results['platt_scaler'].predict_proba(test_logits)
        
        test_nll_before = log_loss(y_test, test_probs_before)
        test_nll_after = log_loss(y_test, test_probs_after)
        test_ece_before = evaluate_calibration(test_probs_before, y_test)
        test_ece_after = evaluate_calibration(test_probs_after, y_test)
        
        y_pred = np.argmax(test_logits, axis=1)
        test_accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test NLL Before: {test_nll_before:.4f}")
        print(f"Test NLL After:  {test_nll_after:.4f}")
        print(f"Test ECE Before: {test_ece_before:.4f}")
        print(f"Test ECE After:  {test_ece_after:.4f}")
        
        # Plot reliability diagram for test
        plot_path_test = viz_dir / f'{safe_name}_test_reliability.png'
        plot_reliability_diagram(
            test_probs_before,
            test_probs_after,
            y_test,
            f"{model_name} (Test - Platt Scaling)",
            plot_path_test
        )
        print(f"Saved test reliability diagram to: {plot_path_test}")
        
        # Store results
        results[model_name] = {
            'validation': {
                'nll_before': val_results['nll_before'],
                'nll_after': val_results['nll_after'],
                'ece_before': float(ece_before),
                'ece_after': float(ece_after),
                'accuracy': val_results['accuracy']
            },
            'test': {
                'nll_before': float(test_nll_before),
                'nll_after': float(test_nll_after),
                'ece_before': float(test_ece_before),
                'ece_after': float(test_ece_after),
                'accuracy': float(test_accuracy)
            }
        }
    
    # Save results
    output_dir = Path('evaluation_results')
    output_dir.mkdir(exist_ok=True)
    
    results_path = output_dir / 'platt_scaling_results.json'
    print(f"\n{'='*60}")
    print(f"Saving results to: {results_path}")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save calibrators
    calibrators_path = output_dir / 'platt_scalers.joblib'
    print(f"Saving Platt scalers to: {calibrators_path}")
    joblib.dump(calibrators, calibrators_path)
    
    # Summary
    print("\n" + "="*60)
    print("PLATT SCALING SUMMARY")
    print("="*60)
    
    print(f"\n{'Model':<30} {'Test NLL':<15} {'Test ECE':<15}")
    print(f"{'':30} {'Before→After':15} {'Before→After':15}")
    print("-" * 75)
    
    for model_name, res in results.items():
        nll_change = f"{res['test']['nll_before']:.3f}→{res['test']['nll_after']:.3f}"
        ece_change = f"{res['test']['ece_before']:.3f}→{res['test']['ece_after']:.3f}"
        print(f"{model_name:<30} {nll_change:<15} {ece_change:<15}")
    
    print("\n" + "="*60)
    print("PLATT SCALING COMPLETED")
    print("="*60)


if __name__ == '__main__':
    main()
