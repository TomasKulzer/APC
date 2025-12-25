"""
Apply Temperature Scaling Calibration to Models

This script calibrates trained models using temperature scaling on the validation set
and evaluates the calibration on the test set.
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
    calibrate_model_temperature,
    get_logits_from_model,
    temperature_scaling,
    evaluate_calibration,
    plot_reliability_diagram,
    plot_confidence_histogram
)


def main():
    print("="*60)
    print("TEMPERATURE SCALING CALIBRATION")
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
    viz_dir = Path('visualizations/calibration/temperature_scaling')
    viz_dir.mkdir(parents=True, exist_ok=True)
    
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
        val_results = calibrate_model_temperature(model, X_val, y_val)
        
        print(f"\nOptimal Temperature: {val_results['temperature']:.4f}")
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
            None,
            val_results['probs_after'],
            y_val,
            f"{model_name} (Validation - Temperature Scaling)",
            plot_path,
            n_bins=15
        )
        print(f"Saved reliability diagram to: {plot_path}")
        
        # Plot confidence histogram for validation
        conf_hist_path = viz_dir / f'{safe_name}_validation_confidence.png'
        plot_confidence_histogram(
            val_results['probs_before'],
            val_results['probs_after'],
            y_val,
            f"{model_name} (Validation)",
            conf_hist_path
        )
        print(f"Saved confidence histogram to: {conf_hist_path}")
        
        # Test on test set
        print("\n--- Test Set Evaluation ---")
        test_logits = get_logits_from_model(model, X_test)
        test_probs_before = softmax(test_logits, axis=1)
        test_probs_after = softmax(temperature_scaling(test_logits, val_results['temperature']), axis=1)
        
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
            None,
            test_probs_after,
            y_test,
            f"{model_name} (Test - Temperature Scaling)",
            plot_path_test,
            n_bins=15
        )
        print(f"Saved test reliability diagram to: {plot_path_test}")
        
        # Plot confidence histogram for test
        conf_hist_path_test = viz_dir / f'{safe_name}_test_confidence.png'
        plot_confidence_histogram(
            test_probs_before,
            test_probs_after,
            y_test,
            f"{model_name} (Test)",
            conf_hist_path_test
        )
        print(f"Saved test confidence histogram to: {conf_hist_path_test}")
        
        # Store results
        results[model_name] = {
            'temperature': val_results['temperature'],
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
    
    results_path = output_dir / 'temperature_scaling_results.json'
    print(f"\n{'='*60}")
    print(f"Saving results to: {results_path}")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Summary
    print("\n" + "="*60)
    print("CALIBRATION SUMMARY")
    print("="*60)
    
    print(f"\n{'Model':<25} {'Temperature':<12} {'Test NLL':<12} {'Test ECE':<12}")
    print(f"{'':25} {'':12} {'Before→After':12} {'Before→After':12}")
    print("-" * 73)
    
    for model_name, res in results.items():
        T = res['temperature']
        nll_change = f"{res['test']['nll_before']:.3f}→{res['test']['nll_after']:.3f}"
        ece_change = f"{res['test']['ece_before']:.3f}→{res['test']['ece_after']:.3f}"
        print(f"{model_name:<25} {T:<12.4f} {nll_change:<12} {ece_change:<12}")
    
    print("\n" + "="*60)
    print("TEMPERATURE SCALING COMPLETED")
    print("="*60)


if __name__ == '__main__':
    main()
