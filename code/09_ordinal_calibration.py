"""
Fixed Ordinal-Aware Calibration for YOUR models
Works with LGBMWrapper, RF, SVM from your oil palm pipeline
"""

import sys
import numpy as np
import joblib
from sklearn.metrics import log_loss, accuracy_score
from sklearn.isotonic import IsotonicRegression
from pathlib import Path
import json
import matplotlib.pyplot as plt
from scipy.special import softmax

# Import visualization
sys.path.append('code')
from confidence_estimation.visualization import plot_reliability_diagram

class OrdinalCalibrator:
    """Ordinal-aware calibrator enforcing P(Y≤k) ≤ P(Y≤k+1)"""
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.isotonic_models = [IsotonicRegression(out_of_bounds='clip') for _ in range(n_classes)]
    
    def fit(self, probs, y_true):
        cum_probs = np.cumsum(probs, axis=1)
        for k in range(self.n_classes):
            targets = (y_true <= k).astype(float)
            self.isotonic_models[k].fit(cum_probs[:, k], targets)
    
    def predict_proba(self, probs):
        cum_probs = np.cumsum(probs, axis=1)
        calibrated_cum = np.zeros_like(cum_probs)
        for k in range(self.n_classes):
            calibrated_cum[:, k] = self.isotonic_models[k].predict(cum_probs[:, k])
        
        # Enforce monotonicity: P(Y≤k) ≤ P(Y≤k+1)
        for k in range(self.n_classes - 1):
            calibrated_cum[:, k + 1] = np.maximum(calibrated_cum[:, k], calibrated_cum[:, k + 1])
        
        # Recover class probs: P(Y=k) = P(Y≤k) - P(Y≤k-1)
        class_probs = np.diff(calibrated_cum, axis=1, prepend=0)
        class_probs = np.clip(class_probs, 0, 1)
        
        # Normalize to sum to 1 (with safety for zero sums)
        row_sums = class_probs.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        class_probs = class_probs / row_sums
        
        return class_probs

def evaluate_calibration(probs, y_true, n_bins=10):
    """Expected Calibration Error"""
    y_pred = np.argmax(probs, axis=1)
    conf = np.max(probs, axis=1)
    labels = np.zeros((len(y_true), probs.shape[1]))
    labels[np.arange(len(y_true)), y_true] = 1
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0
    for bin_lower, bin_upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        in_bin = (conf > bin_lower) & (conf <= bin_upper)
        if in_bin.sum() > 0:
            accuracy = np.mean(y_pred[in_bin] == y_true[in_bin])
            avg_conf = conf[in_bin].mean()
            ece += np.abs(avg_conf - accuracy) * in_bin.sum()
    ece /= len(y_true)
    return ece

def verify_ordinal_constraints(probs):
    """Check P(Y≤k) ≤ P(Y≤k+1)"""
    cum_probs = np.cumsum(probs, axis=1)
    return np.all(cum_probs[:, :-1] <= cum_probs[:, 1:])

def main():
    print("="*60)
    print("FIXED ORDINAL CALIBRATION")
    print("="*60)
    
    # Load data (matches your paths)
    X_val = joblib.load('features/combined/combined_val.joblib')
    y_val = joblib.load('features/combined/labels_val.joblib')
    X_test = joblib.load('features/combined/combined_test.joblib')
    y_test = joblib.load('features/combined/labels_test.joblib')
    
    X_val = np.asarray(X_val.get('features', X_val)) if hasattr(X_val, 'get') else np.asarray(X_val)
    y_val = np.asarray(y_val.get('labels', y_val)) if hasattr(y_val, 'get') else np.asarray(y_val)
    X_test = np.asarray(X_test.get('features', X_test)) if hasattr(X_test, 'get') else np.asarray(X_test)
    y_test = np.asarray(y_test.get('labels', y_test)) if hasattr(y_test, 'get') else np.asarray(y_test)
    
    n_classes = len(np.unique(y_val))
    class_names = [f'Class_{i}' for i in range(n_classes)]
    
    models = [
        ('features/model_gb_lightgbm.joblib', 'LightGBM'),
        # Add others if they exist:
        # ('features/model_rf_mord.joblib', 'RF'),
        # ('features/model_svm_ordinal.joblib', 'SVM'),
    ]
    
    results = {}
    viz_dir = Path('visualizations/calibration/ordinal')
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    for model_path, name in models:
        if not Path(model_path).exists():
            print(f"Skipping {name}: {model_path} not found")
            continue
            
        print(f"\n{'='*50}")
        print(f"Calibrating: {name}")
        print('='*50)
        
        # Load YOUR LGBMWrapper model
        model = joblib.load(model_path)
        
        # Get probabilities (works with LGBMWrapper)
        val_probs_before = model.predict_proba(X_val)
        test_probs_before = model.predict_proba(X_test)
        
        # Fit ordinal calibrator
        ord_calib = OrdinalCalibrator(n_classes)
        ord_calib.fit(val_probs_before, y_val)
        val_probs_after = ord_calib.predict_proba(val_probs_before)
        test_probs_after = ord_calib.predict_proba(test_probs_before)
        
        # Metrics
        val_nll_before = log_loss(y_val, val_probs_before)
        val_nll_after = log_loss(y_val, val_probs_after)
        test_nll_before = log_loss(y_test, test_probs_before)
        test_nll_after = log_loss(y_test, test_probs_after)
        
        val_ece_before = evaluate_calibration(val_probs_before, y_val)
        val_ece_after = evaluate_calibration(val_probs_after, y_val)
        test_ece_before = evaluate_calibration(test_probs_before, y_test)
        test_ece_after = evaluate_calibration(test_probs_after, y_test)
        
        val_acc = accuracy_score(y_val, np.argmax(val_probs_before, axis=1))
        test_acc = accuracy_score(y_test, np.argmax(test_probs_before, axis=1))
        
        constraints_ok = verify_ordinal_constraints(test_probs_after)
        
        print(f"Val NLL:  {val_nll_before:.4f} → {val_nll_after:.4f}")
        print(f"Val ECE:  {val_ece_before:.4f} → {val_ece_after:.4f}")
        print(f"Test NLL: {test_nll_before:.4f} → {test_nll_after:.4f}")
        print(f"Test ECE: {test_ece_before:.4f} → {test_ece_after:.4f}")
        print(f"Test Acc: {test_acc:.4f}")
        print(f"Ordinal OK: {'✓' if constraints_ok else '✗'}")
        
        # Plot reliability diagrams
        safe_name = name.lower().replace(' ', '_')
        
        # Validation reliability diagram
        val_plot_path = viz_dir / f'{safe_name}_validation_reliability.png'
        plot_reliability_diagram(
            val_probs_before,
            val_probs_after,
            y_val,
            f"{name} (Validation - Ordinal Calibration)",
            val_plot_path
        )
        print(f"Saved validation reliability diagram to: {val_plot_path}")
        
        # Test reliability diagram
        test_plot_path = viz_dir / f'{safe_name}_test_reliability.png'
        plot_reliability_diagram(
            test_probs_before,
            test_probs_after,
            y_test,
            f"{name} (Test - Ordinal Calibration)",
            test_plot_path
        )
        print(f"Saved test reliability diagram to: {test_plot_path}")
        
        results[name] = {
            'validation': {
                'nll_before': float(val_nll_before),
                'nll_after': float(val_nll_after),
                'ece_before': float(val_ece_before),
                'ece_after': float(val_ece_after),
                'accuracy': float(val_acc)
            },
            'test': {
                'nll_before': float(test_nll_before),
                'nll_after': float(test_nll_after),
                'ece_before': float(test_ece_before),
                'ece_after': float(test_ece_after),
                'accuracy': float(test_acc)
            },
            'constraints_ok': bool(constraints_ok)
        }
    
    # Save results
    Path('evaluation_results').mkdir(exist_ok=True)
    with open('evaluation_results/ordinal_calibration_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: evaluation_results/ordinal_calibration_results.json")
    print(f"✓ Reliability diagrams saved to: {viz_dir}")
    print("\n✓ ORDINAL CALIBRATION COMPLETE!")

if __name__ == '__main__':
    main()
