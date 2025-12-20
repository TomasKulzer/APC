"""
Isotonic Regression for Model Calibration

This module implements isotonic regression to calibrate model confidence/probability estimates.
Isotonic regression fits a non-parametric, piecewise constant, monotonically increasing
function to convert uncalibrated scores to calibrated probabilities.

Theory:
- Non-parametric calibration method
- Learns a monotonically increasing mapping from scores to probabilities
- More flexible than Platt scaling, less prone to overfitting than complex models
- Works well for binary and multiclass classification (one-vs-rest)

The isotonic function is learned by solving a constrained optimization problem
on the validation set.
"""

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, accuracy_score
from scipy.special import softmax


class IsotonicCalibrator:
    """
    Isotonic regression calibrator.
    
    Fits isotonic regression on model probabilities to calibrate them.
    Uses one-vs-rest approach for multiclass problems.
    """
    
    def __init__(self, n_classes=5):
        """
        Initialize isotonic calibrator.
        
        Args:
            n_classes: Number of classes
        """
        self.n_classes = n_classes
        self.calibrators = []
    
    def fit(self, probs, labels):
        """
        Fit isotonic regression parameters on validation data.
        
        Args:
            probs: Uncalibrated probabilities (n_samples, n_classes)
            labels: True labels (n_samples,)
        
        Returns:
            self
        """
        self.calibrators = []
        
        # Fit one isotonic regressor per class (one-vs-rest)
        for class_idx in range(self.n_classes):
            # Create binary labels for this class
            binary_labels = (labels == class_idx).astype(int)
            
            # Get probabilities for this class
            class_probs = probs[:, class_idx]
            
            # Fit isotonic regression
            iso = IsotonicRegression(out_of_bounds='clip')
            iso.fit(class_probs, binary_labels)
            
            self.calibrators.append(iso)
        
        return self
    
    def predict_proba(self, probs):
        """
        Apply isotonic calibration to get calibrated probabilities.
        
        Args:
            probs: Uncalibrated probabilities (n_samples, n_classes)
        
        Returns:
            Calibrated probabilities (n_samples, n_classes)
        """
        if not self.calibrators:
            raise ValueError("Isotonic calibrator must be fitted first")
        
        n_samples = probs.shape[0]
        calibrated_probs = np.zeros((n_samples, self.n_classes))
        
        # Apply each calibrator
        for class_idx, iso in enumerate(self.calibrators):
            calibrated_probs[:, class_idx] = iso.predict(probs[:, class_idx])
        
        # Normalize to ensure probabilities sum to 1
        row_sums = calibrated_probs.sum(axis=1, keepdims=True)
        calibrated_probs = calibrated_probs / (row_sums + 1e-10)
        
        return calibrated_probs


def isotonic_calibrate(probs, labels):
    """
    Fit isotonic calibration and return calibrator.
    
    Args:
        probs: Uncalibrated probabilities (n_samples, n_classes)
        labels: True labels (n_samples,)
    
    Returns:
        IsotonicCalibrator fitted on the data
    """
    n_classes = probs.shape[1]
    calibrator = IsotonicCalibrator(n_classes=n_classes)
    calibrator.fit(probs, labels)
    return calibrator


def calibrate_model_isotonic(model, X_val, y_val, get_logits_func):
    """
    Calibrate a model using isotonic regression on validation set.
    
    Args:
        model: Trained model
        X_val: Validation features
        y_val: Validation labels
        get_logits_func: Function to extract logits from model
    
    Returns:
        dict with isotonic calibrator and calibration info
    """
    # Get logits from model
    logits = get_logits_func(model, X_val)
    
    # Get uncalibrated probabilities
    probs_before = softmax(logits, axis=1)
    
    # Fit isotonic calibrator
    iso_calibrator = isotonic_calibrate(probs_before, y_val)
    
    # Get calibrated probabilities
    probs_after = iso_calibrator.predict_proba(probs_before)
    
    # Compute metrics
    nll_before = log_loss(y_val, probs_before)
    nll_after = log_loss(y_val, probs_after)
    
    # Predictions don't change with calibration
    y_pred = np.argmax(logits, axis=1)
    accuracy = accuracy_score(y_val, y_pred)
    
    return {
        'iso_calibrator': iso_calibrator,
        'nll_before': float(nll_before),
        'nll_after': float(nll_after),
        'accuracy': float(accuracy),
        'logits': logits,
        'probs_before': probs_before,
        'probs_after': probs_after
    }


def apply_isotonic_calibration(model, X, iso_calibrator, get_logits_func):
    """
    Apply fitted isotonic calibration to get calibrated probabilities.
    
    Args:
        model: Trained model
        X: Input features
        iso_calibrator: Fitted IsotonicCalibrator object
        get_logits_func: Function to extract logits from model
    
    Returns:
        Calibrated probabilities (n_samples, n_classes)
    """
    logits = get_logits_func(model, X)
    probs = softmax(logits, axis=1)
    return iso_calibrator.predict_proba(probs)
