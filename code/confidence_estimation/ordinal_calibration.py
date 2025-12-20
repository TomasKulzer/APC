"""
Ordinal-Aware Calibration for Ordinal Classification

This module implements ordinal-aware calibration that respects the natural ordering
of classes in ordinal classification problems. It ensures that cumulative probabilities
are monotonically increasing, which is a key requirement for ordinal predictions.

Theory:
- For ordinal classes: Class 0 < Class 1 < Class 2 < ... < Class K
- Cumulative probabilities must satisfy: P(Y ≤ k) ≤ P(Y ≤ k+1) for all k
- Individual class probabilities: P(Y = k) = P(Y ≤ k) - P(Y ≤ k-1)
- Uses isotonic regression to enforce monotonicity on cumulative distributions

This approach is specifically designed for ordinal problems like ripeness classification
where the classes have a natural ordering that should be respected in calibration.
"""

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, accuracy_score
from scipy.special import softmax


def probs_to_cumulative(probs):
    """
    Convert class probabilities to cumulative probabilities.
    
    Args:
        probs: Class probabilities (n_samples, n_classes)
    
    Returns:
        Cumulative probabilities (n_samples, n_classes)
        cum_probs[:, k] = P(Y ≤ k)
    """
    return np.cumsum(probs, axis=1)


def cumulative_to_probs(cum_probs):
    """
    Convert cumulative probabilities to class probabilities.
    
    Args:
        cum_probs: Cumulative probabilities (n_samples, n_classes)
    
    Returns:
        Class probabilities (n_samples, n_classes)
        probs[:, k] = P(Y = k)
    """
    n_samples, n_classes = cum_probs.shape
    probs = np.zeros_like(cum_probs)
    
    # P(Y = 0) = P(Y ≤ 0)
    probs[:, 0] = cum_probs[:, 0]
    
    # P(Y = k) = P(Y ≤ k) - P(Y ≤ k-1) for k > 0
    for k in range(1, n_classes):
        probs[:, k] = cum_probs[:, k] - cum_probs[:, k-1]
    
    # Ensure non-negative and normalize
    probs = np.maximum(probs, 0)
    row_sums = probs.sum(axis=1, keepdims=True)
    probs = probs / (row_sums + 1e-10)
    
    return probs


def enforce_monotonicity(cum_probs):
    """
    Enforce monotonicity constraint on cumulative probabilities.
    
    Ensures P(Y ≤ k) ≤ P(Y ≤ k+1) for all k.
    
    Args:
        cum_probs: Cumulative probabilities (n_samples, n_classes)
    
    Returns:
        Monotonic cumulative probabilities
    """
    n_samples, n_classes = cum_probs.shape
    monotonic = np.copy(cum_probs)
    
    # Ensure monotonicity: if cum_probs[k] > cum_probs[k+1], set both to average
    for k in range(n_classes - 1):
        # If not monotonic, use the maximum of current and previous
        monotonic[:, k+1] = np.maximum(monotonic[:, k], monotonic[:, k+1])
    
    # Ensure last column is 1.0 (total probability)
    monotonic[:, -1] = 1.0
    
    return monotonic


class OrdinalCalibrator:
    """
    Ordinal-aware calibrator using isotonic regression on cumulative probabilities.
    
    Ensures that calibrated probabilities respect the ordinal structure:
    P(Y ≤ k) ≤ P(Y ≤ k+1) for all k.
    """
    
    def __init__(self, n_classes=5):
        """
        Initialize ordinal calibrator.
        
        Args:
            n_classes: Number of ordinal classes
        """
        self.n_classes = n_classes
        self.calibrators = []
    
    def fit(self, probs, labels):
        """
        Fit ordinal calibration on validation data.
        
        Args:
            probs: Uncalibrated class probabilities (n_samples, n_classes)
            labels: True ordinal labels (n_samples,)
        
        Returns:
            self
        """
        # Convert to cumulative probabilities
        cum_probs = probs_to_cumulative(probs)
        
        # Create binary cumulative labels: y_cum[k] = 1 if y ≤ k else 0
        n_samples = len(labels)
        cum_labels = np.zeros((n_samples, self.n_classes))
        for i, label in enumerate(labels):
            # All classes <= label get 1
            cum_labels[i, :label+1] = 1
            # All classes > label get 0 (already 0 by default)
        
        # Fit isotonic regression for each cumulative threshold
        self.calibrators = []
        for k in range(self.n_classes):
            iso = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1)
            iso.fit(cum_probs[:, k], cum_labels[:, k])
            self.calibrators.append(iso)
        
        return self
    
    def predict_proba(self, probs):
        """
        Apply ordinal calibration to get calibrated probabilities.
        
        Args:
            probs: Uncalibrated class probabilities (n_samples, n_classes)
        
        Returns:
            Calibrated class probabilities (n_samples, n_classes)
        """
        if not self.calibrators:
            raise ValueError("Ordinal calibrator must be fitted first")
        
        # Convert to cumulative probabilities
        cum_probs = probs_to_cumulative(probs)
        
        n_samples = cum_probs.shape[0]
        calibrated_cum = np.zeros((n_samples, self.n_classes))
        
        # Apply isotonic regression to each cumulative probability
        for k, iso in enumerate(self.calibrators):
            calibrated_cum[:, k] = iso.predict(cum_probs[:, k])
        
        # Enforce monotonicity constraint
        calibrated_cum = enforce_monotonicity(calibrated_cum)
        
        # Convert back to class probabilities
        calibrated_probs = cumulative_to_probs(calibrated_cum)
        
        return calibrated_probs


def ordinal_calibrate(probs, labels):
    """
    Fit ordinal-aware calibration and return calibrator.
    
    Args:
        probs: Uncalibrated class probabilities (n_samples, n_classes)
        labels: True ordinal labels (n_samples,)
    
    Returns:
        OrdinalCalibrator fitted on the data
    """
    n_classes = probs.shape[1]
    calibrator = OrdinalCalibrator(n_classes=n_classes)
    calibrator.fit(probs, labels)
    return calibrator


def calibrate_model_ordinal(model, X_val, y_val, get_logits_func):
    """
    Calibrate a model using ordinal-aware calibration on validation set.
    
    Args:
        model: Trained model
        X_val: Validation features
        y_val: Validation ordinal labels
        get_logits_func: Function to extract logits from model
    
    Returns:
        dict with ordinal calibrator and calibration info
    """
    # Get logits from model
    logits = get_logits_func(model, X_val)
    
    # Get uncalibrated probabilities
    probs_before = softmax(logits, axis=1)
    
    # Fit ordinal calibrator
    ord_calibrator = ordinal_calibrate(probs_before, y_val)
    
    # Get calibrated probabilities
    probs_after = ord_calibrator.predict_proba(probs_before)
    
    # Compute metrics
    nll_before = log_loss(y_val, probs_before)
    nll_after = log_loss(y_val, probs_after)
    
    # Predictions don't change with calibration
    y_pred = np.argmax(logits, axis=1)
    accuracy = accuracy_score(y_val, y_pred)
    
    return {
        'ord_calibrator': ord_calibrator,
        'nll_before': float(nll_before),
        'nll_after': float(nll_after),
        'accuracy': float(accuracy),
        'logits': logits,
        'probs_before': probs_before,
        'probs_after': probs_after
    }


def apply_ordinal_calibration(model, X, ord_calibrator, get_logits_func):
    """
    Apply fitted ordinal calibration to get calibrated probabilities.
    
    Args:
        model: Trained model
        X: Input features
        ord_calibrator: Fitted OrdinalCalibrator object
        get_logits_func: Function to extract logits from model
    
    Returns:
        Calibrated probabilities (n_samples, n_classes)
    """
    logits = get_logits_func(model, X)
    probs = softmax(logits, axis=1)
    return ord_calibrator.predict_proba(probs)


def verify_ordinal_constraints(probs):
    """
    Verify that probabilities satisfy ordinal constraints.
    
    Checks that cumulative probabilities are monotonically increasing.
    
    Args:
        probs: Class probabilities (n_samples, n_classes)
    
    Returns:
        bool: True if all constraints are satisfied
    """
    cum_probs = probs_to_cumulative(probs)
    
    # Check monotonicity
    for k in range(cum_probs.shape[1] - 1):
        if np.any(cum_probs[:, k] > cum_probs[:, k+1] + 1e-6):
            return False
    
    return True
