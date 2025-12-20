"""
Temperature Scaling for Model Calibration

This module implements temperature scaling to calibrate model confidence/probability estimates.
Temperature scaling adjusts the softmax temperature to better align predicted 
probabilities with actual accuracy.

Theory:
- T > 1: Increases uncertainty (softens probabilities)
- T < 1: Increases confidence (sharpens probabilities)  
- T = 1: No adjustment (original model)

The optimal temperature T is found by minimizing negative log-likelihood (NLL)
or another calibration metric on the validation set.
"""

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import log_loss, accuracy_score
from scipy.special import softmax


def temperature_scaling(logits, temperature):
    """
    Apply temperature scaling to logits.
    
    Args:
        logits: Raw model outputs before softmax (n_samples, n_classes)
        temperature: Temperature parameter T
    
    Returns:
        Scaled logits (n_samples, n_classes)
    """
    return logits / temperature


def find_optimal_temperature(logits, labels, method='nll'):
    """
    Find optimal temperature using validation set.
    
    Args:
        logits: Raw model outputs (n_samples, n_classes)
        labels: True labels (n_samples,)
        method: Optimization criterion ('nll' for negative log-likelihood)
    
    Returns:
        Optimal temperature T
    """
    def nll_loss(T):
        """Negative log-likelihood loss after temperature scaling."""
        if T <= 0:
            return np.inf
        
        scaled_logits = temperature_scaling(logits, T)
        probs = softmax(scaled_logits, axis=1)
        
        # Compute negative log-likelihood
        nll = log_loss(labels, probs)
        return nll
    
    # Find optimal temperature in range [0.1, 10]
    result = minimize_scalar(nll_loss, bounds=(0.1, 10.0), method='bounded')
    
    return result.x


def get_logits_from_model(model, X):
    """
    Extract logits (pre-softmax outputs) from a trained model.
    
    Args:
        model: Trained sklearn model (Pipeline or estimator)
        X: Input features
    
    Returns:
        logits: Raw decision function outputs
    """
    # Handle pipeline
    if hasattr(model, 'named_steps'):
        classifier = model.named_steps[list(model.named_steps.keys())[-1]]
    else:
        classifier = model
    
    # Special handling for OrdinalSVMClassifier (threshold-based)
    if hasattr(classifier, 'predict_proba_classes'):
        # This is the threshold-based SVM - use class probabilities
        probs = classifier.predict_proba_classes(X)
        logits = np.log(probs + 1e-10)  # Convert to log-space
        return logits
    
    # Get decision function (logits)
    if hasattr(classifier, 'decision_function'):
        logits = classifier.decision_function(X)
    elif hasattr(classifier, 'predict_proba'):
        # If only predict_proba available, use log probabilities as proxy
        probs = classifier.predict_proba(X)
        logits = np.log(probs + 1e-10)  # Add small epsilon to avoid log(0)
    else:
        raise ValueError("Model does not support decision_function or predict_proba")
    
    return logits


def calibrate_model_temperature(model, X_val, y_val):
    """
    Calibrate a model using temperature scaling on validation set.
    
    Args:
        model: Trained model
        X_val: Validation features
        y_val: Validation labels
    
    Returns:
        dict with temperature and calibration info
    """
    logits = get_logits_from_model(model, X_val)
    optimal_T = find_optimal_temperature(logits, y_val)
    
    # Compute metrics before and after calibration
    probs_before = softmax(logits, axis=1)
    probs_after = softmax(temperature_scaling(logits, optimal_T), axis=1)
    
    nll_before = log_loss(y_val, probs_before)
    nll_after = log_loss(y_val, probs_after)
    
    # Predictions don't change with temperature scaling
    y_pred = np.argmax(logits, axis=1)
    accuracy = accuracy_score(y_val, y_pred)
    
    return {
        'temperature': float(optimal_T),
        'nll_before': float(nll_before),
        'nll_after': float(nll_after),
        'accuracy': float(accuracy),
        'logits': logits,
        'probs_before': probs_before,
        'probs_after': probs_after
    }


def evaluate_calibration(probs, labels, n_bins=10):
    """
    Compute calibration metrics (Expected Calibration Error).
    
    Args:
        probs: Predicted probabilities (n_samples, n_classes)
        labels: True labels (n_samples,)
        n_bins: Number of bins for calibration
    
    Returns:
        ECE (Expected Calibration Error)
    """
    # Get predicted class and confidence
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels)
    
    # Bin predictions by confidence
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece


def apply_temperature_scaling(model, X, temperature):
    """
    Apply learned temperature scaling to get calibrated probabilities.
    
    Args:
        model: Trained model
        X: Input features
        temperature: Learned temperature parameter
    
    Returns:
        Calibrated probabilities (n_samples, n_classes)
    """
    logits = get_logits_from_model(model, X)
    scaled_logits = temperature_scaling(logits, temperature)
    calibrated_probs = softmax(scaled_logits, axis=1)
    
    return calibrated_probs
