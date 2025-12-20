"""
Platt Scaling for Model Calibration

This module implements Platt scaling to calibrate model confidence/probability estimates.
Platt scaling fits a logistic regression model on raw model outputs to convert them
to well-calibrated probability estimates.

Theory:
- Fits parameters A and B such that: P(y=1|f) = 1 / (1 + exp(A*f + B))
- Where f is the raw model output (logit/decision function)
- For multiclass: Apply one-vs-rest or fit per class

The parameters A and B are learned by minimizing negative log-likelihood (NLL)
on the validation set.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score
from scipy.special import softmax


class PlattScaler:
    """
    Platt scaling calibrator using logistic regression.
    
    Fits a logistic regression on model logits to calibrate probabilities.
    """
    
    def __init__(self, n_classes=5):
        """
        Initialize Platt scaler.
        
        Args:
            n_classes: Number of classes
        """
        self.n_classes = n_classes
        self.calibrator = None
    
    def fit(self, logits, labels):
        """
        Fit Platt scaling parameters on validation data.
        
        Args:
            logits: Raw model outputs (n_samples, n_classes)
            labels: True labels (n_samples,)
        
        Returns:
            self
        """
        # Use logistic regression as calibrator
        # For multiclass, this will use one-vs-rest by default
        self.calibrator = LogisticRegression(
            multi_class='multinomial',
            solver='lbfgs',
            max_iter=1000,
            random_state=42
        )
        
        # Fit on logits
        self.calibrator.fit(logits, labels)
        
        return self
    
    def predict_proba(self, logits):
        """
        Apply Platt scaling to get calibrated probabilities.
        
        Args:
            logits: Raw model outputs (n_samples, n_classes)
        
        Returns:
            Calibrated probabilities (n_samples, n_classes)
        """
        if self.calibrator is None:
            raise ValueError("Platt scaler must be fitted first")
        
        return self.calibrator.predict_proba(logits)


def platt_scale(logits, labels):
    """
    Fit Platt scaling and return calibrated probabilities.
    
    Args:
        logits: Raw model outputs (n_samples, n_classes)
        labels: True labels (n_samples,)
    
    Returns:
        PlattScaler fitted on the data
    """
    n_classes = logits.shape[1]
    scaler = PlattScaler(n_classes=n_classes)
    scaler.fit(logits, labels)
    return scaler


def calibrate_model_platt(model, X_val, y_val, get_logits_func):
    """
    Calibrate a model using Platt scaling on validation set.
    
    Args:
        model: Trained model
        X_val: Validation features
        y_val: Validation labels
        get_logits_func: Function to extract logits from model
    
    Returns:
        dict with Platt scaler and calibration info
    """
    # Get logits from model
    logits = get_logits_func(model, X_val)
    
    # Get uncalibrated probabilities
    probs_before = softmax(logits, axis=1)
    
    # Fit Platt scaler
    platt_scaler = platt_scale(logits, y_val)
    
    # Get calibrated probabilities
    probs_after = platt_scaler.predict_proba(logits)
    
    # Compute metrics
    nll_before = log_loss(y_val, probs_before)
    nll_after = log_loss(y_val, probs_after)
    
    # Predictions don't change with calibration
    y_pred = np.argmax(logits, axis=1)
    accuracy = accuracy_score(y_val, y_pred)
    
    return {
        'platt_scaler': platt_scaler,
        'nll_before': float(nll_before),
        'nll_after': float(nll_after),
        'accuracy': float(accuracy),
        'logits': logits,
        'probs_before': probs_before,
        'probs_after': probs_after
    }


def apply_platt_scaling(model, X, platt_scaler, get_logits_func):
    """
    Apply fitted Platt scaling to get calibrated probabilities.
    
    Args:
        model: Trained model
        X: Input features
        platt_scaler: Fitted PlattScaler object
        get_logits_func: Function to extract logits from model
    
    Returns:
        Calibrated probabilities (n_samples, n_classes)
    """
    logits = get_logits_func(model, X)
    return platt_scaler.predict_proba(logits)
