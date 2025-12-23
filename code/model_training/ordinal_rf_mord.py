"""
Ordinal Random Forest using mord library or monotonic constraints

New implementation that uses:
1. mord library (if available) for ordinal classification
2. sklearn's HistGradientBoosting with monotonic constraints as alternative
"""

from typing import Optional
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def ordinal_encode_labels(y: np.ndarray) -> np.ndarray:
    """Convert integer labels (0-4) to ordinal binary matrix."""
    n_classes = len(np.unique(y))
    y_ordinal = np.zeros((len(y), n_classes - 1), dtype=int)
    for i, label in enumerate(y):
        for threshold in range(n_classes - 1):
            y_ordinal[i, threshold] = 1 if label > threshold else 0
    return y_ordinal


def decode_ordinal_predictions(probas: np.ndarray) -> np.ndarray:
    """Decode multioutput probabilities to single ordinal prediction.
    
    Args:
        probas: Array of shape (n_samples, n_thresholds) with probabilities
                for each threshold classifier
    
    Returns:
        predictions: Array of predicted class labels (0-4)
    """
    predictions = np.zeros(len(probas))
    for i, probs in enumerate(probas):
        # Count how many thresholds predict 1 (y > threshold)
        # This gives us the predicted class
        predictions[i] = np.sum(probs > 0.5)
    return predictions.astype(int)


def train_direct_ordinal_rf(X: np.ndarray, y: np.ndarray, n_estimators: int = 500, 
                            max_depth: int = 6, min_samples_leaf: int = 8, random_state: int = 42):
    """
    DIRECT Ordinal RF using standard multiclass classification.
    This treats the ordinal problem as multiclass classification.
    
    Args:
        X: Training features
        y: Integer labels (0-4)
        n_estimators: Number of trees (default: 500)
        max_depth: Max tree depth (default: 6)
        min_samples_leaf: Min samples per leaf (default: 8)
        random_state: Random seed
    
    Returns:
        Fitted pipeline with scaler and RF
    """
    
    print(f"Training with {len(np.unique(y))} classes: {sorted(np.unique(y))}")
    
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight='balanced_subsample',  # Better than 'balanced'
        random_state=random_state,
        n_jobs=-1,
        verbose=1
    )
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', rf)
    ])
    
    print(f"\n{'='*60}")
    print(f"Training DIRECT Ordinal RF:")
    print(f"  - Number of trees: {n_estimators}")
    print(f"  - Max depth: {max_depth}")
    print(f"  - Min samples/leaf: {min_samples_leaf}")
    print(f"  - Class weight: balanced_subsample")
    print(f"{'='*60}\n")
    
    pipeline.fit(X, y)
    return pipeline


def predict_direct(pipeline, X: np.ndarray) -> np.ndarray:
    """
    Predict using direct ordinal RF.
    
    Args:
        pipeline: Fitted pipeline
        X: Features to predict
    
    Returns:
        Predicted class labels (0-4)
    """
    return pipeline.predict(X)

def load_data(features_path: str, labels_path: str):
    """Load features and integer ordinal labels."""
    feats = joblib.load(features_path)
    labs = joblib.load(labels_path)
    
    X = feats.get('features', feats) if isinstance(feats, dict) else feats
    y = labs.get('labels', labs) if isinstance(labs, dict) else labs
    
    return np.asarray(X), np.asarray(y)
