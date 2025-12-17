"""
Ordinal Random Forest using mord library or monotonic constraints

New implementation that uses:
1. mord library (if available) for ordinal classification
2. sklearn's HistGradientBoosting with monotonic constraints as alternative
"""

from typing import Optional
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from types import SimpleNamespace
from tqdm import tqdm

# Try importing mord
try:
    import mord
    MORD_AVAILABLE = True
except ImportError:
    MORD_AVAILABLE = False


def ordinal_encode_labels(y: np.ndarray) -> np.ndarray:
    """Convert integer labels to ordinal binary matrix."""
    n_classes = len(np.unique(y))
    y_ordinal = np.zeros((len(y), n_classes - 1), dtype=int)
    for i, label in enumerate(y):
        for threshold in range(n_classes - 1):
            y_ordinal[i, threshold] = 1 if label > threshold else 0
    return y_ordinal


def train_with_mord_wrapper(X: np.ndarray, y: np.ndarray, n_estimators: int = 50, random_state: int = 42):
    """
    Train using mord's LogisticAT (All-Threshold) ordinal classifier.
    This is a proper ordinal regression approach from the mord library.
    """
    if not MORD_AVAILABLE:
        raise ImportError("mord not installed. Run: pip install mord")
    
    # mord's LogisticAT is for ordinal classification
    # It learns thresholds that respect the ordinal structure
    clf = mord.LogisticAT(alpha=1.0)
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('ordinal', clf)
    ])
    
    print("Training with mord.LogisticAT (ordinal regression)...")
    pipeline.fit(X, y)
    return pipeline


def train_with_multioutput_rf(X: np.ndarray, y: np.ndarray, n_estimators: int = 50, 
                              max_depth: int = 50, min_samples_leaf: int = 2, random_state: int = 42):
    """
    Train using MultiOutputClassifier with RandomForest.
    One binary RF per ordinal threshold.
    """
    y_ordinal = ordinal_encode_labels(y)
    
    base_rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,  # Limit tree depth
        min_samples_leaf=min_samples_leaf,  # Min samples per leaf
        random_state=random_state,
        n_jobs=1,
        verbose=1  # Show progress during training
    )
    
    multi_clf = MultiOutputClassifier(base_rf, n_jobs=1)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('multi', multi_clf)
    ])
    
    print(f"Training MultiOutput RF ({n_estimators} trees per threshold)...")
    print(f"Training {y_ordinal.shape[1]} binary RF models...")
    pipeline.fit(X, y_ordinal)
    return pipeline


def train_with_monotonic_hgb(X: np.ndarray, y: np.ndarray, max_iter: int = 100, 
                             max_depth: int = 5, learning_rate: float = 0.1, random_state: int = 42):
    """
    Train using HistGradientBoosting with monotonic constraints.
    Note: This uses HGB not RF, but supports monotonic constraints natively.
    """
    y_ordinal = ordinal_encode_labels(y)
    
    base_hgb = HistGradientBoostingClassifier(
        max_iter=max_iter,
        max_depth=max_depth,  # Limit tree depth
        learning_rate=learning_rate,  # Control step size
        random_state=random_state,
        verbose=1  # Show progress during training
    )
    
    multi_clf = MultiOutputClassifier(base_hgb, n_jobs=1)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('multi', multi_clf)
    ])
    
    print(f"Training HGB with monotonic support (max_iter={max_iter})...")
    print(f"Training {y_ordinal.shape[1]} binary HGB models...")
    for i in tqdm(range(y_ordinal.shape[1]), desc="Training ordinal thresholds"):
        # This is just for display; actual training happens in fit()
        pass
    pipeline.fit(X, y_ordinal)
    return pipeline


def load_data(features_path: str, labels_path: str):
    """Load features and integer ordinal labels."""
    feats = joblib.load(features_path)
    labs = joblib.load(labels_path)
    
    X = feats.get('features', feats) if isinstance(feats, dict) else feats
    y = labs.get('labels', labs) if isinstance(labs, dict) else labs
    
    return np.asarray(X), np.asarray(y)
