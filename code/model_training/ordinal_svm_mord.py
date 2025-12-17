"""
Ordinal SVM using mord library's LogisticIT

Implementation using mord's LogisticIT (Immediate-Threshold) ordinal classifier.
"""

import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import time

# Try importing mord
try:
    import mord
    MORD_AVAILABLE = True
except ImportError:
    MORD_AVAILABLE = False
    print("Warning: mord not installed. Run: pip install mord")


def train_with_mord_logistic_it(X: np.ndarray, y: np.ndarray, alpha: float = 1.0, 
                                max_iter: int = 1000, verbose: int = 0):
    """
    Train using mord's LogisticIT (Immediate-Threshold variant).
    Ordinal regression approach that respects the ordering of classes.
    
    Parameters:
    - alpha: Regularization strength (higher = more regularization)
    - max_iter: Maximum iterations for convergence
    - verbose: Verbosity level (0=silent, 1=some info, 2=detailed)
    """
    if not MORD_AVAILABLE:
        raise ImportError("mord not installed. Run: pip install mord")
    
    clf = mord.LogisticIT(alpha=alpha, max_iter=max_iter, verbose=verbose)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('ordinal', clf)
    ])
    
    print("Training with mord.LogisticIT (ordinal logistic regression)...")
    print("Fitting model (this may take a while for large datasets)...")
    
    # Use tqdm with a manual update loop to show progress
    with tqdm(total=100, desc="Training progress", bar_format='{l_bar}{bar}| {elapsed}') as pbar:
        # Start fitting in the background, but we can't easily track real progress
        # so we'll show an indeterminate progress bar
        pbar.update(10)
        pipeline.fit(X, y)
        pbar.update(90)
    
    print("Training complete!")
    return pipeline


def load_data(features_path: str, labels_path: str):
    """Load features and integer ordinal labels."""
    feats = joblib.load(features_path)
    labs = joblib.load(labels_path)
    
    X = feats.get('features', feats) if isinstance(feats, dict) else feats
    y = labs.get('labels', labs) if isinstance(labs, dict) else labs
    
    return np.asarray(X), np.asarray(y)
