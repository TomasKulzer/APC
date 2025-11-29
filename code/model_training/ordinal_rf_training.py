"""
Ordinal Random Forest training module

This module trains Random Forest classifiers using ordinal encoding with multi-output classification.
RandomForestClassifier natively supports multi-output, so no wrapper is needed.
"""

from typing import Optional, Dict
import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid, cross_val_score
from types import SimpleNamespace
from tqdm import tqdm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train_ordinal_rf(
    X: np.ndarray,
    y_ordinal: np.ndarray,
    param_grid: Optional[Dict] = None,
    cv: int = 5,
    n_jobs: int = -1,
    random_state: int = 42,
):
    """
    Train an ordinal Random Forest classifier using multi-output classification.
    
    Parameters:
    - X: training features (n_samples, n_features)
    - y_ordinal: ordinal labels (n_samples, n_ordinal_dims) - binary matrix
    - param_grid: parameter grid for search
    - cv: cross-validation folds
    - n_jobs: number of parallel jobs
    - random_state: random seed
    
    Returns:
    - fitted model with best_estimator_, best_params_, best_score_, cv_results_
    """
    if param_grid is None:
        param_grid = {
            'rf__n_estimators': [50, 100, 200],
            'rf__max_depth': [10, 20, None],
            'rf__min_samples_split': [2, 5, 10],
        }
    
    # Random Forest natively supports multi-output
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(
            oob_score=True,
            n_jobs=n_jobs,
            random_state=random_state
        ))
    ])
    
    # Manual search with progress bar
    candidates = list(ParameterGrid(param_grid))
    
    results = []
    for params in tqdm(candidates, desc='Ordinal RF hyperparameter candidates'):
        pipeline.set_params(**params)
        try:
            scores = cross_val_score(pipeline, X, y_ordinal, cv=cv, n_jobs=1, scoring='accuracy')
            results.append({
                'params': params,
                'mean_test_score': float(scores.mean()),
                'std_test_score': float(scores.std())
            })
        except Exception as e:
            results.append({
                'params': params,
                'mean_test_score': float('-inf'),
                'std_test_score': None,
                'error': str(e)
            })
    
    # Select best
    best = max(results, key=lambda r: r['mean_test_score'])
    best_params = best['params']
    
    # Fit on full training data
    pipeline.set_params(**best_params)
    pipeline.fit(X, y_ordinal)
    
    # Get OOB score if available
    oob_score = None
    if hasattr(pipeline.named_steps['rf'], 'oob_score_'):
        oob_score = pipeline.named_steps['rf'].oob_score_
    
    # Return searcher-like object
    searcher = SimpleNamespace()
    searcher.best_params_ = best_params
    searcher.best_score_ = best['mean_test_score']
    searcher.cv_results_ = results
    searcher.best_estimator_ = pipeline
    searcher.oob_score_ = oob_score
    searcher.is_ordinal = True
    return searcher


def load_ordinal_data(features_path: str, labels_path: str):
    """Load features and ordinal labels from separate files."""
    feats_data = joblib.load(features_path)
    labels_data = joblib.load(labels_path)
    
    features = feats_data.get('features', feats_data)
    labels = labels_data.get('labels', labels_data)
    
    return np.asarray(features), np.asarray(labels)


def decode_ordinal_predictions(y_ordinal_pred: np.ndarray) -> np.ndarray:
    """
    Convert ordinal predictions back to class indices.
    For ordinal encoding: sum of binary outputs = class index
    
    Args:
        y_ordinal_pred: (n_samples, n_ordinal_dims) binary predictions
    
    Returns:
        class_indices: (n_samples,) integer class predictions
    """
    return np.sum(y_ordinal_pred, axis=1).astype(int)
