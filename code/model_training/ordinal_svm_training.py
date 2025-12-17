"""
Ordinal SVM training module

This module trains SVM classifiers using ordinal encoding with multi-output classification.
Each ordinal dimension is treated as a binary classification problem.
"""

from typing import Optional, Dict
import os
import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import ParameterGrid, ParameterSampler, cross_val_score, StratifiedKFold
from types import SimpleNamespace
from tqdm import tqdm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train_ordinal_svm(
    X: np.ndarray,
    y_ordinal: np.ndarray,
    param_grid: Optional[Dict] = None,
    cv: int = 5,
    search: str = 'grid',
    n_jobs: int = -1,
    random_state: int = 42,
):
    """
    Train an ordinal SVM classifier using multi-output binary classification.
    
    Parameters
    - X: training features (n_samples, n_features)
    - y_ordinal: ordinal labels (n_samples, n_ordinal_dims) - binary matrix
    - param_grid: parameter grid for search
    - cv: cross-validation folds
    - search: 'grid' or 'randomized'
    - n_jobs: number of parallel jobs
    - random_state: random seed
    
    Returns
    - fitted search object with best_estimator_, best_params_, best_score_, cv_results_
    """
    if param_grid is None:
        # Minimal grid for fast testing
        param_grid = {
            'multioutput__estimator__C': [1],
            'multioutput__estimator__gamma': ['scale'],
        }
    
    # Create multi-output pipeline
    # Each ordinal dimension gets its own binary SVC
    base_svc = SVC(kernel='rbf', probability=True, cache_size=1000, random_state=random_state)
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('multioutput', MultiOutputClassifier(base_svc, n_jobs=1))
    ])
    
    # Manual search loop with progress bar
    if search == 'grid':
        candidates = list(ParameterGrid(param_grid))
    elif search == 'randomized':
        candidates = list(ParameterSampler(param_grid, n_iter=10, random_state=random_state))
    else:
        raise ValueError("search must be 'grid' or 'randomized'")
    
    results = []
    # Create CV splitter - use first ordinal dimension for stratification
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    
    # Iterate candidates with nested progress bars
    for params in tqdm(candidates, desc='Ordinal SVM hyperparameter search', position=0):
        pipeline.set_params(**params)
        try:
            scores = []
            # Manual CV loop with progress bar for each fold
            for fold_idx, (train_idx, val_idx) in enumerate(tqdm(cv_splitter.split(X, y_ordinal[:, 0]), 
                                                                   total=cv, 
                                                                   desc=f'  CV folds (C={params.get("multioutput__estimator__C", "?")})', 
                                                                   position=1, 
                                                                   leave=False)):
                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                y_train_fold, y_val_fold = y_ordinal[train_idx], y_ordinal[val_idx]
                
                pipeline.fit(X_train_fold, y_train_fold)
                score = pipeline.score(X_val_fold, y_val_fold)
                scores.append(score)
            
            scores = np.array(scores)
            results.append({'params': params, 'mean_test_score': float(scores.mean()), 'std_test_score': float(scores.std())})
        except Exception as e:
            results.append({'params': params, 'mean_test_score': float('-inf'), 'std_test_score': None, 'error': str(e)})
    
    # Choose best candidate
    best = max(results, key=lambda r: r['mean_test_score'])
    best_params = best['params']
    
    # Fit pipeline on full training data with best params
    pipeline.set_params(**best_params)
    pipeline.fit(X, y_ordinal)
    
    # Create searcher-like object
    searcher = SimpleNamespace()
    searcher.best_params_ = best_params
    searcher.best_score_ = best['mean_test_score']
    searcher.cv_results_ = results
    searcher.best_estimator_ = pipeline
    searcher.is_ordinal = True
    return searcher


def load_ordinal_data(features_path: str, labels_path: str):
    """Load features and ordinal labels from separate files."""
    feats_data = joblib.load(features_path)
    labels_data = joblib.load(labels_path)
    
    features = feats_data.get('features', feats_data) if isinstance(feats_data, dict) else feats_data
    # Ordinal labels are stored in a dict with key 'labels'
    if isinstance(labels_data, dict):
        labels = labels_data.get('labels', labels_data)
    else:
        labels = labels_data
    
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
