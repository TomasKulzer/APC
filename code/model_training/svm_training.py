"""
SVM training module

This module contains functions for training SVM classifiers with hyperparameter search.
"""

from typing import Optional, Dict
import os
import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import ParameterGrid, ParameterSampler, cross_val_score, StratifiedKFold
from types import SimpleNamespace
from tqdm import tqdm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train_svm(
    X: np.ndarray,
    y: np.ndarray,
    param_grid: Optional[Dict] = None,
    cv: int = 5,
    search: str = 'grid',
    n_jobs: int = -1,
    random_state: int = 42,
):
    """
    Train an SVM classifier using a scaling + SVC pipeline and hyperparameter search.

    Parameters
    - X, y: training data (features and integer labels)
    - param_grid: parameter grid for GridSearchCV. If None, a sensible default will be used.
    - cv: cross-validation folds
    - search: 'grid' or 'randomized'
    - n_jobs: number of parallel jobs for search
    - random_state: random seed for reproducibility (used by randomized search)

    Returns
    - fitted search object with best_estimator_, best_params_, best_score_, cv_results_
    """
    if param_grid is None:
        # Minimal grid for fast testing
        param_grid = {
            'svc__C': [1],
            'svc__gamma': ['scale'],
        }

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(kernel='rbf', probability=True, cache_size=1000))
    ])

    # Manual search loop with progress bar so user sees candidate progress.
    if search == 'grid':
        candidates = list(ParameterGrid(param_grid))
    elif search == 'randomized':
        candidates = list(ParameterSampler(param_grid, n_iter=10, random_state=random_state))
    else:
        raise ValueError("search must be 'grid' or 'randomized'")

    results = []
    # Create CV splitter
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    
    # Iterate candidates with nested progress bars for better visibility
    for params in tqdm(candidates, desc='SVM hyperparameter search', position=0):
        pipeline.set_params(**params)
        try:
            scores = []
            # Manual CV loop with progress bar for each fold
            for fold_idx, (train_idx, val_idx) in enumerate(tqdm(cv_splitter.split(X, y), 
                                                                   total=cv, 
                                                                   desc=f'  CV folds (C={params.get("svc__C", "?")})', 
                                                                   position=1, 
                                                                   leave=False)):
                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                y_train_fold, y_val_fold = y[train_idx], y[val_idx]
                
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
    pipeline.fit(X, y)

    # Construct a lightweight searcher-like object for compatibility with saved artifacts
    searcher = SimpleNamespace()
    searcher.best_params_ = best_params
    searcher.best_score_ = best['mean_test_score']
    searcher.cv_results_ = results
    searcher.best_estimator_ = pipeline
    return searcher


def load_data(path: str):
    """Load training data from a joblib file, supporting multiple common key formats."""
    data = joblib.load(path)
    # Support multiple common key names
    if 'X' in data and 'y' in data:
        return data['X'], data['y']
    if 'features' in data and 'labels' in data:
        return data['features'], data['labels']
    if 'features' in data and 'labels' not in data and 'y' in data:
        return data['features'], data['y']
    # Check for separate labels file (common in combined/ workflow)
    if 'features' in data:
        # Try to find labels_{split}.joblib in same directory
        abs_path = os.path.abspath(path)
        dirname = os.path.dirname(abs_path)
        basename = os.path.basename(abs_path)
        # Extract split name: e.g., combined_train.joblib -> train
        if basename.startswith('combined_') and basename.endswith('.joblib'):
            split_name = basename.replace('combined_', '').replace('.joblib', '')
            labels_path = os.path.join(dirname, f'labels_{split_name}.joblib')
            if os.path.exists(labels_path):
                labels_data = joblib.load(labels_path)
                if 'labels' in labels_data:
                    return data['features'], labels_data['labels']
                elif isinstance(labels_data, (list, np.ndarray)):
                    return data['features'], labels_data
    raise KeyError('Input file must contain keys (X,y) or (features,labels), or have a separate labels file')
