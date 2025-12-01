"""
Gradient Boosting training module

This module trains Gradient Boosting classifiers with hyperparameter tuning.
"""

from typing import Optional, Dict
import os
import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import ParameterGrid, cross_val_score, StratifiedKFold
from types import SimpleNamespace
from tqdm import tqdm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train_gradient_boosting(
    X: np.ndarray,
    y: np.ndarray,
    param_grid: Optional[Dict] = None,
    cv: int = 5,
    n_jobs: int = -1,
    random_state: int = 42,
):
    """
    Train a Gradient Boosting classifier with hyperparameter search.
    
    Parameters:
    - X: training features (n_samples, n_features)
    - y: training labels (n_samples,)
    - param_grid: parameter grid for search
    - cv: cross-validation folds
    - n_jobs: number of parallel jobs
    - random_state: random seed
    
    Returns:
    - fitted model with best_estimator_, best_params_, best_score_, cv_results_
    """
    if param_grid is None:
        param_grid = {
            'gb__n_estimators': [50, 100],
            'gb__learning_rate': [0.01, 0.1],
            'gb__max_depth': [3, 5],
            'gb__subsample': [0.8, 1.0],
        }
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('gb', GradientBoostingClassifier(random_state=random_state))
    ])
    
    # Manual search with progress bar
    candidates = list(ParameterGrid(param_grid))
    
    # Create CV splitter
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    
    results = []
    for params in tqdm(candidates, desc='Gradient Boosting hyperparameter search', position=0):
        pipeline.set_params(**params)
        try:
            scores = []
            # Manual CV loop with progress bar for each fold
            n_est = params.get('gb__n_estimators', '?')
            lr = params.get('gb__learning_rate', '?')
            for fold_idx, (train_idx, val_idx) in enumerate(tqdm(cv_splitter.split(X, y), 
                                                                   total=cv, 
                                                                   desc=f'  CV folds (n_est={n_est}, lr={lr})', 
                                                                   position=1, 
                                                                   leave=False)):
                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                y_train_fold, y_val_fold = y[train_idx], y[val_idx]
                
                pipeline.fit(X_train_fold, y_train_fold)
                score = pipeline.score(X_val_fold, y_val_fold)
                scores.append(score)
            
            scores = np.array(scores)
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
    pipeline.fit(X, y)
    
    # Return searcher-like object
    searcher = SimpleNamespace()
    searcher.best_params_ = best_params
    searcher.best_score_ = best['mean_test_score']
    searcher.cv_results_ = results
    searcher.best_estimator_ = pipeline
    return searcher


def load_data(features_path: str, labels_path: str):
    """Load features and labels from separate files."""
    feats_data = joblib.load(features_path)
    labels_data = joblib.load(labels_path)
    
    features = feats_data.get('features', feats_data)
    labels = labels_data.get('labels', labels_data)
    
    return np.asarray(features), np.asarray(labels)
