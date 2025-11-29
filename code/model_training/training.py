"""
SVM training script

This script loads training data (features and labels), performs scaling + SVM
training with hyperparameter search (GridSearchCV by default), and saves the
best model to disk using joblib.

Usage examples:

# Train using pre-split train file created by split_dataset.py
/home/skido/APC/.venv/bin/python code/model_training/training.py \
    --train ../features/combined/train.joblib \
    --out model_svm.joblib

# Use randomized search instead of grid search
/home/skido/APC/.venv/bin/python code/model_training/training.py \
    --train ../features/combined/train.joblib \
    --out model_svm.joblib --search randomized

The script expects the train file to be a joblib containing keys `X` and `y`.
If your feature file uses different keys (e.g., `features` / `labels`) it will
attempt to detect them.
"""

from typing import Optional, Dict
import os
import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import argparse


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
    - fitted search object (GridSearchCV or RandomizedSearchCV)
    """
    if param_grid is None:
        param_grid = {
            'svc__C': [0.1, 1, 10],
            'svc__gamma': ['scale', 'auto', 0.001],
        }

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(kernel='rbf', probability=True))
    ])

    if search == 'grid':
        searcher = GridSearchCV(pipeline, param_grid=param_grid, cv=cv, n_jobs=n_jobs, verbose=2)
    elif search == 'randomized':
        # For randomized, param_grid should be distributions; we still accept same grid by sampling
        searcher = RandomizedSearchCV(pipeline, param_distributions=param_grid, n_iter=10, cv=cv, n_jobs=n_jobs, random_state=random_state, verbose=2)
    else:
        raise ValueError("search must be 'grid' or 'randomized'")

    searcher.fit(X, y)
    return searcher


def _load_data(path: str):
    data = joblib.load(path)
    # Support multiple common key names
    if 'X' in data and 'y' in data:
        return data['X'], data['y']
    if 'features' in data and 'labels' in data:
        return data['features'], data['labels']
    if 'features' in data and 'labels' not in data and 'y' in data:
        return data['features'], data['y']
    raise KeyError('Input file must contain keys (X,y) or (features,labels)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SVM classifier with hyperparameter search')
    parser.add_argument('--train', required=True, help='Path to train joblib file (keys X,y or features,labels)')
    parser.add_argument('--out', required=True, help='Output path for trained model (joblib)')
    parser.add_argument('--search', choices=['grid', 'randomized'], default='grid', help='Search strategy')
    parser.add_argument('--cv', type=int, default=5, help='Cross-validation folds')
    parser.add_argument('--n_jobs', type=int, default=-1, help='Parallel jobs for search')
    parser.add_argument('--random_state', type=int, default=42)

    args = parser.parse_args()

    X_train, y_train = _load_data(args.train)

    print(f'Training samples: {X_train.shape[0]}')

    searcher = train_svm(X_train, y_train, cv=args.cv, search=args.search, n_jobs=args.n_jobs, random_state=args.random_state)

    print('Best params:')
    print(searcher.best_params_)
    print(f'Best CV score: {searcher.best_score_:.4f}')

    # Save the full search object (contains best_estimator_)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    joblib.dump(searcher, args.out)
    print(f'Saved trained model/search object to: {args.out}')
