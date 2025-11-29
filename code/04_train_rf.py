#!/usr/bin/env python3
"""
04 - Train Random Forest classifier with GridSearchCV

Usage:
    python code/04_train_rf.py --train features/combined/combined_train.joblib --out features/model_rf.joblib

This imports the `train_random_forest` function from `code/model_training/rf_training.py`.
"""
import os
import sys
import argparse
import joblib

sys.path.insert(0, os.path.dirname(__file__))

from model_training.rf_training import train_random_forest, load_data


def main(train_path: str, out_path: str, search: str = 'grid', cv: int = 5, n_jobs: int = -1, use_oob: bool = True):
    X_train, y_train = load_data(train_path)
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Using OOB scoring: {use_oob}")
    
    searcher = train_random_forest(X_train, y_train, cv=cv, search=search, n_jobs=n_jobs, use_oob=use_oob)

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    joblib.dump(searcher, out_path)
    
    print(f"\nBest parameters: {searcher.best_params_}")
    print(f"Best CV score: {searcher.best_score_:.4f}")
    if hasattr(searcher, 'oob_score_'):
        print(f"OOB score: {searcher.oob_score_:.4f}")
    print(f"\nSaved trained model to: {out_path}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--train', default='features/combined/combined_train.joblib')
    p.add_argument('--out', default='features/model_rf.joblib')
    p.add_argument('--search', choices=['grid', 'randomized'], default='grid')
    p.add_argument('--cv', type=int, default=5)
    p.add_argument('--n_jobs', type=int, default=-1)
    p.add_argument('--use_oob', action='store_true', default=True, help='Enable out-of-bag scoring')
    args = p.parse_args()
    main(args.train, args.out, search=args.search, cv=args.cv, n_jobs=args.n_jobs, use_oob=args.use_oob)
