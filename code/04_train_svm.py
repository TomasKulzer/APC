#!/usr/bin/env python3
"""
04 - Train SVM classifier with GridSearchCV

Usage:
    python code/04_train_svm.py --train features/combined/combined_train.joblib --out features/model_svm.joblib

This imports the `train_svm` function from `code/model_training/svm_training.py`.
"""
import os
import sys
import argparse
import joblib

sys.path.insert(0, os.path.dirname(__file__))

from model_training.svm_training import train_svm, load_data


def main(train_path: str, out_path: str, search: str = 'grid', cv: int = 5, n_jobs: int = -1):
    X_train, y_train = load_data(train_path)
    print(f"Training samples: {X_train.shape[0]}")
    searcher = train_svm(X_train, y_train, cv=cv, search=search, n_jobs=n_jobs)

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    joblib.dump(searcher, out_path)
    print(f"Saved trained model/search object to: {out_path}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--train', default='features/combined/combined_train.joblib')
    p.add_argument('--out', default='features/model_svm.joblib')
    p.add_argument('--search', choices=['grid', 'randomized'], default='grid')
    p.add_argument('--cv', type=int, default=5)
    p.add_argument('--n_jobs', type=int, default=-1)
    args = p.parse_args()
    main(args.train, args.out, search=args.search, cv=args.cv, n_jobs=args.n_jobs)
