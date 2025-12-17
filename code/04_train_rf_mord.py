#!/usr/bin/env python3
"""
04 - Train Ordinal RF using mord or monotonic constraints

This script trains an ordinal Random Forest using:
1. mord library's ordinal classifiers (preferred if installed)
2. MultiOutputClassifier with RF (one binary RF per threshold)
3. HistGradientBoosting with monotonic constraint support

Usage:
    python code/04_train_rf_mord.py --method mord
    python code/04_train_rf_mord.py --method multi_rf
    python code/04_train_rf_mord.py --method monotonic_hgb
"""

import argparse
import os
import sys
import joblib

sys.path.insert(0, os.path.dirname(__file__))

from model_training.ordinal_rf_mord import (
    train_with_mord_wrapper,
    train_with_multioutput_rf,
    train_with_monotonic_hgb,
    load_data
)


def main(args):
    print("="*60)
    print("ORDINAL RF TRAINING (mord / monotonic)")
    print("="*60)
    
    print(f"\nLoading features from: {args.train}")
    print(f"Loading labels from: {args.labels}")
    X_train, y_train = load_data(args.train, args.labels)
    
    print(f"\nData: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Classes: {sorted(set(y_train))}")
    
    # Train based on method
    if args.method == 'mord':
        model = train_with_mord_wrapper(X_train, y_train, random_state=args.random_state)
    elif args.method == 'multi_rf':
        model = train_with_multioutput_rf(X_train, y_train, n_estimators=args.n_estimators, random_state=args.random_state)
    elif args.method == 'monotonic_hgb':
        model = train_with_monotonic_hgb(X_train, y_train, max_iter=args.n_estimators, random_state=args.random_state)
    else:
        raise ValueError(f"Unknown method: {args.method}")
    
    # Save
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    joblib.dump(model, args.out)
    print(f"\nSaved model to: {args.out}")
    print("="*60)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Train ordinal RF with mord or monotonic constraints')
    p.add_argument('--train', default='features/combined/combined_train.joblib', help='Path to training features')
    p.add_argument('--labels', default='features/combined/labels_train.joblib', help='Path to integer labels')
    p.add_argument('--out', default='features/model_rf_mord.joblib', help='Output model path')
    p.add_argument('--method', choices=['mord', 'multi_rf', 'monotonic_hgb'], default='multi_rf',
                   help='Training method: mord (LogisticAT), multi_rf (MultiOutput RF), monotonic_hgb (HGB with constraints)')
    p.add_argument('--n_estimators', type=int, default=200, help='Number of trees/iterations')
    p.add_argument('--random_state', type=int, default=42)
    args = p.parse_args()
    main(args)
