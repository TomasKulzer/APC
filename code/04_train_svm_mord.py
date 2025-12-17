#!/usr/bin/env python3
"""
04 - Train Ordinal SVM using mord LogisticIT

This script trains an ordinal SVM using mord library's LogisticIT
(Immediate-Threshold ordinal regression classifier).

Usage:
    python code/04_train_svm_mord.py
"""

import argparse
import os
import sys
import joblib

sys.path.insert(0, os.path.dirname(__file__))

from model_training.ordinal_svm_mord import (
    train_with_mord_logistic_it,
    load_data
)


def main(args):
    print("="*60)
    print("ORDINAL SVM TRAINING (mord LogisticIT)")
    print("="*60)
    
    print(f"\nLoading features from: {args.train}")
    print(f"Loading labels from: {args.labels}")
    X_train, y_train = load_data(args.train, args.labels)
    
    print(f"\nData: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Classes: {sorted(set(y_train))}")
    
    # Train with mord LogisticIT
    model = train_with_mord_logistic_it(
        X_train, y_train, 
        alpha=args.alpha, 
        max_iter=args.max_iter,
        verbose=args.verbose
    )
    
    # Save
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    joblib.dump(model, args.out)
    print(f"\nSaved model to: {args.out}")
    print("="*60)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Train ordinal SVM with mord LogisticIT')
    p.add_argument('--train', default='features/combined/combined_train.joblib', help='Path to training features')
    p.add_argument('--labels', default='features/combined/labels_train.joblib', help='Path to integer labels')
    p.add_argument('--out', default='features/model_svm_mord.joblib', help='Output model path')
    p.add_argument('--alpha', type=float, default=1.0, help='Regularization strength (higher = more regularization)')
    p.add_argument('--max_iter', type=int, default=1000, help='Maximum iterations for convergence')
    p.add_argument('--verbose', type=int, default=0, choices=[0, 1, 2], help='Verbosity level')
    args = p.parse_args()
    main(args)
