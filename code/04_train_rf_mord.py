#!/usr/bin/env python3
"""
04 - Train FIXED Ordinal Random Forest

This script trains the FIXED ordinal RF model with:
- class_weight='balanced' for threshold imbalance
- Shallow trees (max_depth=6) to prevent overfitting
- Strong regularization (min_samples_leaf=8)

Usage:
    python code/04_train_rf_mord.py
    python code/04_train_rf_mord.py --max_depth 8 --min_samples_leaf 12
"""

import argparse
import os
import sys
import joblib

sys.path.insert(0, os.path.dirname(__file__))

from model_training.ordinal_rf_mord import train_with_multioutput_rf, load_data


def train_model(args):
    """Train the FIXED ordinal RF model."""
    
    print("="*80)
    print("TRAINING FIXED ORDINAL RANDOM FOREST")
    print("="*80)
    print("\nFIXES APPLIED:")
    print("  [X] class_weight='balanced' - handles threshold imbalance")
    print(f"  [X] max_depth={args.max_depth} - prevents overfitting")
    print(f"  [X] min_samples_leaf={args.min_samples_leaf} - regularization")
    print("="*80)
    
    # Load training data
    print("\nLoading training data...")
    X_train, y_train = load_data(args.train, args.labels)
    print(f"  Training: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    
    from collections import Counter
    train_counts = Counter(y_train)
    for cls in sorted(train_counts.keys()):
        print(f"    Class {cls}: {train_counts[cls]} samples")
    
    # Train FIXED model
    print("\nTraining model...")
    model = train_with_multioutput_rf(
        X_train, y_train,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.random_state
    )
    
    # Save model
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    joblib.dump(model, args.out)
    print(f"\n[OK] Saved model to: {args.out}")
    print("="*80)


def main(args):
    train_model(args)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Train FIXED ordinal RF')
    p.add_argument('--train', default='features/combined/combined_train.joblib', help='Path to training features')
    p.add_argument('--labels', default='features/combined/labels_train.joblib', help='Path to integer labels')
    p.add_argument('--out', default='features/model_rf_mord.joblib', help='Output model path')
    p.add_argument('--n_estimators', type=int, default=800, help='Number of trees (default: 500)')
    p.add_argument('--max_depth', type=int, default=4, help='Max tree depth (default: 6)')
    p.add_argument('--min_samples_leaf', type=int, default=12, help='Min samples per leaf (default: 8)')
    p.add_argument('--random_state', type=int, default=42)
    args = p.parse_args()
    main(args)
