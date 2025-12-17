"""
Train Ordinal SVM Classifier

Trains a multi-output SVM classifier using ordinal encoding where each ordinal dimension
is treated as a binary classification problem.
"""

import sys
import os
import argparse

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from model_training.ordinal_svm_training import train_ordinal_svm, load_ordinal_data
import joblib


def main(train_path: str, labels_path: str, out_path: str, cv: int = 5):
    print("="*60)
    print("ORDINAL SVM TRAINING")
    print("="*60)
    
    # Load data
    print(f"\nLoading features from: {train_path}")
    print(f"Loading ordinal labels from: {labels_path}")
    X_train, y_train_ordinal = load_ordinal_data(train_path, labels_path)
    
    print(f"\nData loaded:")
    print(f"  Features shape: {X_train.shape}")
    print(f"  Ordinal labels shape: {y_train_ordinal.shape}")
    
    # Train with minimal grid (fast)
    searcher = train_ordinal_svm(X_train, y_train_ordinal, cv=cv, n_jobs=-1)
    
    print(f"\nBest CV score: {searcher.best_score_:.4f}")
    print(f"Best parameters: {searcher.best_params_}")
    
    # Save model
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    joblib.dump(searcher, out_path)
    print(f"\nSaved trained model to: {out_path}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--train', default='features/combined/combined_train.joblib')
    p.add_argument('--labels', default='features/combined/labels_train_ordinal.joblib')
    p.add_argument('--out', default='features/model_svm_ordinal.joblib')
    p.add_argument('--cv', type=int, default=5)
    args = p.parse_args()
    main(args.train, args.labels, args.out, cv=args.cv)
