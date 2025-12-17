"""
Train Ordinal Gradient Boosting using LightGBM

Uses LightGBM with multiclass objective for ordinal classification.
"""

import argparse
from pathlib import Path
import joblib
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from model_training.ordinal_gb_lightgbm import (
    train_with_lightgbm,
    load_data,
    LIGHTGBM_AVAILABLE
)


def main():
    parser = argparse.ArgumentParser(
        description='Train ordinal gradient boosting with LightGBM'
    )
    parser.add_argument(
        '--n_iterations',
        type=int,
        default=500,
        help='Number of boosting iterations (default: 500)'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.1,
        help='Learning rate (default: 0.1)'
    )
    parser.add_argument(
        '--features_dir',
        type=str,
        default='features/combined',
        help='Directory containing combined features'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='features/model_gb_lightgbm.joblib',
        help='Output model path'
    )
    
    args = parser.parse_args()
    
    # Check library availability
    if not LIGHTGBM_AVAILABLE:
        print("ERROR: LightGBM not installed. Install with: pip install lightgbm")
        sys.exit(1)
    
    # Load training data
    print(f"Loading training data from {args.features_dir}/...")
    X_train, y_train = load_data(
        f'{args.features_dir}/combined_train.joblib',
        f'{args.features_dir}/labels_train.joblib'  # Use integer labels, not ordinal
    )
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Labels shape: {y_train.shape}")
    print(f"Number of classes: {len(set(y_train))}")
    print(f"Label distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print()
    
    # Train model with LightGBM
    model = train_with_lightgbm(
        X_train, y_train,
        num_iterations=args.n_iterations,
        learning_rate=args.learning_rate
    )
    
    # Save model
    print(f"\nSaving model to {args.output}...")
    joblib.dump(model, args.output)
    print("Model saved successfully!")
    
    # Quick training accuracy check
    train_preds = model.predict(X_train)
    train_acc = (train_preds == y_train).mean()
    print(f"\nTraining accuracy: {train_acc:.4f}")


if __name__ == '__main__':
    import numpy as np
    main()
