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

from model_training.gb import (
    train_with_lightgbm,
    load_data,
    LIGHTGBM_AVAILABLE
)


def main():
    parser = argparse.ArgumentParser(
        description='Train ordinal gradient boosting with LightGBM (with hyperparameter tuning and early stopping)'
    )
    parser.add_argument(
        '--n_iterations',
        type=int,
        default=5000,
        help='Number of boosting iterations (default: 5000)'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.01,
        help='Learning rate (default: 0.02, lower for better generalization)'
    )
    parser.add_argument(
        '--num_leaves',
        type=int,
        default=50,
        help='Maximum tree leaves (default: 31)'
    )
    parser.add_argument(
        '--max_depth',
        type=int,
        default=8,
        help='Maximum tree depth (default: 8)'
    )
    parser.add_argument(
        '--feature_fraction',
        type=float,
        default=0.5,
        help='Fraction of features per iteration (default: 0.8)'
    )
    parser.add_argument(
        '--bagging_fraction',
        type=float,
        default=0.5,
        help='Fraction of data per iteration (default: 0.8)'
    )
    parser.add_argument(
        '--min_child_samples',
        type=int,
        default=50,
        help='Minimum samples per leaf (default: 50, from tuning)'
    )
    parser.add_argument(
        '--reg_alpha',
        type=float,
        default=0.1,
        help='L1 regularization (default: 0.1, increase to reduce overfitting)'
    )
    parser.add_argument(
        '--reg_lambda',
        type=float,
        default=0.1,
        help='L2 regularization (default: 0.1, increase to reduce overfitting)'
    )
    parser.add_argument(
        '--use_tuning',
        action='store_true',
        help='Enable hyperparameter tuning (slower but may improve accuracy)'
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
        default='features/model_gb.joblib',
        help='Output model path'
    )
    
    args = parser.parse_args()
    
    # Check library availability
    if not LIGHTGBM_AVAILABLE:
        print("ERROR: LightGBM not installed. Install with: pip install lightgbm")
        sys.exit(1)
    
    # Display configuration
    print("="*70)
    print("LightGBM Ordinal Classification Training")
    print("="*70)
    print(f"Iterations:        {args.n_iterations}")
    print(f"Learning rate:     {args.learning_rate}")
    print(f"Num leaves:        {args.num_leaves}")
    print(f"Max depth:         {args.max_depth}")
    print(f"Feature fraction:  {args.feature_fraction}")
    print(f"Bagging fraction:  {args.bagging_fraction}")
    print(f"Min child samples: {args.min_child_samples}")
    print(f"L1 regularization: {args.reg_alpha}")
    print(f"L2 regularization: {args.reg_lambda}")
    print(f"Hyperparameter tuning: {'Yes' if args.use_tuning else 'No'}")
    print("="*70)
    
    # Load training data
    print(f"\nLoading training data from {args.features_dir}/...")
    X_train, y_train = load_data(
        f'{args.features_dir}/combined_train.joblib',
        f'{args.features_dir}/labels_train.joblib'
    )
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Labels shape: {y_train.shape}")
    print(f"Number of classes: {len(set(y_train))}")
    
    # Train model with LightGBM
    model = train_with_lightgbm(
        X_train, y_train,
        num_iterations=args.n_iterations,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        max_depth=args.max_depth,
        feature_fraction=args.feature_fraction,
        bagging_fraction=args.bagging_fraction,
        min_child_samples=args.min_child_samples,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        use_tuning=args.use_tuning
    )
    
    # Load and evaluate on validation set
    print("\n" + "="*70)
    print("VALIDATION SET EVALUATION")
    print("="*70)
    try:
        X_val, y_val = load_data(
            f'{args.features_dir}/combined_val.joblib',
            f'{args.features_dir}/labels_val.joblib'
        )
        print(f"Validation data shape: {X_val.shape}")
        
        val_acc = model.score(X_val, y_val)
        train_acc_full = model.score(X_train, y_train)
        gap = train_acc_full - val_acc
        
        print(f"\nTraining set accuracy:   {train_acc_full:.4f} ({train_acc_full*100:.2f}%)")
        print(f"Validation set accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
        print(f"Train-Val gap:           {gap:.4f} ({gap*100:.2f}%)")
        
        if gap > 0.15:
            print("\n  High train-val gap suggests overfitting!")
            print("   Recommendations:")
            print("   - Increase regularization: --reg_alpha 0.5 --reg_lambda 0.5")
            print("   - Lower learning rate: --learning_rate 0.01")
            print("   - Reduce complexity: --num_leaves 31 --max_depth 6")
        elif gap > 0.10:
            print("\n Moderate overfitting detected")
            print("   Try: --reg_alpha 0.3 --reg_lambda 0.3")
        elif val_acc > 0.85:
            print("\n Excellent validation performance!")
        else:
            print(f"\n Validation: {val_acc*100:.1f}% - Gap: {gap*100:.1f}%")
        
    except FileNotFoundError:
        print("Validation data not found. Skipping validation evaluation.")
    
    print("="*70)
    
    # Save model
    print(f"\nSaving model to {args.output}...")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output)
    print(f"Model saved successfully to {args.output}!")


if __name__ == '__main__':
    import numpy as np
    main()
