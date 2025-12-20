"""
Train an ordinal SVM using threshold-based approach for oil palm fruit ripeness classification.

This script loads training features and labels, trains a threshold-based ordinal SVM,
and saves the trained model to disk.

Usage:
    python 04_train_svm.py --train features/combined/combined_train.joblib \
                           --labels features/combined/labels_train.joblib \
                           --out features/model_svm_ordinal.joblib \
                           --C 1.0 \
                           --gamma scale \
                           --verbose 1
"""

import argparse
import joblib
import sys
from pathlib import Path

# Add parent directory to path to import from model_training
sys.path.insert(0, str(Path(__file__).parent))

from model_training.ordinal_svm_threshold import train_threshold_svm, load_data


def main():
    """
    Main function to train an ordinal threshold SVM model.
    
    Parses command-line arguments, loads data, trains the model,
    and saves the trained classifier to disk.
    """
    parser = argparse.ArgumentParser(
        description="Train ordinal SVM using threshold-based approach"
    )
    
    parser.add_argument(
        '--train',
        type=str,
        default='features/combined/combined_train.joblib',
        help='Path to training features (joblib file) (default: features/combined/combined_train.joblib)'
    )
    
    parser.add_argument(
        '--labels',
        type=str,
        default='features/combined/labels_train.joblib',
        help='Path to training labels (joblib file) (default: features/combined/labels_train.joblib)'
    )
    
    parser.add_argument(
        '--out',
        type=str,
        default='features/model_svm_ordinal.joblib',
        help='Output path for the trained model (joblib file) (default: features/model_svm_ordinal.joblib)'
    )
    
    parser.add_argument(
        '--C',
        type=float,
        default=1.0,
        help='Regularization parameter for SVM (default: 1.0)'
    )
    
    parser.add_argument(
        '--gamma',
        type=str,
        default='scale',
        help='Kernel coefficient for RBF (default: scale)'
    )
    
    parser.add_argument(
        '--kernel',
        type=str,
        default='linear',
        choices=['linear', 'rbf'],
        help='Kernel type: linear (fast) or rbf (slower but more accurate) (default: linear)'
    )
    
    parser.add_argument(
        '--max_iter',
        type=int,
        default=2000,
        help='Maximum iterations for SVM solver, -1 for no limit (default: 500)'
    )
    
    parser.add_argument(
        '--subsample',
        type=float,
        default=1.0,
        help='Fraction of training data to use (0.0-1.0). Use 0.3-0.5 for much faster training (default: 1.0)'
    )
    
    parser.add_argument(
        '--verbose',
        type=int,
        default=0,
        choices=[0, 1, 2],
        help='Verbosity level: 0 (silent), 1 (info), 2 (debug) (default: 0)'
    )
    
    args = parser.parse_args()
    
    # Convert gamma to float if it's a number string
    gamma = args.gamma
    try:
        gamma = float(gamma)
    except ValueError:
        pass  # Keep as string (e.g., 'scale', 'auto')
    
    # Print configuration
    print("=" * 70)
    print("Training Ordinal SVM (Threshold-based Approach)")
    print("=" * 70)
    print(f"Training features: {args.train}")
    print(f"Training labels:   {args.labels}")
    print(f"Output model:      {args.out}")
    print(f"C parameter:       {args.C}")
    print(f"Kernel:            {args.kernel}")
    print(f"Gamma:             {gamma}")
    print(f"Max iterations:    {args.max_iter}")
    print(f"Subsample:         {args.subsample*100:.0f}%")
    print(f"Verbosity:         {args.verbose}")
    print("=" * 70)
    
    # Load data
    print("\n[1/3] Loading training data...")
    X_train, y_train = load_data(args.train, args.labels)
    
    print(f"  Features shape: {X_train.shape}")
    print(f"  Labels shape:   {y_train.shape}")
    print(f"  Unique classes: {sorted(set(y_train))}")
    print(f"  Class distribution:")
    
    # Print class distribution
    class_names = ['Immature', 'Partially Ripe', 'Fully Ripe', 'Overripe', 'Decayed']
    for cls in range(5):
        count = (y_train == cls).sum()
        print(f"    {cls} ({class_names[cls]}): {count} samples")
    
    # Train model
    print(f"\n[2/3] Training Ordinal SVM model...")
    print(f"  This will train 4 binary SVMs (one per threshold)...")
    if args.kernel == 'linear':
        print(f"  Using LINEAR kernel with LinearSVC (optimized for speed)")
    else:
        print(f"  Using RBF kernel (slower but may be more accurate)")
    
    if args.subsample < 1.0:
        print(f"  Using {args.subsample*100:.0f}% of training data for faster training")
    
    classifier = train_threshold_svm(
        X_train,
        y_train,
        C=args.C,
        gamma=gamma,
        kernel=args.kernel,
        max_iter=args.max_iter,
        subsample=args.subsample,
        verbose=args.verbose
    )
    
    # Save model
    print(f"\n[3/3] Saving trained model to {args.out}...")
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, args.out)
    
    print("\n" + "=" * 70)
    print("Training complete!")
    print(f"Model saved to: {args.out}")
    print("=" * 70)


if __name__ == "__main__":
    main()
