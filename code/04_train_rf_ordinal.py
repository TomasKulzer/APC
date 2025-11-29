"""
Train Ordinal Random Forest Classifier

Trains a multi-output Random Forest classifier using ordinal encoding where each ordinal
dimension is treated as a binary classification problem. Random Forest natively supports
multi-output classification.
"""

import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from model_training.ordinal_rf_training import train_ordinal_rf, load_ordinal_data, decode_ordinal_predictions
import joblib
import numpy as np
from pathlib import Path


def main():
    # Paths
    features_path = 'features/combined/combined_train.joblib'
    labels_path = 'features/combined/labels_train_ordinal.joblib'
    output_path = 'features/model_rf_ordinal.joblib'
    
    print("="*60)
    print("ORDINAL RANDOM FOREST TRAINING")
    print("="*60)
    
    # Load data
    print(f"\nLoading features from: {features_path}")
    print(f"Loading ordinal labels from: {labels_path}")
    X_train, y_train_ordinal = load_ordinal_data(features_path, labels_path)
    
    print(f"\nData loaded:")
    print(f"  Features shape: {X_train.shape}")
    print(f"  Ordinal labels shape: {y_train_ordinal.shape}")
    print(f"  Number of ordinal dimensions: {y_train_ordinal.shape[1]}")
    
    # Define hyperparameter grid
    param_grid = {
        'rf__n_estimators': [50, 100, 200],
        'rf__max_depth': [10, 20, None],
        'rf__min_samples_split': [2, 5, 10],
    }
    
    print(f"\nHyperparameter grid:")
    print(f"  n_estimators: {param_grid['rf__n_estimators']}")
    print(f"  max_depth: {param_grid['rf__max_depth']}")
    print(f"  min_samples_split: {param_grid['rf__min_samples_split']}")
    print(f"  Total candidates: {len(param_grid['rf__n_estimators']) * len(param_grid['rf__max_depth']) * len(param_grid['rf__min_samples_split'])}")
    
    # Train
    print("\n" + "-"*60)
    print("Starting ordinal Random Forest training with CV...")
    print("-"*60)
    
    searcher = train_ordinal_rf(
        X_train,
        y_train_ordinal,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        random_state=42
    )
    
    # Print results
    print("\n" + "="*60)
    print("TRAINING RESULTS")
    print("="*60)
    print(f"\nBest parameters: {searcher.best_params_}")
    print(f"Best CV score (accuracy): {searcher.best_score_:.4f}")
    if searcher.oob_score_ is not None:
        print(f"OOB score: {searcher.oob_score_:.4f}")
    
    print("\nTop 5 candidates:")
    sorted_results = sorted(searcher.cv_results_, key=lambda r: r.get('mean_test_score', float('-inf')), reverse=True)
    for i, result in enumerate(sorted_results[:5], 1):
        params_str = str(result['params'])
        score = result.get('mean_test_score', float('-inf'))
        std = result.get('std_test_score', 0)
        print(f"  {i}. Score: {score:.4f} (+/- {std:.4f})")
        print(f"     {params_str}")
    
    # Save model
    print(f"\nSaving model to: {output_path}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(searcher.best_estimator_, output_path)
    
    # Test decoding
    print("\n" + "-"*60)
    print("Testing ordinal prediction decoding...")
    print("-"*60)
    sample_predictions = searcher.best_estimator_.predict(X_train[:5])
    print(f"Sample ordinal predictions (first 5):")
    print(sample_predictions)
    
    decoded_classes = decode_ordinal_predictions(sample_predictions)
    print(f"\nDecoded to class indices:")
    print(decoded_classes)
    
    # Compare with actual
    actual_classes = np.sum(y_train_ordinal[:5], axis=1)
    print(f"\nActual class indices:")
    print(actual_classes)
    
    print("\n" + "="*60)
    print("ORDINAL RANDOM FOREST TRAINING COMPLETED")
    print("="*60)


if __name__ == '__main__':
    main()
