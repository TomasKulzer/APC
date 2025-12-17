"""
Train Ordinal Gradient Boosting Classifier

Trains a Gradient Boosting classifier using ordinal encoding where each ordinal dimension
is treated as a binary classification problem.
"""

import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from model_training.gb_training import load_ordinal_data
import joblib
from pathlib import Path


def main():
    # Paths
    features_path = 'features/combined/combined_train.joblib'
    labels_path = 'features/combined/labels_train_ordinal.joblib'
    val_features_path = 'features/combined/combined_val.joblib'
    val_labels_path = 'features/combined/labels_val_ordinal.joblib'
    output_path = 'features/model_gb_ordinal.joblib'
    
    print("="*60)
    print("ORDINAL GRADIENT BOOSTING TRAINING")
    print("="*60)
    
    # Load data
    print(f"\nLoading features from: {features_path}")
    print(f"Loading ordinal labels from: {labels_path}")
    X_train, y_train_ordinal = load_ordinal_data(features_path, labels_path)
    
    print(f"Loading validation data from: {val_features_path}")
    X_val, y_val_ordinal = load_ordinal_data(val_features_path, val_labels_path)
    
    print(f"\nData loaded:")
    print(f"  Train features shape: {X_train.shape}")
    print(f"  Train ordinal labels shape: {y_train_ordinal.shape}")
    print(f"  Val features shape: {X_val.shape}")
    print(f"  Number of ordinal dimensions: {y_train_ordinal.shape[1]}")
    
    # Use simple training without CV due to high dimensionality
    # GB is extremely slow with 26k features, so we train with minimal config
    print("\n" + "-"*60)
    print("Note: Using validation set instead of CV for speed")
    print("GB is very slow with 26,468 features")
    print("-"*60)
    
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import accuracy_score
    
    # Minimal config for fast training
    params = {'n_estimators': 50, 'learning_rate': 0.1, 'max_depth': 3}
    
    print(f"\nTraining ordinal GB with params: {params}")
    
    base_gb = GradientBoostingClassifier(random_state=42, verbose=1, **params)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('multioutput', MultiOutputClassifier(base_gb, n_jobs=1))
    ])
    
    pipeline.fit(X_train, y_train_ordinal)
    
    train_score = accuracy_score(y_train_ordinal, pipeline.predict(X_train))
    val_score = accuracy_score(y_val_ordinal, pipeline.predict(X_val))
    
    print(f"\n  Train accuracy: {train_score:.4f}")
    print(f"  Val accuracy: {val_score:.4f}")
    
    # Save model
    print("\n" + "="*60)
    print("TRAINING RESULTS")
    print("="*60)
    print(f"\nParameters: {params}")
    print(f"Validation accuracy: {val_score:.4f}")
    print(f"Train accuracy: {train_score:.4f}")
    
    print(f"\nSaving model to: {output_path}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_path)
    
    print("\n" + "="*60)
    print("ORDINAL GRADIENT BOOSTING TRAINING COMPLETED")
    print("="*60)


if __name__ == '__main__':
    main()
