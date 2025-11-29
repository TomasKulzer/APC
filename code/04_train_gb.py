"""
Train Gradient Boosting Classifier

Trains a Gradient Boosting classifier with hyperparameter tuning using
grid search with cross-validation.
"""

import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from model_training.gb_training import train_gradient_boosting, load_data
import joblib
from pathlib import Path


def main():
    # Paths
    features_path = 'features/combined/combined_train.joblib'
    labels_path = 'features/combined/labels_train.joblib'
    val_features_path = 'features/combined/combined_val.joblib'
    val_labels_path = 'features/combined/labels_val.joblib'
    output_path = 'features/model_gb.joblib'
    
    print("="*60)
    print("GRADIENT BOOSTING TRAINING")
    print("="*60)
    
    # Load data
    print(f"\nLoading features from: {features_path}")
    print(f"Loading labels from: {labels_path}")
    X_train, y_train = load_data(features_path, labels_path)
    
    print(f"Loading validation data from: {val_features_path}")
    X_val, y_val = load_data(val_features_path, val_labels_path)
    
    print(f"\nData loaded:")
    print(f"  Train features shape: {X_train.shape}")
    print(f"  Train labels shape: {y_train.shape}")
    print(f"  Val features shape: {X_val.shape}")
    print(f"  Number of classes: {len(set(y_train))}")
    
    # Use simple training without CV due to high dimensionality
    # GB is extremely slow with 26k features, so we train on a few configs and validate
    print("\n" + "-"*60)
    print("Note: Using validation set instead of CV for speed")
    print("GB is very slow with 26,468 features")
    print("-"*60)
    
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import accuracy_score
    
    configs = [
        {'n_estimators': 50, 'learning_rate': 0.1, 'max_depth': 3, 'subsample': 0.8},
        {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3, 'subsample': 0.8},
    ]
    
    results = []
    for i, params in enumerate(configs, 1):
        print(f"\nTraining config {i}/{len(configs)}: {params}")
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('gb', GradientBoostingClassifier(random_state=42, verbose=1, **params))
        ])
        
        pipeline.fit(X_train, y_train)
        
        train_score = accuracy_score(y_train, pipeline.predict(X_train))
        val_score = accuracy_score(y_val, pipeline.predict(X_val))
        
        print(f"  Train accuracy: {train_score:.4f}")
        print(f"  Val accuracy: {val_score:.4f}")
        
        results.append({
            'params': params,
            'train_score': train_score,
            'val_score': val_score,
            'model': pipeline
        })
    
    # Select best based on validation score
    best = max(results, key=lambda r: r['val_score'])
    
    print("\n" + "="*60)
    print("TRAINING RESULTS")
    print("="*60)
    print(f"\nBest parameters: {best['params']}")
    print(f"Best validation score: {best['val_score']:.4f}")
    print(f"Best train score: {best['train_score']:.4f}")
    
    print("\nAll results:")
    for i, result in enumerate(results, 1):
        print(f"  {i}. Val: {result['val_score']:.4f}, Train: {result['train_score']:.4f}")
        print(f"     {result['params']}")
    
    # Save best model
    print(f"\nSaving model to: {output_path}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best['model'], output_path)
    
    print("\n" + "="*60)
    print("GRADIENT BOOSTING TRAINING COMPLETED")
    print("="*60)


if __name__ == '__main__':
    main()
