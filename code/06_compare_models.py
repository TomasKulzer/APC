"""
Compare Integer vs Ordinal Classification Performance

Evaluates both integer and ordinal classification approaches on the test set
and provides detailed performance comparison.
"""

import sys
import os

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def decode_ordinal_predictions(y_ordinal_pred: np.ndarray) -> np.ndarray:
    """Convert ordinal predictions to class indices."""
    return np.sum(y_ordinal_pred, axis=1).astype(int)


def load_data():
    """Load test data for both integer and ordinal approaches."""
    # Integer labels
    int_features = joblib.load('features/combined/combined_test.joblib')
    int_labels = joblib.load('features/combined/labels_test.joblib')
    
    # Ordinal labels
    ord_labels = joblib.load('features/combined/labels_test_ordinal.joblib')
    
    # Encoder info for class names
    encoder_info = joblib.load('features/combined/ordinal_encoder_info.joblib')
    
    X_test = int_features.get('features', int_features)
    y_test_int = int_labels.get('labels', int_labels)
    y_test_ord = ord_labels.get('labels', ord_labels)
    
    return np.asarray(X_test), np.asarray(y_test_int), np.asarray(y_test_ord), encoder_info


def evaluate_model(model, X_test, y_test, class_names, model_name):
    """Evaluate a model and print detailed metrics."""
    print("\n" + "="*60)
    print(f"{model_name}")
    print("="*60)
    
    y_pred = model.predict(X_test)
    
    # For ordinal models, decode predictions
    if hasattr(model, 'is_ordinal') or 'ordinal' in model_name.lower():
        print("\nOrdinal predictions (first 5):")
        print(y_pred[:5])
        y_pred = decode_ordinal_predictions(y_pred)
        print(f"Decoded to classes: {y_pred[:5]}")
    
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Per-class accuracy
    print("\nPer-class accuracy:")
    for i, class_name in enumerate(class_names):
        if cm[i].sum() > 0:
            class_acc = cm[i, i] / cm[i].sum()
            print(f"  {class_name}: {class_acc:.4f} ({class_acc*100:.2f}%)")
    
    return accuracy, y_pred, cm


def main():
    print("="*60)
    print("INTEGER VS ORDINAL CLASSIFICATION COMPARISON")
    print("="*60)
    
    # Load test data
    print("\nLoading test data...")
    X_test, y_test_int, y_test_ord, encoder_info = load_data()
    
    class_names = encoder_info['class_names']
    print(f"Test samples: {X_test.shape[0]}")
    print(f"Features: {X_test.shape[1]}")
    print(f"Classes: {class_names}")
    
    # Load models
    print("\n" + "-"*60)
    print("Loading models...")
    print("-"*60)
    
    models_to_compare = []
    
    # Integer SVM
    try:
        model_data = joblib.load('features/model_svm.joblib')
        # Handle both SimpleNamespace and direct model objects
        model_svm_int = model_data.best_estimator_ if hasattr(model_data, 'best_estimator_') else model_data
        models_to_compare.append(('Integer SVM', model_svm_int, y_test_int))
        print("✓ Integer SVM loaded")
    except FileNotFoundError:
        print("✗ Integer SVM not found (features/model_svm.joblib)")
    
    # Integer RF
    try:
        model_data = joblib.load('features/model_rf.joblib')
        # Handle both SimpleNamespace and direct model objects
        model_rf_int = model_data.best_estimator_ if hasattr(model_data, 'best_estimator_') else model_data
        models_to_compare.append(('Integer Random Forest', model_rf_int, y_test_int))
        print("✓ Integer Random Forest loaded")
    except FileNotFoundError:
        print("✗ Integer Random Forest not found (features/model_rf.joblib)")
    
    # Ordinal SVM
    try:
        model_svm_ord = joblib.load('features/model_svm_ordinal.joblib')
        models_to_compare.append(('Ordinal SVM', model_svm_ord, y_test_int))  # Compare against integer labels
        print("✓ Ordinal SVM loaded")
    except FileNotFoundError:
        print("✗ Ordinal SVM not found (features/model_svm_ordinal.joblib)")
    
    # Ordinal RF
    try:
        model_rf_ord = joblib.load('features/model_rf_ordinal.joblib')
        models_to_compare.append(('Ordinal Random Forest', model_rf_ord, y_test_int))  # Compare against integer labels
        print("✓ Ordinal Random Forest loaded")
    except FileNotFoundError:
        print("✗ Ordinal Random Forest not found (features/model_rf_ordinal.joblib)")
    
    if len(models_to_compare) == 0:
        print("\nNo models found! Please train models first.")
        return
    
    # Evaluate each model
    results = {}
    for model_name, model, y_true in models_to_compare:
        accuracy, y_pred, cm = evaluate_model(model, X_test, y_true, class_names, model_name)
        results[model_name] = {
            'accuracy': accuracy,
            'predictions': y_pred,
            'confusion_matrix': cm
        }
    
    # Summary comparison
    print("\n" + "="*60)
    print("SUMMARY COMPARISON")
    print("="*60)
    
    print("\nTest Accuracy Ranking:")
    sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    for i, (name, res) in enumerate(sorted_results, 1):
        print(f"  {i}. {name}: {res['accuracy']:.4f} ({res['accuracy']*100:.2f}%)")
    
    # Compare integer vs ordinal for each classifier type
    print("\n" + "-"*60)
    print("Integer vs Ordinal Comparison:")
    print("-"*60)
    
    if 'Integer SVM' in results and 'Ordinal SVM' in results:
        int_acc = results['Integer SVM']['accuracy']
        ord_acc = results['Ordinal SVM']['accuracy']
        diff = ord_acc - int_acc
        print(f"\nSVM:")
        print(f"  Integer: {int_acc:.4f}")
        print(f"  Ordinal: {ord_acc:.4f}")
        print(f"  Difference: {diff:+.4f} ({diff*100:+.2f}%)")
        if diff > 0:
            print(f"  → Ordinal encoding improved SVM by {diff*100:.2f}%")
        elif diff < 0:
            print(f"  → Integer encoding performed better by {-diff*100:.2f}%")
        else:
            print(f"  → No difference")
    
    if 'Integer Random Forest' in results and 'Ordinal Random Forest' in results:
        int_acc = results['Integer Random Forest']['accuracy']
        ord_acc = results['Ordinal Random Forest']['accuracy']
        diff = ord_acc - int_acc
        print(f"\nRandom Forest:")
        print(f"  Integer: {int_acc:.4f}")
        print(f"  Ordinal: {ord_acc:.4f}")
        print(f"  Difference: {diff:+.4f} ({diff*100:+.2f}%)")
        if diff > 0:
            print(f"  → Ordinal encoding improved RF by {diff*100:.2f}%")
        elif diff < 0:
            print(f"  → Integer encoding performed better by {-diff*100:.2f}%")
        else:
            print(f"  → No difference")
    
    print("\n" + "="*60)
    print("COMPARISON COMPLETED")
    print("="*60)


if __name__ == '__main__':
    main()
