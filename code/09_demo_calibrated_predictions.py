"""
Demo: Calibrated Gradient Boosting Predictions with Confidence Scores

Shows predictions for 4 test samples from each class using Gradient Boosting
with temperature scaling calibration. Displays actual images with confidence scores.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import joblib
from confidence_estimation import apply_temperature_scaling, get_logits_from_model
from scipy.special import softmax
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path


def plot_samples_with_predictions(image_paths, predictions, true_labels, probs_calibrated, 
                                   class_names, output_path):
    """
    Plot sample images with their predictions and confidence scores.
    
    Args:
        image_paths: List of image file paths
        predictions: Predicted class indices
        true_labels: True class indices
        probs_calibrated: Calibrated probabilities
        class_names: List of class names
        output_path: Where to save the plot
    """
    n_samples = len(image_paths)
    cols = 4
    rows = (n_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (img_path, pred, true, probs) in enumerate(zip(image_paths, predictions, true_labels, probs_calibrated)):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]
        
        # Load and display image
        try:
            img = Image.open(img_path)
            ax.imshow(img)
        except:
            ax.text(0.5, 0.5, 'Image not found', ha='center', va='center')
        
        ax.axis('off')
        
        # Get prediction info
        pred_name = class_names[pred]
        true_name = class_names[true]
        confidence = probs[pred]
        
        # Determine if correct
        is_correct = pred == true
        color = 'green' if is_correct else 'red'
        status = '✓' if is_correct else '✗'
        
        # Create title with prediction and confidence
        title = f'{status} Pred: {pred_name}\n'
        title += f'True: {true_name}\n'
        title += f'Confidence: {confidence:.1%}'
        
        ax.set_title(title, fontsize=10, color=color, weight='bold')
        
        # Add confidence bar for all classes
        textstr = '\n'.join([f'{name}: {prob:.1%}' for name, prob in zip(class_names, probs)])
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', bbox=props)
    
    # Hide empty subplots
    for idx in range(n_samples, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to: {output_path}")


def main():
    print("="*80)
    print("CALIBRATED GRADIENT BOOSTING - TEST SAMPLE PREDICTIONS")
    print("="*80)
    
    # Load test data and splits
    print("\nLoading test data...")
    X_test = joblib.load('features/combined/combined_test.joblib')
    y_test = joblib.load('features/combined/labels_test.joblib')
    splits = joblib.load('dataset/splits.joblib')
    
    X_test = np.asarray(X_test.get('features', X_test))
    y_test = np.asarray(y_test.get('labels', y_test))
    
    # Get test image paths
    test_paths = splits['test_paths']
    test_labels = splits['test_labels']
    
    # Load class names
    encoder_info = joblib.load('features/combined/ordinal_encoder_info.joblib')
    class_names = encoder_info['class_names']
    
    # Load model
    print("Loading Gradient Boosting model...")
    model_data = joblib.load('features/model_gb.joblib')
    model = model_data.best_estimator_ if hasattr(model_data, 'best_estimator_') else model_data
    
    # Load temperature from calibration results
    print("Loading temperature scaling calibration...")
    import json
    with open('evaluation_results/temperature_scaling_results.json', 'r') as f:
        calibration_results = json.load(f)
    temperature = calibration_results['Gradient Boosting']['temperature']
    
    print(f"Using temperature: T = {temperature:.4f}")
    
    # Get logits and probabilities
    print("\nComputing predictions...")
    logits = get_logits_from_model(model, X_test)
    
    # Uncalibrated probabilities
    probs_uncalibrated = softmax(logits, axis=1)
    
    # Calibrated probabilities using temperature scaling
    probs_calibrated = apply_temperature_scaling(model, X_test, temperature)
    
    # Get predictions
    predictions = np.argmax(logits, axis=1)
    
    # Create output directory
    viz_dir = Path('visualizations/calibration')
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Select 4 samples from each class and plot
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS FOR TEST SAMPLES (4 per class)")
    print("="*80)
    
    for class_idx, class_name in enumerate(class_names):
        # Find indices of this class in test set
        class_indices = np.where(y_test == class_idx)[0]
        
        # Select first 4 samples (or all if fewer than 4)
        sample_indices = class_indices[:min(4, len(class_indices))]
        
        print(f"\n{'='*80}")
        print(f"TRUE CLASS: {class_name.upper()}")
        print('='*80)
        
        # Collect data for plotting
        sample_paths = [test_paths[idx] for idx in sample_indices]
        sample_preds = [predictions[idx] for idx in sample_indices]
        sample_true = [y_test[idx] for idx in sample_indices]
        sample_probs = [probs_calibrated[idx] for idx in sample_indices]
        
        # Create visualization for this class
        output_path = viz_dir / f'predictions_{class_name.lower()}.png'
        plot_samples_with_predictions(
            sample_paths, sample_preds, sample_true, sample_probs,
            class_names, output_path
        )
        
        for sample_num, idx in enumerate(sample_indices, 1):
            predicted_class = predictions[idx]
            predicted_name = class_names[predicted_class]
            
            # Get confidence scores
            uncalib_confidence = probs_uncalibrated[idx, predicted_class]
            calib_confidence = probs_calibrated[idx, predicted_class]
            
            # Check if prediction is correct
            is_correct = "✓ CORRECT" if predicted_class == class_idx else "✗ INCORRECT"
            
            print(f"\nSample {sample_num} (Test Index: {idx}):")
            print(f"  Image: {test_paths[idx]}")
            print(f"  Predicted Class: {predicted_name}")
            print(f"  Status: {is_correct}")
            print(f"  Uncalibrated Confidence: {uncalib_confidence:.4f} ({uncalib_confidence*100:.2f}%)")
            print(f"  Calibrated Confidence:   {calib_confidence:.4f} ({calib_confidence*100:.2f}%)")
    
    # Create combined visualization with all classes
    print("\n" + "="*80)
    print("Creating combined visualization...")
    print("="*80)
    
    all_sample_indices = []
    for class_idx in range(len(class_names)):
        class_indices = np.where(y_test == class_idx)[0]
        all_sample_indices.extend(class_indices[:min(4, len(class_indices))])
    
    combined_paths = [test_paths[idx] for idx in all_sample_indices]
    combined_preds = [predictions[idx] for idx in all_sample_indices]
    combined_true = [y_test[idx] for idx in all_sample_indices]
    combined_probs = [probs_calibrated[idx] for idx in all_sample_indices]
    
    combined_output = viz_dir / 'predictions_all_classes.png'
    plot_samples_with_predictions(
        combined_paths, combined_preds, combined_true, combined_probs,
        class_names, combined_output
    )
    
    # Summary statistics
    print("\n" + "="*80)
    print("OVERALL TEST SET STATISTICS")
    print("="*80)
    
    accuracy = np.mean(predictions == y_test)
    avg_uncalib_conf = np.mean(np.max(probs_uncalibrated, axis=1))
    avg_calib_conf = np.mean(np.max(probs_calibrated, axis=1))
    
    print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Average Uncalibrated Confidence: {avg_uncalib_conf:.4f} ({avg_uncalib_conf*100:.2f}%)")
    print(f"Average Calibrated Confidence:   {avg_calib_conf:.4f} ({avg_calib_conf*100:.2f}%)")
    
    # Per-class accuracy
    print(f"\nPer-Class Accuracy:")
    for class_idx, class_name in enumerate(class_names):
        class_mask = y_test == class_idx
        class_acc = np.mean(predictions[class_mask] == y_test[class_mask])
        class_count = np.sum(class_mask)
        print(f"  {class_name:15s}: {class_acc:.4f} ({class_acc*100:.2f}%) - {class_count} samples")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
