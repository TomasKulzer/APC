#!/usr/bin/env python3
"""
05 - Visualize features (HOG, Color Histogram, SIFT) for the same sample

Usage:
    python code/05_visualize.py --sample-idx 0

This creates visualizations in visualizations/features/ directory.
"""
import os
import sys
import argparse
import joblib
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

sys.path.insert(0, os.path.dirname(__file__))

from feature_extraction.hog_extractor import HOGFeatureExtractor
from feature_extraction.color_histogram import ColorHistogramExtractor
from feature_extraction.sift import SIFTBagOfWords


def visualize_all_features(sample_idx: int = 0, output_dir: str = 'visualizations/features'):
    """
    Visualize HOG, Color Histogram, and SIFT features for the same sample.
    
    Args:
        sample_idx: Index of the sample to visualize from the training set
        output_dir: Base directory for saving visualizations
    """
    # Load splits to get sample paths
    splits = joblib.load('Dataset/splits.joblib')
    train_paths = splits['train_paths']
    train_labels = splits['train_labels']
    
    if sample_idx >= len(train_paths):
        print(f"Sample index {sample_idx} out of range (max: {len(train_paths)-1})")
        return
    
    # Get the sample path
    sample_path = train_paths[sample_idx]
    # Handle paths that may already include dataset prefix
    if sample_path.startswith('Dataset' + os.sep):
        # Path already has dataset prefix, use as-is
        pass
    elif os.path.isabs(sample_path):
        # Absolute path, use as-is
        pass
    else:
        # Relative path without dataset prefix
        sample_path = os.path.join('Dataset', sample_path)
    
    sample_label = train_labels[sample_idx]
    
    print(f"Visualizing sample {sample_idx}: {sample_path}")
    print(f"Label: {sample_label}")
    
    # Create output directories
    os.makedirs(os.path.join(output_dir, 'hog'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'color_hist'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'sift'), exist_ok=True)
    
    # Read image
    img_bgr = cv2.imread(sample_path)
    if img_bgr is None:
        print(f"Failed to load image: {sample_path}")
        return
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # === HOG Visualization ===
    print("Creating HOG visualization...")
    from skimage.feature import hog
    from skimage import color
    
    img_gray_ski = color.rgb2gray(img_rgb)
    hog_features, hog_image = hog(
        img_gray_ski,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        visualize=True,
        feature_vector=True
    )
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.imshow(img_rgb)
    ax1.set_title(f'Original Image (Label: {sample_label})')
    ax1.axis('off')
    
    ax2.imshow(hog_image, cmap='gray')
    ax2.set_title('HOG Features')
    ax2.axis('off')
    
    plt.tight_layout()
    hog_path = os.path.join(output_dir, 'hog', f'sample_{sample_idx}_hog.png')
    plt.savefig(hog_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {hog_path}")
    
    # === Color Histogram Visualization ===
    print("Creating Color Histogram visualization...")
    color_extractor = ColorHistogramExtractor(bins_per_channel=32)
    color_hist = color_extractor.extract_histogram(img_bgr)
    
    bins = color_extractor.bins
    h = color_hist[0:bins]
    s = color_hist[bins:2*bins]
    v = color_hist[2*bins:3*bins]
    
    fig = plt.figure(figsize=(14, 5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.5])
    
    # Left: image
    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(img_rgb)
    ax_img.set_title(f'Original Image (Label: {sample_label})')
    ax_img.axis('off')
    
    # Right: HSV histograms stacked
    ax_hist = fig.add_subplot(gs[0, 1])
    ax_hist.axis('off')
    
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    
    # Create color bars for hue
    bin_edges_h = np.linspace(0, 180, bins + 1)
    bin_centers_h = 0.5 * (bin_edges_h[:-1] + bin_edges_h[1:])
    hsv_vals = np.stack([bin_centers_h / 180.0, np.ones(bins), np.ones(bins)], axis=1)
    colors_h = hsv_to_rgb(hsv_vals)
    
    ax_h = inset_axes(ax_hist, width='100%', height='28%', bbox_to_anchor=(0, 0.70, 1, 0.28), bbox_transform=ax_hist.transAxes)
    ax_h.bar(np.arange(bins), h, color=colors_h, width=1.0)
    ax_h.set_title('Hue Histogram', fontsize=10)
    ax_h.set_xlim(-0.5, bins - 0.5)
    ax_h.tick_params(labelsize=8)
    
    ax_s = inset_axes(ax_hist, width='100%', height='28%', bbox_to_anchor=(0, 0.36, 1, 0.28), bbox_transform=ax_hist.transAxes)
    ax_s.bar(np.arange(bins), s, color='tab:orange', width=1.0)
    ax_s.set_title('Saturation Histogram', fontsize=10)
    ax_s.set_xlim(-0.5, bins - 0.5)
    ax_s.tick_params(labelsize=8)
    
    ax_v = inset_axes(ax_hist, width='100%', height='28%', bbox_to_anchor=(0, 0.02, 1, 0.28), bbox_transform=ax_hist.transAxes)
    ax_v.bar(np.arange(bins), v, color='tab:gray', width=1.0)
    ax_v.set_title('Value Histogram', fontsize=10)
    ax_v.set_xlim(-0.5, bins - 0.5)
    ax_v.tick_params(labelsize=8)
    
    plt.tight_layout()
    color_path = os.path.join(output_dir, 'color_hist', f'sample_{sample_idx}_colorhist.png')
    plt.savefig(color_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {color_path}")
    
    # === SIFT Visualization ===
    print("Creating SIFT visualization...")
    sift_bow = SIFTBagOfWords(k=128)
    keypoints, descriptors = sift_bow.sift.detectAndCompute(img_gray, None)
    
    # Draw keypoints
    img_kp = cv2.drawKeypoints(img_bgr, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img_kp_rgb = cv2.cvtColor(img_kp, cv2.COLOR_BGR2RGB)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.imshow(img_kp_rgb)
    ax.set_title(f'SIFT Keypoints ({len(keypoints)} detected) - Label: {sample_label}')
    ax.axis('off')
    
    plt.tight_layout()
    sift_path = os.path.join(output_dir, 'sift', f'sample_{sample_idx}_sift.png')
    plt.savefig(sift_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {sift_path}")
    
    print(f"\nAll visualizations saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Visualize features for a sample')
    parser.add_argument('--sample-idx', type=int, default=0, help='Index of sample to visualize from training set')
    parser.add_argument('--output-dir', default='visualizations/features', help='Output directory for visualizations')
    args = parser.parse_args()
    
    visualize_all_features(args.sample_idx, args.output_dir)


if __name__ == '__main__':
    main()
