#!/usr/bin/env python3
"""
01 - Prepare & inspect dataset

Usage:
    python code/01_prepare_data.py

This script lists classes, counts images per class and prints a sample image shape.
"""
import os
import sys
from pathlib import Path

# Ensure `code/` is on sys.path so imports work when running from repo root
sys.path.insert(0, os.path.dirname(__file__))

from data_loading_and_preprocessing.image_loader import ImageLoader
from sklearn.model_selection import train_test_split
import joblib
import numpy as np
from tqdm import tqdm
import cv2


def main(data_dir: str = 'dataset', n_augmentations: int = 3):
    """
    Prepare dataset splits and optionally generate augmented training images.
    
    Parameters:
    - data_dir: Root directory containing class subdirectories
    - n_augmentations: Number of augmented versions to generate per training image (0 = no augmentation)
    """
    if not os.path.isdir(data_dir):
        print(f"Dataset directory '{data_dir}' not found.")
        return

    loader = ImageLoader(data_dir, image_size=(224, 224), mode='train', augment=False)

    print('Classes found:')
    for name in loader.class_names:
        print(f" - {name}")

    # Count per class
    counts = {name: 0 for name in loader.class_names}
    for p, lbl in zip(loader.image_paths, loader.labels):
        counts[loader.class_names[lbl]] += 1

    print('\nImage counts per class:')
    for k, v in counts.items():
        print(f" - {k}: {v}")

    total = len(loader)
    print(f"\nTotal samples: {total}")
    if total > 0:
        img, lbl = loader[0]
        print(f"Sample image shape (H,W,C): {img.shape}, sample label index: {lbl}")

    # Prepare stratified splits (60% train, 20% val, 20% test)
    X = np.array(loader.image_paths)
    y = np.array(loader.labels)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    # Store relative paths with respect to data_dir
    # If loader.image_paths are absolute, make them relative
    def rel_paths(paths):
        rel = []
        for p in paths:
            if os.path.isabs(p):
                try:
                    rel.append(os.path.relpath(p, data_dir))
                except Exception:
                    rel.append(p)
            else:
                rel.append(p)
        return rel

    # Generate augmented training images if requested
    if n_augmentations > 0:
        print(f"\n=== Generating {n_augmentations} augmented versions per training image ===")
        aug_dir = os.path.join(data_dir, 'augmented')
        os.makedirs(aug_dir, exist_ok=True)
        
        # Create subdirectories for each class
        for class_name in loader.class_names:
            os.makedirs(os.path.join(aug_dir, class_name), exist_ok=True)
        
        # Create augmentation loader
        aug_loader = ImageLoader(data_dir, image_size=(224, 224), mode='train', augment=True)
        
        augmented_paths = []
        augmented_labels = []
        
        for idx, (path, label) in enumerate(tqdm(zip(X_train, y_train), total=len(X_train), desc="Augmenting training images")):
            class_name = loader.class_names[label]
            # Original filename
            orig_filename = os.path.splitext(os.path.basename(path))[0]
            
            # Path is already absolute from loader.image_paths
            full_path = path
            
            image = aug_loader.load_image(full_path)
            
            # Generate n_augmentations versions
            for aug_idx in range(n_augmentations):
                # Apply augmentation
                aug_image = aug_loader._apply_augmentations(image)
                
                # Save augmented image
                aug_filename = f"{orig_filename}_aug{aug_idx}.jpg"
                aug_path = os.path.join(aug_dir, class_name, aug_filename)
                
                # Convert RGB to BGR for cv2.imwrite
                aug_image_bgr = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(aug_path, aug_image_bgr)
                
                # Store relative path
                rel_aug_path = os.path.relpath(aug_path, data_dir)
                augmented_paths.append(rel_aug_path)
                augmented_labels.append(label)
        
        print(f"Generated {len(augmented_paths)} augmented images in {aug_dir}")
        
        # Combine original training data with augmented data
        train_paths_combined = rel_paths(X_train.tolist()) + augmented_paths
        train_labels_combined = y_train.tolist() + augmented_labels
        
        print(f"Total training samples: {len(train_paths_combined)} (original: {len(X_train)}, augmented: {len(augmented_paths)})")
    else:
        train_paths_combined = rel_paths(X_train.tolist())
        train_labels_combined = y_train.tolist()
    
    splits = {
        'train_paths': train_paths_combined,
        'train_labels': train_labels_combined,
        'val_paths': rel_paths(X_val.tolist()),
        'val_labels': y_val.tolist(),
        'test_paths': rel_paths(X_test.tolist()),
        'test_labels': y_test.tolist(),
        'class_names': loader.class_names,  # Save class names for ordinal encoding
    }

    splits_file = os.path.join(data_dir, 'splits.joblib')
    joblib.dump(splits, splits_file)
    print(f"Saved dataset splits to: {splits_file}")
    print(f"Class order (for ordinal encoding): {loader.class_names}")


if __name__ == '__main__':
    main()
