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


def main(data_dir: str = 'dataset'):
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

    splits = {
        'train_paths': rel_paths(X_train.tolist()),
        'train_labels': y_train.tolist(),
        'val_paths': rel_paths(X_val.tolist()),
        'val_labels': y_val.tolist(),
        'test_paths': rel_paths(X_test.tolist()),
        'test_labels': y_test.tolist(),
    }

    splits_file = os.path.join(data_dir, 'splits.joblib')
    joblib.dump(splits, splits_file)
    print(f"Saved dataset splits to: {splits_file}")


if __name__ == '__main__':
    main()
