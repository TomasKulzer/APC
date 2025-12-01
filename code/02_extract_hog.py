#!/usr/bin/env python3
"""
02 - Extract HOG features for all images

Usage:
    python code/02_extract_hog.py --data dataset --out ../features/hog_features.joblib

Defaults assume you run from repo root (script adds `code/` to sys.path automatically).
"""
import os
import sys
import joblib

# Ensure `code/` is on sys.path so imports work when running from repo root
sys.path.insert(0, os.path.dirname(__file__))

from feature_extraction.hog_extractor import HOGFeatureExtractor


def main():
    data_dir = 'dataset'
    splits_file = os.path.join(data_dir, 'splits.joblib')
    out_dir = 'features'
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(splits_file):
        raise FileNotFoundError(f"Splits file not found: {splits_file}. Run 01_prepare_data.py first to create splits.")

    splits = joblib.load(splits_file)
    extractor = HOGFeatureExtractor()

    for split_name in ('train', 'val', 'test'):
        rel_paths = splits[f'{split_name}_paths']
        labels = splits[f'{split_name}_labels']
        save_path = os.path.join(out_dir, f'hog_{split_name}.joblib')
        print(f"Extracting HOG for split '{split_name}' ({len(rel_paths)} samples) -> {save_path}")
        extractor.process_dataset(data_path=data_dir, save_path=save_path, batch_size=32, image_paths=rel_paths, labels=labels, split_name=split_name)
        print('')


if __name__ == '__main__':
    main()
