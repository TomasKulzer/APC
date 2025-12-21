#!/usr/bin/env python3
"""
03 - Combine per-split features and scale

This script expects per-split feature files produced by the extractors:
  - ../features/hog_train.joblib, hog_val.joblib, hog_test.joblib
  - ../features/colorhist_train.joblib, ...
  - ../features/sift_train.joblib, ...

It combines available feature types and saves combined scaled splits into ../features/combined/
Also creates ordinal-encoded labels for ordinal classification.

Run from repo root:
    python code/03_combine_features.py
"""
import os
import sys
import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from feature_extraction.feature_combiner import load_features_maybe, combine_feature_sets, fit_and_apply_scaler
from data_loading_and_preprocessing.ordinal_encoding import OrdinalEncoder


def load_split_features(base_dir, prefix):
    """Load features saved by extractors (expects joblib files with 'features' and 'labels').
    Returns (features, labels)
    """
    if not os.path.exists(base_dir):
        raise FileNotFoundError(base_dir)
    data = joblib.load(base_dir)
    if isinstance(data, dict) and 'features' in data:
        feats = np.asarray(data['features'])
        labels = data.get('labels', None)
        return feats, labels
    else:
        return np.asarray(data), None


def main():
    feat_dir = 'features'
    combined_out = os.path.join('features', 'combined')
    os.makedirs(combined_out, exist_ok=True)

    # For each split, gather available feature arrays
    splits = ['train', 'val', 'test']
    # feature types and file name patterns
    types = [('hog', os.path.join(feat_dir, 'hog_{}.joblib')),
             ('color', os.path.join(feat_dir, 'colorhist_{}.joblib')),
             ('sift', os.path.join(feat_dir, 'sift_{}.joblib'))]

    # We'll build lists per split
    per_split_feats = {s: [] for s in splits}
    per_split_labels = {s: None for s in splits}

    for name, pattern in types:
        for s in splits:
            path = pattern.format(s)
            if os.path.exists(path):
                data = joblib.load(path)
                feats = data.get('features') if isinstance(data, dict) and 'features' in data else data
                lbls = data.get('labels') if isinstance(data, dict) and 'labels' in data else None
                per_split_feats[s].append(np.asarray(feats))
                if per_split_labels[s] is None and lbls is not None:
                    per_split_labels[s] = lbls
            else:
                print(f'Warning: feature file not found for {name}/{s}: {path} (skipping)')

    # Ensure we have at least one feature per split
    for s in splits:
        if not per_split_feats[s]:
            raise SystemExit(f'No feature arrays found for split {s}. Aborting combine.')

    # Combine features horizontally per split
    X_train = combine_feature_sets([a for a in per_split_feats['train']])
    X_val = combine_feature_sets([a for a in per_split_feats['val']])
    X_test = combine_feature_sets([a for a in per_split_feats['test']])

    # Save raw combined versions
    joblib.dump({'features': X_train, 'labels': per_split_labels['train']}, os.path.join(combined_out, 'combined_train_raw.joblib'))
    joblib.dump({'features': X_val, 'labels': per_split_labels['val']}, os.path.join(combined_out, 'combined_val_raw.joblib'))
    joblib.dump({'features': X_test, 'labels': per_split_labels['test']}, os.path.join(combined_out, 'combined_test_raw.joblib'))

    # Fit scaler on train only and save scaled splits
    scaler_path, transformed = fit_and_apply_scaler(X_train, {'train': X_train, 'val': X_val, 'test': X_test}, combined_out)
    
    # Load splits to get class names for ordinal encoding
    splits_data = joblib.load('Dataset/splits.joblib')

    
    # Save integer labels (original)
    joblib.dump({'labels': per_split_labels['train']}, os.path.join(combined_out, 'labels_train.joblib'))
    joblib.dump({'labels': per_split_labels['val']}, os.path.join(combined_out, 'labels_val.joblib'))
    joblib.dump({'labels': per_split_labels['test']}, os.path.join(combined_out, 'labels_test.joblib'))
    
    # Create and save ordinal labels
  
    

    

    
    print('Saved combined and scaled features to', combined_out)
    print('Scaler:', scaler_path)


if __name__ == '__main__':
    main()
