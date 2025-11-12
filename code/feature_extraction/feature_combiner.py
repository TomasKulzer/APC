import os
import argparse
from typing import Optional, Tuple

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_features_maybe(path: str) -> Tuple[np.ndarray, Optional[list]]:
    """Load a features file saved by joblib or a raw numpy array.

    Returns (features, labels_or_none)
    """
    data = joblib.load(path)
    if isinstance(data, dict):
        # prefer explicit keys; avoid using truthiness on arrays
        if 'features' in data:
            feats = data['features']
        elif 'X' in data:
            feats = data['X']
        else:
            # fallback: maybe the dict itself is the features
            feats = data

        if 'labels' in data:
            labels = data['labels']
        elif 'y' in data:
            labels = data['y']
        else:
            labels = None
        return np.asarray(feats), labels
    else:
        return np.asarray(data), None


def combine_feature_sets(list_of_feature_arrays):
    # Ensure all are 2D and same number of rows
    arrays = [np.asarray(a) for a in list_of_feature_arrays if a is not None]
    if not arrays:
        raise ValueError('No feature arrays provided to combine')
    n = arrays[0].shape[0]
    for a in arrays:
        if a.shape[0] != n:
            raise ValueError('All feature arrays must have same number of samples (rows)')
    return np.hstack(arrays)


def stratified_split(X, y, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42):
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError('train/val/test sizes must sum to 1.0')

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=train_size, stratify=y, random_state=random_state)
    # split temp into val and test (relative sizes)
    rel_val = val_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, train_size=rel_val, stratify=y_temp, random_state=random_state)
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def fit_and_apply_scaler(X_train: np.ndarray, Xs: dict, out_dir: str, scaler_name: str = 'scaler.joblib'):
    scaler = StandardScaler()
    scaler.fit(X_train)
    os.makedirs(out_dir, exist_ok=True)
    scaler_path = os.path.join(out_dir, scaler_name)
    joblib.dump(scaler, scaler_path)

    transformed = {}
    for k, X in Xs.items():
        transformed[k] = scaler.transform(X)
        joblib.dump({'features': transformed[k]}, os.path.join(out_dir, f'combined_{k}.joblib'))

    return scaler_path, transformed


def main():
    parser = argparse.ArgumentParser(description='Combine feature sets and normalize with StandardScaler')
    parser.add_argument('--hog', help='Path to hog features joblib', default='../features/hog_features.joblib')
    parser.add_argument('--sift', help='Path to sift bow features joblib (optional)', default=None)
    parser.add_argument('--color', help='Path to color hist features joblib', default='../features/color_hist_features.joblib')
    parser.add_argument('--labels', help='Optional labels file (joblib) if not embedded in features')
    parser.add_argument('--out-dir', help='Directory to save combined features and scaler', default='../features/combined')
    parser.add_argument('--train-size', type=float, default=0.7)
    parser.add_argument('--val-size', type=float, default=0.15)
    parser.add_argument('--test-size', type=float, default=0.15)
    parser.add_argument('--random-state', type=int, default=42)
    args = parser.parse_args()

    feats_list = []
    labels = None

    if args.hog and os.path.exists(args.hog):
        hog_feats, hog_labels = load_features_maybe(args.hog)
        feats_list.append(hog_feats)
        if labels is None and hog_labels is not None:
            labels = hog_labels

    if args.sift and os.path.exists(args.sift):
        sift_feats, sift_labels = load_features_maybe(args.sift)
        feats_list.append(sift_feats)
        if labels is None and sift_labels is not None:
            labels = sift_labels

    if args.color and os.path.exists(args.color):
        color_feats, color_labels = load_features_maybe(args.color)
        feats_list.append(color_feats)
        if labels is None and color_labels is not None:
            labels = color_labels

    if not feats_list:
        raise SystemExit('No feature files found. Provide at least one of --hog, --sift, --color')

    X = combine_feature_sets(feats_list)
    labels = labels if labels is not None else None

    os.makedirs(args.out_dir, exist_ok=True)
    # Save raw combined features
    joblib.dump({'features': X, 'labels': labels}, os.path.join(args.out_dir, 'combined_raw.joblib'))

    # If labels are present, do stratified split and fit scaler on train only
    if labels is not None:
        y = np.asarray(labels)
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = stratified_split(X, y, train_size=args.train_size, val_size=args.val_size, test_size=args.test_size, random_state=args.random_state)
        scaler_path, transformed = fit_and_apply_scaler(X_train, {'train': X_train, 'val': X_val, 'test': X_test}, args.out_dir)
        # Save labels for each split
        joblib.dump({'labels': y_train}, os.path.join(args.out_dir, 'labels_train.joblib'))
        joblib.dump({'labels': y_val}, os.path.join(args.out_dir, 'labels_val.joblib'))
        joblib.dump({'labels': y_test}, os.path.join(args.out_dir, 'labels_test.joblib'))
        print('Saved scaler to', scaler_path)
        print('Saved transformed splits to', args.out_dir)
    else:
        # No labels: fit scaler on entire set (user should be careful)
        scaler = StandardScaler()
        scaler.fit(X)
        scaler_path = os.path.join(args.out_dir, 'scaler.joblib')
        joblib.dump(scaler, scaler_path)
        X_trans = scaler.transform(X)
        joblib.dump({'features': X_trans}, os.path.join(args.out_dir, 'combined_scaled.joblib'))
        print('No labels provided: fitted scaler on all data and saved to', scaler_path)


if __name__ == '__main__':
    main()
