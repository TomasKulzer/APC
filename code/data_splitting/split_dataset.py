import os
from typing import Optional, Tuple
import joblib
import numpy as np
from sklearn.model_selection import train_test_split


def stratified_split(X: np.ndarray,
                     y: np.ndarray,
                     train_frac: float = 0.6,
                     val_frac: float = 0.2,
                     test_frac: float = 0.2,
                     random_state: int = 42,
                     save_dir: Optional[str] = None
                     ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Perform a stratified split into train/val/test sets with proportions
    train_frac, val_frac, test_frac (they must sum to 1.0).

    Implementation follows a two-step split using sklearn.model_selection.train_test_split
    as requested in the task description:

    1) Split into train and temp (test_size = val_frac + test_frac)
    2) Split temp into val and test (equal proportion of temp allocated to val/test)

    Parameters:
    - X: feature array (n_samples, ...)
    - y: label array (n_samples,)
    - train_frac, val_frac, test_frac: fractions that must sum to 1.0
    - random_state: RNG seed for reproducible splits
    - save_dir: optional directory to save splits with joblib. Files saved:
        - train.joblib, val.joblib, test.joblib (each a dict with 'X' and 'y')

    Returns:
    - (X_train, y_train), (X_val, y_val), (X_test, y_test)
    """
    if not np.isclose(train_frac + val_frac + test_frac, 1.0):
        raise ValueError("train_frac + val_frac + test_frac must sum to 1.0")

    # First split: train vs temp
    test_size = 1.0 - train_frac  # fraction to hold out for val+test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Now split X_temp into validation and test
    # The proportion of X_temp that should become validation is val_frac/(val_frac+test_frac)
    temp_val_frac = val_frac / (val_frac + test_frac)

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1.0 - temp_val_frac), random_state=random_state, stratify=y_temp
    )

    # Optional save
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        joblib.dump({'X': X_train, 'y': y_train}, os.path.join(save_dir, 'train.joblib'))
        joblib.dump({'X': X_val, 'y': y_val}, os.path.join(save_dir, 'val.joblib'))
        joblib.dump({'X': X_test, 'y': y_test}, os.path.join(save_dir, 'test.joblib'))

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


if __name__ == '__main__':
    import argparse
    from feature_extraction.hog_extractor import HOGFeatureExtractor

    parser = argparse.ArgumentParser(description='Stratified split for dataset saved features.')
    parser.add_argument('--features', required=True, help='Path to saved features joblib (with keys features/labels)')
    parser.add_argument('--out_dir', default='../features/combined', help='Directory to save train/val/test splits')
    parser.add_argument('--train_frac', type=float, default=0.6)
    parser.add_argument('--val_frac', type=float, default=0.2)
    parser.add_argument('--test_frac', type=float, default=0.2)
    parser.add_argument('--random_state', type=int, default=42)

    args = parser.parse_args()

    # Load saved features file. Accepts either the hog joblib format {'features','labels'}
    data = joblib.load(args.features)
    # Support a couple of common key names
    if 'features' in data and 'labels' in data:
        X = data['features']
        y = data['labels']
    elif 'X' in data and 'y' in data:
        X = data['X']
        y = data['y']
    else:
        raise KeyError('Saved features file must contain keys (features, labels) or (X, y)')

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = stratified_split(
        X, y,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        random_state=args.random_state,
        save_dir=args.out_dir
    )

    print('Split completed:')
    print(f"Train: {X_train.shape[0]} samples")
    print(f"Val:   {X_val.shape[0]} samples")
    print(f"Test:  {X_test.shape[0]} samples")

    print(f"Saved splits to: {os.path.abspath(args.out_dir)}")
