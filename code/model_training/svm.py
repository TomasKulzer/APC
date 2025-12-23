"""
Ordinal SVM using threshold-based approach for ordinal regression.

This module implements ordinal regression using multiple binary SVMs,
one for each threshold in the ordinal scale. This approach models
the cumulative probabilities P(y > k) for each threshold k.
"""

import numpy as np
import joblib
from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Union, Dict, Any, List
from tqdm import tqdm


def load_data(features_path: str, labels_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load features and labels from disk using joblib.
    
    Parameters:
    -----------
    features_path : str
        Path to the joblib file containing features.
    labels_path : str
        Path to the joblib file containing labels.
    
    Returns:
    --------
    X : np.ndarray
        2D array of features with shape (n_samples, n_features).
    y : np.ndarray
        1D array of ordinal labels (integers 0-4).
    """
    # Load features
    features_data = joblib.load(features_path)
    if isinstance(features_data, dict):
        X = features_data.get('features', features_data.get('X'))
    else:
        X = features_data
    
    # Load labels
    labels_data = joblib.load(labels_path)
    if isinstance(labels_data, dict):
        y = labels_data.get('labels', labels_data.get('y'))
    else:
        y = labels_data
    
    # Ensure numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Flatten y if needed
    if y.ndim > 1:
        y = y.ravel()
    
    return X, y


def ordinal_encode_thresholds(y: np.ndarray) -> np.ndarray:
    """
    Convert ordinal labels to binary threshold matrix.
    
    For each sample, creates binary indicators for each threshold:
    - Column 0: 1 if y > 0 (i.e., not class 0)
    - Column 1: 1 if y > 1 (i.e., class 2, 3, or 4)
    - Column 2: 1 if y > 2 (i.e., class 3 or 4)
    - Column 3: 1 if y > 3 (i.e., class 4)
    
    Parameters:
    -----------
    y : np.ndarray
        1D array of ordinal labels (0-4).
    
    Returns:
    --------
    y_binary : np.ndarray
        2D array of shape (n_samples, 4) with binary threshold indicators.
    """
    n_samples = len(y)
    n_thresholds = 4  # For 5 classes (0-4), we have 4 thresholds
    
    y_binary = np.zeros((n_samples, n_thresholds), dtype=int)
    
    for i in range(n_thresholds):
        # Threshold i: 1 if y > i, 0 otherwise
        y_binary[:, i] = (y > i).astype(int)
    
    return y_binary


def decode_svm_predictions(probas: np.ndarray) -> np.ndarray:
    """
    Correct threshold decoding: find class interval where P(y > k) transitions from 1 to 0.
    
    This is the standard decoding method for ordinal SVMs. For each sample,
    we find the first threshold k where P(y > k) < 0.5, which indicates
    the predicted class is k.
    
    Parameters:
    -----------
    probas : np.ndarray
        Array of shape (n_samples, 4) containing probabilities P(y > k)
        for each threshold k.
    
    Returns:
    --------
    y_pred : np.ndarray
        1D array of predicted ordinal classes (0-4).
    """
    n_samples = probas.shape[0]
    y_pred = np.zeros(n_samples, dtype=int)
    
    for i in range(n_samples):
        # Find first threshold where P(y > k) < 0.5 (crossing point)
        for k in range(probas.shape[1]):
            if probas[i, k] < 0.5:
                y_pred[i] = k
                break
        else:
            # All P(y > k) >= 0.5 → highest class
            y_pred[i] = probas.shape[1]
    
    # Clip to valid range [0, 4]
    return np.clip(y_pred, 0, 4)


class OrdinalSVMClassifier:
    """
    Ordinal SVM classifier using threshold approach.
    
    Trains multiple binary SVM classifiers, one for each threshold,
    and combines their predictions to make ordinal predictions.
    """
    
    def __init__(self, C: float = 1.0, gamma: Union[str, float] = 'scale', 
                 kernel: str = 'rbf', max_iter: int = 500,
                 use_linear_svc: bool = True,
                 verbose: int = 0, random_state: int = 42):
        """
        Initialize the ordinal SVM classifier.
        
        Parameters:
        -----------
        C : float, default=1.0
            Regularization parameter for SVM.
        gamma : str or float, default='scale'
            Kernel coefficient for RBF kernel.
        kernel : str, default='rbf'
            Kernel type ('rbf' or 'linear'). Linear is much faster.
        max_iter : int, default=500
            Maximum iterations for SVM solver. Use -1 for no limit.
        use_linear_svc : bool, default=True
            If True and kernel='linear', use LinearSVC (much faster) instead of SVC.
        verbose : int, default=0
            Verbosity level.
        random_state : int, default=42
            Random state for reproducibility.
        """
        self.C = C
        self.gamma = gamma
        self.kernel = kernel
        self.max_iter = max_iter
        self.use_linear_svc = use_linear_svc
        self.verbose = verbose
        self.random_state = random_state
        self.pipelines = []
        self.n_thresholds = 4
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the ordinal SVM model.
        
        Parameters:
        -----------
        X : np.ndarray
            Training features, shape (n_samples, n_features).
        y : np.ndarray
            Training labels, shape (n_samples,) with values 0-4.
        
        Returns:
        --------
        self
        """
        # Encode labels to threshold format
        y_binary = ordinal_encode_thresholds(y)
        
        # Train one SVM for each threshold
        self.pipelines = []
        
        # Always show progress bar for training
        print("Training SVM thresholds...")
        for i in tqdm(range(self.n_thresholds), desc="SVM thresholds", unit="threshold"):
            if self.verbose > 0:
                tqdm.write(f"  Training threshold {i}/4 (y>{i})...")
            
            # Create pipeline with scaler and SVM
            # Use LinearSVC for linear kernel (much faster)
            if self.kernel == 'linear' and self.use_linear_svc:
                # LinearSVC is much faster for linear kernel
                svc = LinearSVC(
                    C=self.C,
                    max_iter=self.max_iter if self.max_iter > 0 else 1000,
                    random_state=self.random_state,
                    dual='auto',  # Automatically choose primal/dual
                    class_weight='balanced',  # Handle class imbalance
                    verbose=0
                )
                # Calibrate for probabilities with better CV for improved P(y > k) estimates
                svc = CalibratedClassifierCV(svc, cv=5, n_jobs=-1)
            else:
                # Use standard SVC for RBF or if LinearSVC disabled
                svc_params = {
                    'kernel': self.kernel,
                    'C': self.C,
                    'probability': True,
                    'random_state': self.random_state,
                    'cache_size': 1000,  # Increase cache even more
                    'class_weight': 'balanced',  # Handle class imbalance
                    'verbose': 0
                }
                
                # Add max_iter if specified
                if self.max_iter > 0:
                    svc_params['max_iter'] = self.max_iter
                
                # Add gamma only for non-linear kernels
                if self.kernel != 'linear':
                    svc_params['gamma'] = self.gamma
                
                svc = SVC(**svc_params)
            
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('svm', svc)
            ])
            
            # Fit on binary labels for this threshold
            pipeline.fit(X, y_binary[:, i])
            self.pipelines.append(pipeline)
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict threshold probabilities.
        
        Parameters:
        -----------
        X : np.ndarray
            Features to predict, shape (n_samples, n_features).
        
        Returns:
        --------
        probas : np.ndarray
            Threshold probabilities, shape (n_samples, 4).
            probas[:, i] = P(y > i)
        """
        n_samples = X.shape[0]
        probas = np.zeros((n_samples, self.n_thresholds))
        
        for i, pipeline in enumerate(self.pipelines):
            # Get probability of class 1 (y > threshold)
            probas[:, i] = pipeline.predict_proba(X)[:, 1]
        
        return probas
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict ordinal classes.
        
        Parameters:
        -----------
        X : np.ndarray
            Features to predict, shape (n_samples, n_features).
        
        Returns:
        --------
        y_pred : np.ndarray
            Predicted ordinal classes, shape (n_samples,) with values 0-4.
        """
        probas = self.predict_proba(X)
        return decode_svm_predictions(probas)
    
    def predict_proba_classes(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities (for calibration/evaluation).
        
        Converts threshold probabilities P(y > k) to class probabilities P(y = c).
        
        Parameters:
        -----------
        X : np.ndarray
            Features to predict, shape (n_samples, n_features).
        
        Returns:
        --------
        class_probas : np.ndarray
            Class probabilities, shape (n_samples, 5) for classes 0-4.
        """
        threshold_probas = self.predict_proba(X)
        n_samples = threshold_probas.shape[0]
        n_classes = 5
        
        class_probas = np.zeros((n_samples, n_classes))
        
        # Convert threshold probabilities to class probabilities
        # P(y = 0) = 1 - P(y > 0)
        # P(y = 1) = P(y > 0) - P(y > 1)
        # P(y = 2) = P(y > 1) - P(y > 2)
        # P(y = 3) = P(y > 2) - P(y > 3)
        # P(y = 4) = P(y > 3)
        
        for i in range(n_samples):
            # Add padding for easier indexing
            p_greater = np.concatenate([[1.0], threshold_probas[i], [0.0]])
            
            for c in range(n_classes):
                class_probas[i, c] = p_greater[c] - p_greater[c + 1]
        
        # Normalize to ensure probabilities sum to 1 (handle numerical issues)
        class_probas = np.maximum(class_probas, 0)  # Ensure non-negative
        row_sums = class_probas.sum(axis=1, keepdims=True)
        class_probas = class_probas / (row_sums + 1e-10)
        
        return class_probas


def train_threshold_svm(
    X: np.ndarray,
    y: np.ndarray,
    C: float = 1.0,
    gamma: Union[str, float] = 'scale',
    kernel: str = 'linear',
    max_iter: int = 500,
    subsample: float = 1.0,
    verbose: int = 0
) -> OrdinalSVMClassifier:
    """
    Train an ordinal SVM using threshold-based approach.
    
    Creates 4 binary SVM classifiers (one for each threshold) to model
    the ordinal nature of the labels. Each SVM predicts P(y > k) for
    threshold k.
    
    Parameters:
    -----------
    X : np.ndarray
        2D array of features with shape (n_samples, n_features).
        Can be combined features (HOG + SIFT + color histograms, etc.).
    y : np.ndarray
        1D array of integer ordinal labels, ranging from 0 to 4:
        - 0: Immature
        - 1: Partially Ripe
        - 2: Fully Ripe
        - 3: Overripe
        - 4: Decayed
    C : float, default=1.0
        Regularization parameter. The strength of the regularization is
        inversely proportional to C.
    gamma : str or float, default='scale'
        Kernel coefficient for RBF kernel. If 'scale', uses 1/(n_features * X.var()).
    kernel : str, default='linear'
        Kernel type ('linear' or 'rbf'). Linear is much faster for large datasets.
    max_iter : int, default=500
        Maximum iterations for SVM solver. Use -1 for no limit.
    subsample : float, default=1.0
        Fraction of data to use for training (0.0-1.0). Use <1.0 for faster training.
        E.g., 0.5 uses 50% of the data.
    verbose : int, default=0
        Verbosity level (0, 1, or 2).
    
    Returns:
    --------
    classifier : OrdinalSVMClassifier
        Fitted ordinal SVM classifier with predict() and predict_proba() methods.
    """
    if verbose > 0:
        print(f"Training Ordinal SVM with C={C}, kernel={kernel}, gamma={gamma}, max_iter={max_iter}...")
        print(f"Features shape: {X.shape}")
        print(f"Labels shape: {y.shape}")
    
    # Subsample data if requested
    if subsample < 1.0:
        n_samples = int(len(X) * subsample)
        indices = np.random.RandomState(42).permutation(len(X))[:n_samples]
        X = X[indices]
        y = y[indices]
        if verbose > 0:
            print(f"Subsampled to {len(X)} samples ({subsample*100:.0f}% of data)")
    
    # Create and fit classifier
    # Use LinearSVC by default for linear kernel (much faster)
    use_linear_svc = (kernel == 'linear')
    
    classifier = OrdinalSVMClassifier(
        C=C, gamma=gamma, kernel=kernel, max_iter=max_iter, 
        use_linear_svc=use_linear_svc, verbose=verbose
    )
    classifier.fit(X, y)
    
    if verbose > 0:
        print("Training completed.")
    
    return classifier
