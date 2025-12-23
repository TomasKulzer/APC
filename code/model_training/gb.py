"""  
Ordinal Gradient Boosting using LightGBM

Uses LightGBM with multiclass objective for ordinal classification.
Includes hyperparameter tuning, early stopping, and class balancing.
"""

import joblib
import numpy as np
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.utils.class_weight import compute_class_weight

# Try importing LightGBM
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Warning: LightGBM not installed. Run: pip install lightgbm")


class LGBMWrapper:
    """Wrapper for LightGBM model with sklearn-like interface."""
    def __init__(self, model=None, scaler=None, params=None, split_data=None):
        self.model = model
        self.scaler = scaler
        self.params = params or {}
        self.split_data = split_data or {}  # Store train/val split info
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        probs = self.model.predict(X_scaled)
        return np.argmax(probs, axis=1)
    
    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def score(self, X, y):
        return np.mean(self.predict(X) == y)


def train_with_lightgbm(
    X: np.ndarray, 
    y: np.ndarray, 
    num_iterations: int = 5000,
    learning_rate: float = 0.02,  # Lower for better generalization
    num_leaves: int = 50,  # From tuning results
    max_depth: int = 8,  # From tuning results
    feature_fraction: float = 0.8,  # From tuning results
    bagging_fraction: float = 0.8,
    min_child_samples: int = 50,  # From tuning results
    reg_alpha: float = 0.1,  # L1 regularization
    reg_lambda: float = 0.1,  # L2 regularization
    use_tuning: bool = False,
    random_state: int = 42
):
    """
    Train using LightGBM with multiclass objective for ordinal classification.
    
    Parameters:
    -----------
    X : np.ndarray
        Training features.
    y : np.ndarray
        Training labels (0-4).
    num_iterations : int, default=5000
        Number of boosting iterations.
    learning_rate : float, default=0.02
        Learning rate (lower for better generalization).
    num_leaves : int, default=50
        Maximum tree leaves for base learners (from tuning).
    max_depth : int, default=8
        Maximum tree depth for base learners (from tuning).
    feature_fraction : float, default=0.8
        Fraction of features to use per iteration (from tuning).
    bagging_fraction : float, default=0.8
        Fraction of data to use per iteration.
    min_child_samples : int, default=50
        Minimum samples per leaf (from tuning).
    reg_alpha : float, default=0.1
        L1 regularization term on weights.
    reg_lambda : float, default=0.1
        L2 regularization term on weights.
    use_tuning : bool, default=False
        Whether to use hyperparameter tuning (slower but better).
    random_state : int, default=42
        Random seed.
    
    Returns:
    --------
    model : LGBMWrapper
        Trained LightGBM model wrapper.
    """
    if not LIGHTGBM_AVAILABLE:
        raise ImportError("LightGBM not installed. Run: pip install lightgbm")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Compute class weights to handle imbalance
    classes = np.unique(y)
    class_weights = compute_class_weight('balanced', classes=classes, y=y)
    class_weight_dict = dict(zip(classes, class_weights))
    sample_weights = np.array([class_weight_dict[label] for label in y])
    
    print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"Class weights: {class_weight_dict}")
    print(f"Training on full dataset: {len(X)} samples")
    
    # Use full training data (no internal split)
    X_train, y_train, w_train = X_scaled, y, sample_weights
    
    # Base parameters
    params = {
        'objective': 'multiclass',
        'num_class': len(classes),
        'metric': 'multi_logloss',
        'learning_rate': learning_rate,
        'num_leaves': num_leaves,
        'max_depth': max_depth,
        'feature_fraction': feature_fraction,
        'bagging_fraction': bagging_fraction,
        'bagging_freq': 5,
        'min_child_samples': min_child_samples,
        'reg_alpha': reg_alpha,  # L1 regularization
        'reg_lambda': reg_lambda,  # L2 regularization
        'min_split_gain': 0.1,  # Minimum loss reduction for split
        'verbose': 1,
        'random_state': random_state,
        'force_col_wise': True,
        'boost_from_average': True  # Better for multiclass
    }
    
    if use_tuning:
        print("\n" + "="*70)
        print("HYPERPARAMETER TUNING (this may take a while...)")
        print("="*70)
        
        # Simplified tuning grid for speed
        param_grid = {
            'num_leaves': [15, 31, 50],
            'max_depth': [6, 8, 10],
            'feature_fraction': [0.7, 0.8, 0.9],
            'min_child_samples': [20, 50]
        }
        
        # Use LGBMClassifier for GridSearchCV compatibility
        lgb_clf = lgb.LGBMClassifier(
            objective='multiclass',
            num_class=len(classes),
            learning_rate=learning_rate,
            n_estimators=num_iterations,
            random_state=random_state,
            class_weight='balanced',
            force_col_wise=True,
            verbose=-1
        )
        
        grid_search = GridSearchCV(
            lgb_clf, 
            param_grid, 
            cv=3,  # 3-fold for speed
            scoring='accuracy',
            n_jobs=-1,
            verbose=2
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        # Update params with best values
        params.update(grid_search.best_params_)
        
        # Return the best estimator wrapped
        best_model = grid_search.best_estimator_
        wrapper = LGBMWrapper(model=best_model.booster_, scaler=scaler, params=params)
        
        # Show training accuracy
        train_acc = wrapper.score(X_train, y_train)
        print(f"\nFull training set accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
        print("Note: Evaluate on external validation set for true generalization performance.")
        
        return wrapper
    
    else:
        # Standard training on full dataset
        train_data = lgb.Dataset(X_train, label=y_train, weight=w_train)
        
        callbacks = [lgb.log_evaluation(period=100)]
        
        print(f"\nTraining on full dataset (no internal validation split)...")
        print(f"Parameters: {params}")
        print(f"Training LightGBM (multiclass, {num_iterations} iterations)...\n")
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=num_iterations,
            valid_sets=[train_data],
            valid_names=['train'],
            callbacks=callbacks
        )
        
        wrapper = LGBMWrapper(model=model, scaler=scaler, params=params)
        
        # Compute and display training accuracy
        train_acc = wrapper.score(X_train, y_train)
        print(f"\nFull training set accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
        print("Note: Evaluate on external validation set for true generalization performance.")
        
        return wrapper


def load_data(features_path: str, labels_path: str):
    """Load features and integer ordinal labels."""
    feats = joblib.load(features_path)
    labs = joblib.load(labels_path)
    
    X = feats.get('features', feats) if isinstance(feats, dict) else feats
    y = labs.get('labels', labs) if isinstance(labs, dict) else labs
    
    return np.asarray(X), np.asarray(y)
