"""
Ordinal Gradient Boosting using LightGBM

Uses LightGBM with multiclass objective for ordinal classification.
"""

import joblib
import numpy as np
import sys
from sklearn.preprocessing import StandardScaler

# Try importing LightGBM
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Warning: LightGBM not installed. Run: pip install lightgbm")


class LGBMWrapper:
    """Wrapper for LightGBM model with sklearn-like interface."""
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        probs = self.model.predict(X_scaled)
        return np.argmax(probs, axis=1)
    
    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def score(self, X, y):
        return np.mean(self.predict(X) == y)


def train_with_lightgbm(X: np.ndarray, y: np.ndarray, num_iterations: int = 100,
                        learning_rate: float = 0.1, random_state: int = 42):
    """
    Train using LightGBM with multiclass objective for ordinal classification.
    """
    if not LIGHTGBM_AVAILABLE:
        raise ImportError("LightGBM not installed. Run: pip install lightgbm")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create LightGBM dataset
    train_data = lgb.Dataset(X_scaled, label=y)
    
    # Simple, effective parameters for multiclass
    params = {
        'objective': 'multiclass',
        'num_class': len(np.unique(y)),
        'metric': 'multi_logloss',
        'learning_rate': learning_rate,
        'num_leaves': 31,
        'verbose': 1,
        'random_state': random_state,
        'force_col_wise': True
    }
    
    print(f"Training LightGBM (multiclass, {num_iterations} iterations)...")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=num_iterations,
        callbacks=[lgb.log_evaluation(period=10)]
    )
    
    return LGBMWrapper(model, scaler)


def load_data(features_path: str, labels_path: str):
    """Load features and integer ordinal labels."""
    feats = joblib.load(features_path)
    labs = joblib.load(labels_path)
    
    X = feats.get('features', feats) if isinstance(feats, dict) else feats
    y = labs.get('labels', labs) if isinstance(labs, dict) else labs
    
    return np.asarray(X), np.asarray(y)
