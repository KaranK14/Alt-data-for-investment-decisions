"""
XGBoost-based feature selection.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
import xgboost as xgb

from .config import (
    FEATURE_SELECTION_XGB_PARAMS,
    MIN_FEATURES_AFTER_SELECTION,
)


def select_features_with_xgboost(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    xgb_params: Optional[dict] = None,
    min_importance_ratio: float = 1.0,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, List[str], xgb.XGBClassifier]:
    """
    Select features using XGBoost importance scores.
    
    Retains features with importance > mean(importance) * min_importance_ratio.
    
    Returns:
        Tuple of (feature_mask, selected_feature_names, fitted_xgb_model)
    """
    if xgb_params is None:
        xgb_params = FEATURE_SELECTION_XGB_PARAMS.copy()
    
    # Override random_state if provided
    if random_state is not None:
        xgb_params = xgb_params.copy()
        xgb_params['random_state'] = random_state
    
    print(f"\nFeature Selection (XGBoost):")
    print(f"  Input features: {X_train.shape[1]}")
    
    # Handle missing values and convert to numeric
    X_train_clean = X_train.copy()
    if X_train_clean.isna().any().any():
        for col in X_train_clean.columns:
            if X_train_clean[col].dtype in ['int64', 'float64']:
                X_train_clean[col] = X_train_clean[col].fillna(X_train_clean[col].median())
            else:
                X_train_clean[col] = X_train_clean[col].fillna(X_train_clean[col].mode()[0] if len(X_train_clean[col].mode()) > 0 else 0)
    
    # Convert categorical to numeric
    X_train_numeric = X_train_clean.copy()
    for col in X_train_numeric.columns:
        if X_train_numeric[col].dtype == 'object':
            try:
                X_train_numeric[col] = pd.to_numeric(X_train_numeric[col], errors='coerce')
            except:
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                X_train_numeric[col] = le.fit_transform(X_train_numeric[col].astype(str))
    
    X_train_numeric = X_train_numeric.fillna(0)
    
    # Fit XGBoost
    model = xgb.XGBClassifier(**xgb_params)
    model.fit(X_train_numeric.values, y_train)
    
    # Get feature importances
    importances = model.feature_importances_
    mean_importance = importances.mean()
    threshold = mean_importance * min_importance_ratio
    
    feature_mask = importances > threshold
    
    # Safety check
    n_selected = feature_mask.sum()
    if n_selected < MIN_FEATURES_AFTER_SELECTION:
        print(f"  Warning: Only {n_selected} features selected (minimum: {MIN_FEATURES_AFTER_SELECTION})")
        top_indices = np.argsort(importances)[-MIN_FEATURES_AFTER_SELECTION:]
        feature_mask = np.zeros_like(feature_mask, dtype=bool)
        feature_mask[top_indices] = True
        n_selected = MIN_FEATURES_AFTER_SELECTION
    
    selected_feature_names = X_train.columns[feature_mask].tolist()
    
    print(f"  Selected features: {n_selected} / {len(importances)} ({n_selected/len(importances)*100:.1f}%)")
    print(f"  Importance threshold: {threshold:.6f} (mean: {mean_importance:.6f})")
    
    return feature_mask, selected_feature_names, model


def apply_feature_mask(
    X: pd.DataFrame,
    feature_mask: np.ndarray,
    feature_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """Apply feature selection mask to DataFrame."""
    if feature_names is None:
        selected_cols = X.columns[feature_mask].tolist()
    else:
        selected_cols = [name for name, keep in zip(feature_names, feature_mask) if keep]
        selected_cols = [col for col in selected_cols if col in X.columns]
    
    X_selected = X[selected_cols].copy()
    return X_selected

