"""
XGBoost model implementation - supports both tabular and tabular+embeddings modes.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from typing import Dict, Tuple, Optional
import warnings
import joblib

from sklearn.preprocessing import LabelEncoder
from ..config import (
    XGB_GRID_PARAMS, N_ITERATIONS, CV_FOLDS, RANDOM_SEED,
    ALL_IMBALANCE_STRATEGIES, MODELS_DIR,
)
from ..imbalance import apply_imbalance_strategy


def convert_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert DataFrame columns to numeric types for XGBoost.
    
    Handles:
    - Object dtype columns: converts to numeric using LabelEncoder or numeric conversion
    - Boolean columns: converts to int (0/1)
    - Missing values: fills with 0
    """
    df_clean = df.copy()
    
    # Store label encoders for each column (in case we need to apply same mapping to test set)
    label_encoders = {}
    
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            # Try to convert to numeric directly first
            try:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                df_clean[col] = df_clean[col].fillna(0)
            except:
                # If that fails, use LabelEncoder
                le = LabelEncoder()
                # Handle NaN by converting to string 'nan' first
                col_series = df_clean[col].astype(str)
                df_clean[col] = le.fit_transform(col_series)
                label_encoders[col] = le
        elif df_clean[col].dtype == 'bool':
            # Convert boolean to int
            df_clean[col] = df_clean[col].astype(int)
        elif pd.api.types.is_categorical_dtype(df_clean[col]):
            # Convert category codes to numeric
            df_clean[col] = df_clean[col].cat.codes
        else:
            # For numeric types, just fill NaN
            df_clean[col] = df_clean[col].fillna(0)
    
    # Final fillna for any remaining NaN
    df_clean = df_clean.fillna(0)
    
    # Ensure all columns are numeric
    for col in df_clean.columns:
        if not pd.api.types.is_numeric_dtype(df_clean[col]):
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)
    
    return df_clean


def train_evaluate_xgb(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    n_iterations: int = N_ITERATIONS,
    imbalance_strategies: list = None,
    random_state: int = RANDOM_SEED,
    use_feature_selection: bool = False
) -> Dict[str, Dict[str, float]]:
    """Train and evaluate XGBoost with multiple imbalance strategies."""
    if imbalance_strategies is None:
        imbalance_strategies = ALL_IMBALANCE_STRATEGIES
    
    print(f"\n{'='*80}")
    print(f"XGBoost Model Training")
    print(f"{'='*80}")
    print(f"  Iterations per strategy: {n_iterations}")
    print(f"  Strategies: {imbalance_strategies}")
    
    # Convert to numeric once (handles object dtype columns)
    print(f"\n  Preprocessing data...")
    X_train_clean = convert_to_numeric(X_train)
    X_val_clean = convert_to_numeric(X_val)
    X_test_clean = convert_to_numeric(X_test)
    
    # Run grid search ONCE before iterating strategies (optimization)
    # Hyperparameters shouldn't depend on imbalance strategy
    print(f"\n  Performing grid search ({CV_FOLDS}-fold CV) on full training data...")
    grid_params = XGB_GRID_PARAMS.copy()
    base_xgb = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=random_state
    )
    
    grid_search = GridSearchCV(
        base_xgb, grid_params, cv=CV_FOLDS,
        scoring='roc_auc', n_jobs=-1, verbose=1  # verbose=1 to show progress
    )
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grid_search.fit(X_train_clean, y_train)
    
    best_params = grid_search.best_params_
    print(f"  Best params: {best_params}")
    print(f"  Best CV score: {grid_search.best_score_:.4f}")
    print(f"  (These hyperparameters will be reused for all imbalance strategies)\n")
    
    results = {}
    
    for strategy in imbalance_strategies:
        print(f"\n  Strategy: {strategy}")
        print(f"  {'-'*78}")
        
        metrics_list = []
        
        for iteration in range(n_iterations):
            X_train_processed, y_train_processed = apply_imbalance_strategy(
                X_train, y_train, strategy,
                random_state=random_state + iteration,
                target_ratio=None  # Ratio is determined by strategy name
            )
            
            # Convert processed (downsampled) data to numeric
            # Note: Need to maintain column alignment with X_train_clean
            X_train_processed_clean = convert_to_numeric(X_train_processed)
            # X_test_clean already converted above (reuse)
            
            # Update random_state in params (avoid duplicate argument)
            params = best_params.copy()
            params['random_state'] = random_state + iteration
            params['objective'] = 'binary:logistic'
            params['eval_metric'] = 'logloss'
            params['use_label_encoder'] = False
            xgb_model = xgb.XGBClassifier(**params)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                xgb_model.fit(X_train_processed_clean, y_train_processed)
            
            y_test_pred = xgb_model.predict(X_test_clean)
            y_test_proba = xgb_model.predict_proba(X_test_clean)[:, 1]
            
            metrics = {
                "accuracy": accuracy_score(y_test, y_test_pred),
                "precision": precision_score(y_test, y_test_pred, zero_division=0),
                "recall": recall_score(y_test, y_test_pred, zero_division=0),
                "f1": f1_score(y_test, y_test_pred, zero_division=0),
                "roc_auc": roc_auc_score(y_test, y_test_proba),
            }
            
            metrics_list.append(metrics)
            
            if (iteration + 1) % 5 == 0:
                print(f"    Completed iteration {iteration + 1}/{n_iterations}")
        
        metrics_df = pd.DataFrame(metrics_list)
        results[strategy] = {
            "accuracy_mean": metrics_df["accuracy"].mean(),
            "accuracy_std": metrics_df["accuracy"].std(),
            "precision_mean": metrics_df["precision"].mean(),
            "precision_std": metrics_df["precision"].std(),
            "recall_mean": metrics_df["recall"].mean(),
            "recall_std": metrics_df["recall"].std(),
            "f1_mean": metrics_df["f1"].mean(),
            "f1_std": metrics_df["f1"].std(),
            "roc_auc_mean": metrics_df["roc_auc"].mean(),
            "roc_auc_std": metrics_df["roc_auc"].std(),
        }
        
        print(f"\n    Test Set Results (mean ± std):")
        print(f"      Accuracy:  {results[strategy]['accuracy_mean']:.4f} ± {results[strategy]['accuracy_std']:.4f}")
        print(f"      Precision: {results[strategy]['precision_mean']:.4f} ± {results[strategy]['precision_std']:.4f}")
        print(f"      Recall:    {results[strategy]['recall_mean']:.4f} ± {results[strategy]['recall_std']:.4f}")
        print(f"      F1:        {results[strategy]['f1_mean']:.4f} ± {results[strategy]['f1_std']:.4f}")
        print(f"      ROC-AUC:   {results[strategy]['roc_auc_mean']:.4f} ± {results[strategy]['roc_auc_std']:.4f}")
    
    return results

