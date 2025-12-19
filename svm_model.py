"""
SVM (Support Vector Machine) model implementation - baseline for tabular-only features.
"""

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from typing import Dict
import warnings

from ..config import SVM_PARAMS, N_ITERATIONS, IMBALANCE_STRATEGY_CLASS_WEIGHT
from ..imbalance import apply_imbalance_strategy


def train_evaluate_svm(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    n_iterations: int = N_ITERATIONS,
    sampling_scheme: str = IMBALANCE_STRATEGY_CLASS_WEIGHT,
    random_state: int = 42
) -> Dict[str, float]:
    """Train and evaluate SVM model with repeated runs."""
    print(f"\n{'='*80}")
    print(f"SVM Model Training")
    print(f"{'='*80}")
    print(f"  Iterations: {n_iterations}")
    print(f"  Sampling scheme: {sampling_scheme}")
    
    # Impute missing values (SVM doesn't handle NaN)
    # Use median for numeric, most frequent for categorical
    n_missing_train = X_train.isna().sum().sum()
    if n_missing_train > 0:
        print(f"  Imputing {n_missing_train} missing values in training set...")
    
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
    
    X_train_imputed = X_train.copy()
    X_val_imputed = X_val.copy()
    X_test_imputed = X_test.copy()
    
    # Impute numeric columns with median
    if len(numeric_cols) > 0:
        numeric_imputer = SimpleImputer(strategy='median')
        X_train_imputed[numeric_cols] = pd.DataFrame(
            numeric_imputer.fit_transform(X_train[numeric_cols]),
            columns=numeric_cols,
            index=X_train.index
        )
        X_val_imputed[numeric_cols] = pd.DataFrame(
            numeric_imputer.transform(X_val[numeric_cols]),
            columns=numeric_cols,
            index=X_val.index
        )
        X_test_imputed[numeric_cols] = pd.DataFrame(
            numeric_imputer.transform(X_test[numeric_cols]),
            columns=numeric_cols,
            index=X_test.index
        )
    
    # Impute categorical columns with most frequent
    if len(categorical_cols) > 0:
        categorical_imputer = SimpleImputer(strategy='most_frequent')
        X_train_imputed[categorical_cols] = pd.DataFrame(
            categorical_imputer.fit_transform(X_train[categorical_cols]),
            columns=categorical_cols,
            index=X_train.index
        )
        X_val_imputed[categorical_cols] = pd.DataFrame(
            categorical_imputer.transform(X_val[categorical_cols]),
            columns=categorical_cols,
            index=X_val.index
        )
        X_test_imputed[categorical_cols] = pd.DataFrame(
            categorical_imputer.transform(X_test[categorical_cols]),
            columns=categorical_cols,
            index=X_test.index
        )
    
    # Verify no NaN remains (fallback safety check)
    if X_train_imputed.isna().any().any() or X_val_imputed.isna().any().any() or X_test_imputed.isna().any().any():
        print("  WARNING: Some NaN values remain after imputation! Filling with 0.")
        X_train_imputed = X_train_imputed.fillna(0)
        X_val_imputed = X_val_imputed.fillna(0)
        X_test_imputed = X_test_imputed.fillna(0)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_val_scaled = scaler.transform(X_val_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)
    
    metrics_list = []
    
    for iteration in range(n_iterations):
        if sampling_scheme == IMBALANCE_STRATEGY_CLASS_WEIGHT:
            X_train_processed = X_train_scaled
            y_train_processed = y_train.copy()
            svm_params = SVM_PARAMS.copy()
        else:
            X_train_df = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
            X_train_processed_df, y_train_processed = apply_imbalance_strategy(
                X_train_df, y_train, sampling_scheme, random_state=random_state + iteration
            )
            X_train_processed = X_train_processed_df.values
            svm_params = {k: v for k, v in SVM_PARAMS.items() if k != "class_weight"}
        
        # Update random_state in params (avoid duplicate argument)
        svm_params = svm_params.copy()
        svm_params['random_state'] = random_state + iteration
        model = SVC(**svm_params)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train_processed, y_train_processed)
        
        y_test_pred = model.predict(X_test_scaled)
        y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        metrics = {
            "accuracy": accuracy_score(y_test, y_test_pred),
            "precision": precision_score(y_test, y_test_pred, zero_division=0),
            "recall": recall_score(y_test, y_test_pred, zero_division=0),
            "f1": f1_score(y_test, y_test_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_test_proba),
        }
        
        metrics_list.append(metrics)
        
        if (iteration + 1) % 5 == 0:
            print(f"  Completed iteration {iteration + 1}/{n_iterations}")
    
    # Aggregate metrics
    metrics_df = pd.DataFrame(metrics_list)
    result = {
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
    
    print(f"\n  Test Set Results (mean ± std over {n_iterations} iterations):")
    print(f"    Accuracy:  {result['accuracy_mean']:.4f} ± {result['accuracy_std']:.4f}")
    print(f"    Precision: {result['precision_mean']:.4f} ± {result['precision_std']:.4f}")
    print(f"    Recall:    {result['recall_mean']:.4f} ± {result['recall_std']:.4f}")
    print(f"    F1:        {result['f1_mean']:.4f} ± {result['f1_std']:.4f}")
    print(f"    ROC-AUC:   {result['roc_auc_mean']:.4f} ± {result['roc_auc_std']:.4f}")
    
    return result

