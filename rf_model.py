"""
Random Forest model implementation - supports both tabular and tabular+embeddings modes.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from typing import Dict
import warnings

from ..config import (
    RF_GRID_PARAMS, N_ITERATIONS, CV_FOLDS, RANDOM_SEED,
    IMBALANCE_STRATEGY_CLASS_WEIGHT,
    IMBALANCE_STRATEGY_DOWNSAMPLE_BALANCED,
    IMBALANCE_STRATEGY_DOWNSAMPLE_1_5,
    IMBALANCE_STRATEGY_PARTIAL,
    ALL_IMBALANCE_STRATEGIES,
)
from ..imbalance import apply_imbalance_strategy


def train_evaluate_rf(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    n_iterations: int = N_ITERATIONS,
    imbalance_strategies: list = None,
    random_state: int = RANDOM_SEED
) -> Dict[str, Dict[str, float]]:
    """Train and evaluate Random Forest with multiple imbalance strategies."""
    if imbalance_strategies is None:
        imbalance_strategies = ALL_IMBALANCE_STRATEGIES
    
    print(f"\n{'='*80}")
    print(f"Random Forest Model Training")
    print(f"{'='*80}")
    print(f"  Iterations per strategy: {n_iterations}")
    print(f"  Strategies: {imbalance_strategies}")
    
    results = {}
    
    for strategy in imbalance_strategies:
        print(f"\n  Strategy: {strategy}")
        print(f"  {'-'*78}")
        
        # Hyperparameter tuning
        if strategy == IMBALANCE_STRATEGY_CLASS_WEIGHT:
            grid_params = RF_GRID_PARAMS.copy()
        else:
            grid_params = {k: v for k, v in RF_GRID_PARAMS.items() if k != "class_weight"}
            grid_params["class_weight"] = [None]
        
        base_rf = RandomForestClassifier(random_state=random_state)
        
        print(f"    Performing grid search ({CV_FOLDS}-fold CV)...")
        grid_search = GridSearchCV(
            base_rf, grid_params, cv=CV_FOLDS,
            scoring='roc_auc', n_jobs=-1, verbose=0
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            grid_search.fit(X_train, y_train)
        
        best_params = grid_search.best_params_
        print(f"    Best params: {best_params}")
        print(f"    Best CV score: {grid_search.best_score_:.4f}")
        
        # Repeated runs
        metrics_list = []
        
        for iteration in range(n_iterations):
            X_train_processed, y_train_processed = apply_imbalance_strategy(
                X_train, y_train, strategy,
                random_state=random_state + iteration,
                target_ratio=None  # Ratio is determined by strategy name
            )
            
            # Update random_state in params (avoid duplicate argument)
            params = best_params.copy()
            params['random_state'] = random_state + iteration
            rf = RandomForestClassifier(**params)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rf.fit(X_train_processed, y_train_processed)
            
            y_test_pred = rf.predict(X_test)
            y_test_proba = rf.predict_proba(X_test)[:, 1]
            
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
        
        # Aggregate
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

