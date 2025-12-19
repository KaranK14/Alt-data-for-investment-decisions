"""
Class imbalance handling utilities.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional

try:
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.over_sampling import RandomOverSampler
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    print("Warning: imbalanced-learn not installed. Install with: pip install imbalanced-learn")


def downsample_majority(
    X: pd.DataFrame,
    y: np.ndarray,
    target_ratio: float = 1.0,
    random_state: Optional[int] = None
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Downsample majority class to achieve target ratio."""
    if not IMBLEARN_AVAILABLE:
        raise ImportError("imbalanced-learn is required. Install with: pip install imbalanced-learn")
    
    unique, counts = np.unique(y, return_counts=True)
    minority_class = unique[np.argmin(counts)]
    majority_class = unique[np.argmax(counts)]
    
    minority_count = counts.min()
    majority_count = counts.max()
    majority_target = int(minority_count * target_ratio)
    
    # For downsampling, target must be <= current majority count
    # If target_ratio would require more samples than available, cap it at current count
    if majority_target > majority_count:
        majority_target = majority_count
        print(f"    Warning: Requested ratio {target_ratio:.1f} would require {int(minority_count * target_ratio)} majority samples, but only {majority_count} available. Using {majority_count}.")
    
    sampling_strategy = {majority_class: majority_target}
    
    rus = RandomUnderSampler(
        sampling_strategy=sampling_strategy,
        random_state=random_state
    )
    
    X_resampled, y_resampled = rus.fit_resample(X, y)
    
    if isinstance(X, pd.DataFrame):
        X_resampled = pd.DataFrame(
            X_resampled,
            columns=X.columns,
            index=pd.RangeIndex(len(X_resampled))
        )
    
    return X_resampled, y_resampled


def oversample_minority(
    X: pd.DataFrame,
    y: np.ndarray,
    target_ratio: float = 1.0,
    random_state: Optional[int] = None
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Oversample minority class to achieve target ratio."""
    if not IMBLEARN_AVAILABLE:
        raise ImportError("imbalanced-learn is required. Install with: pip install imbalanced-learn")
    
    unique, counts = np.unique(y, return_counts=True)
    minority_class = unique[np.argmin(counts)]
    majority_class = unique[np.argmax(counts)]
    minority_count = counts.min()
    majority_count = counts.max()
    minority_target = int(majority_count * target_ratio)
    
    # For oversampling, target must be >= current minority count
    # If target_ratio would require fewer samples than current, use current count
    if minority_target < minority_count:
        minority_target = minority_count
        print(f"    Warning: Requested ratio {target_ratio:.1f} would require {int(majority_count * target_ratio)} minority samples, but current minority has {minority_count}. Using {minority_count} (no oversampling needed).")
    
    sampling_strategy = {minority_class: minority_target}
    
    ros = RandomOverSampler(
        sampling_strategy=sampling_strategy,
        random_state=random_state
    )
    
    X_resampled, y_resampled = ros.fit_resample(X, y)
    
    if isinstance(X, pd.DataFrame):
        X_resampled = pd.DataFrame(
            X_resampled,
            columns=X.columns,
            index=pd.RangeIndex(len(X_resampled))
        )
    
    return X_resampled, y_resampled


def apply_imbalance_strategy(
    X: pd.DataFrame,
    y: np.ndarray,
    strategy: str,
    random_state: Optional[int] = None,
    target_ratio: Optional[float] = None
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Apply imbalance handling strategy."""
    from .config import (
        IMBALANCE_STRATEGY_CLASS_WEIGHT,
        IMBALANCE_STRATEGY_DOWNSAMPLE_BALANCED,
        IMBALANCE_STRATEGY_DOWNSAMPLE_1_5,
        IMBALANCE_STRATEGY_PARTIAL,
    )
    
    if strategy == IMBALANCE_STRATEGY_CLASS_WEIGHT:
        return X.copy(), y.copy()
    elif strategy == IMBALANCE_STRATEGY_DOWNSAMPLE_BALANCED:
        # Fully balanced (1:1 ratio)
        return downsample_majority(X, y, target_ratio=1.0, random_state=random_state)
    elif strategy == IMBALANCE_STRATEGY_DOWNSAMPLE_1_5:
        # Partially downsampled (1:1.5 ratio)
        return downsample_majority(X, y, target_ratio=1.5, random_state=random_state)
    elif strategy == IMBALANCE_STRATEGY_PARTIAL:
        # Partially downsampled (1:2 ratio)
        if target_ratio is None:
            target_ratio = 2.0
        return downsample_majority(X, y, target_ratio=target_ratio, random_state=random_state)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

