"""
Data loading and feature splitting utilities.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List
from sklearn.model_selection import train_test_split

from .config import (
    DATASET_PATH,
    LABEL_COL,
    ID_COL,
    PCA_EMB_PREFIX,
    TEXT_COLUMNS,
    JSON_COLUMNS,
    LEAKAGE_COLUMNS,
    MANUALLY_EXCLUDED_COLUMNS,
    TEST_SIZE,
    VAL_SIZE,
    RANDOM_SEED,
)


def load_dataset() -> pd.DataFrame:
    """
    Load the main dataset.
    
    Returns:
        DataFrame with all trials and features
    """
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")
    
    df = pd.read_parquet(DATASET_PATH)
    print(f"Loaded dataset: {len(df):,} trials × {len(df.columns)} columns")
    return df


def identify_pca_columns(df: pd.DataFrame) -> List[str]:
    """Identify PCA embedding columns."""
    pca_cols = [col for col in df.columns if col.startswith(PCA_EMB_PREFIX)]
    return pca_cols


def identify_tabular_columns(df: pd.DataFrame, exclude_leakage: bool = True) -> List[str]:
    """
    Identify tabular (structured) feature columns.
    
    Excludes identifiers, labels, text columns, JSON columns, PCA embeddings, 
    leakage columns, and manually excluded columns.
    """
    exclude = [ID_COL, LABEL_COL, "feasibility_label"]
    exclude.extend(TEXT_COLUMNS)
    exclude.extend(JSON_COLUMNS)
    exclude.extend(identify_pca_columns(df))
    exclude.extend(MANUALLY_EXCLUDED_COLUMNS)  # Manually excluded based on domain knowledge
    
    if exclude_leakage:
        exclude.extend(LEAKAGE_COLUMNS)
        # Also exclude any column with leakage keywords (dynamic check)
        for col in df.columns:
            col_lower = col.lower()
            if any(leakage in col_lower for leakage in [
                "posted_date", "completion_date", "is_historical", 
                "submit_date", "status_verified", "why_stopped", "has_results"
            ]):
                if col not in exclude:
                    exclude.append(col)
    
    tabular_cols = [col for col in df.columns if col not in exclude]
    return tabular_cols


def split_features_labels(
    df: pd.DataFrame,
    mode: str = "tabular"
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split dataset into features (X) and labels (y).
    
    Args:
        df: DataFrame with all features
        mode: "tabular" or "tabular_plus_embeddings"
        
    Returns:
        Tuple of (X, y)
    """
    if LABEL_COL not in df.columns:
        raise ValueError(f"Label column '{LABEL_COL}' not found in dataset")
    
    # Drop rows with missing labels
    df_valid = df[df[LABEL_COL].notna()].copy()
    y = df_valid[LABEL_COL].astype(int)
    
    # Get feature columns based on mode
    if mode == "tabular":
        feature_cols = identify_tabular_columns(df_valid)
    elif mode == "tabular_plus_embeddings":
        tabular_cols = identify_tabular_columns(df_valid)
        pca_cols = identify_pca_columns(df_valid)
        feature_cols = tabular_cols + pca_cols
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'tabular' or 'tabular_plus_embeddings'")
    
    X = df_valid[feature_cols].copy()
    
    print(f"\nMode: {mode}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Trials: {len(X):,}")
    print(f"  Class distribution: {y.value_counts().to_dict()}")
    
    return X, y


def train_val_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    val_size: float = VAL_SIZE,
    random_state: int = RANDOM_SEED,
    stratify: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Split data into train, validation, and test sets.
    
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # First split: separate test set
    stratify_y = y if stratify else None
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_y
    )
    
    # Second split: separate train and validation
    stratify_temp = y_temp if stratify else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size,
        random_state=random_state,
        stratify=stratify_temp
    )
    
    print(f"\nData Split:")
    print(f"  Train: {len(X_train):,} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Val:   {len(X_val):,} ({len(X_val)/len(X)*100:.1f}%)")
    print(f"  Test:  {len(X_test):,} ({len(X_test)/len(X)*100:.1f}%)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def remove_constant_features(X: pd.DataFrame) -> pd.DataFrame:
    """Remove features with only one unique value."""
    constant_cols = [col for col in X.columns if X[col].nunique() <= 1]
    if constant_cols:
        print(f"  Removing {len(constant_cols)} constant features")
        X = X.drop(columns=constant_cols)
    return X

