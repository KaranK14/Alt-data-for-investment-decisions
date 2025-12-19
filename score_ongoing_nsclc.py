"""
Deployment Script: Score Ongoing NSCLC Phase 2/3 Industry Interventional Trials

This script applies the trained leakage-free model to ongoing NSCLC trials
and generates investor-friendly rankings, risk buckets, and visualizations.

Usage:
    python scripts/deployment/score_ongoing_nsclc.py
"""

import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import warnings
from typing import Tuple, Dict, List, Optional
from datetime import datetime

# Add parent directory to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.config import (
    DATASET_PATH,
    ID_COL,
    MODELS_DIR,
    RESULTS_DIR,
    RANDOM_SEED,
)
from scripts.data_loading import (
    split_features_labels,
    remove_constant_features,
    identify_tabular_columns,
    identify_pca_columns,
)
from scripts.models.xgb_model import convert_to_numeric

# Suppress warnings
warnings.filterwarnings('ignore')

# Output directories
NSCLC_DIR = RESULTS_DIR / "nsclc"
NSCLC_DIR.mkdir(parents=True, exist_ok=True)
CHARTS_DIR = NSCLC_DIR / "charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

# Set style for plots
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except OSError:
    try:
        plt.style.use('seaborn-darkgrid')
    except OSError:
        plt.style.use('default')
sns.set_palette("husl")


def find_dataset() -> Path:
    """
    Find dataset for scoring ongoing trials.
    
    First tries to find a dataset with ongoing trials (e.g., 05_trials_with_pca.parquet).
    Falls back to enhanced_hist if needed (though it may not have ongoing trials).
    """
    data_dir = PROJECT_ROOT / "data_enhanced"
    
    # Prefer datasets that might have ongoing trials (before historical filtering)
    preferred_files = [
        "05_trials_with_pca.parquet",
        "04_trials_with_embeddings.parquet",
        "03_trials_with_eligibility.parquet",
    ]
    
    for filename in preferred_files:
        candidate = data_dir / filename
        if candidate.exists():
            # Check if it has ongoing trials
            try:
                import pandas as pd
                df_check = pd.read_parquet(candidate)
                if 'overall_status' in df_check.columns:
                    ongoing = df_check['overall_status'].isin(['RECRUITING', 'ACTIVE_NOT_RECRUITING', 'ENROLLING_BY_INVITATION'])
                    if ongoing.sum() > 0:
                        print(f"  Found dataset with ongoing trials: {filename}")
                        return candidate
            except:
                pass
    
    # Fallback to enhanced_hist
    if DATASET_PATH.exists():
        print(f"  Using enhanced_hist dataset (may not have ongoing trials)")
        return DATASET_PATH
    
    # Search for any enhanced_hist file
    if data_dir.exists():
        parquet_files = list(data_dir.glob("*enhanced_hist*.parquet"))
        if parquet_files:
            return max(parquet_files, key=lambda p: p.stat().st_mtime)
    
    raise FileNotFoundError(f"Dataset not found at {DATASET_PATH} or in {data_dir}")


def load_model_and_metadata() -> Tuple[any, Dict, Optional[any]]:
    """Load the saved model, metadata, and scaler if available."""
    model_path = MODELS_DIR / "best_model.pkl"
    metadata_path = MODELS_DIR / "best_model_metadata.json"
    scaler_path = MODELS_DIR / "best_model_scaler.pkl"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")
    
    print(f"Loading model from: {model_path}")
    model = joblib.load(model_path)
    
    print(f"Loading metadata from: {metadata_path}")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    scaler = None
    if scaler_path.exists():
        print(f"Loading scaler from: {scaler_path}")
        scaler = joblib.load(scaler_path)
    
    return model, metadata, scaler


def filter_deployment_cohort(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter trials to ongoing NSCLC trials.
    
    Note: Dataset is already filtered for Phase 2/3, Interventional, and Industry sponsored.
    We only need to filter for:
    - Disease = NSCLC (is_nsclc == 1 or condition text match)
    - Ongoing status only (overall_status in ['RECRUITING','ACTIVE_NOT_RECRUITING','ENROLLING_BY_INVITATION'])
    """
    print("\n" + "=" * 80)
    print("Filtering Deployment Cohort")
    print("=" * 80)
    print("  Note: Dataset already filtered for Phase 2/3, Interventional, Industry sponsored")
    
    initial_count = len(df)
    print(f"  Initial trials: {initial_count:,}")
    
    # 1. NSCLC filter
    if 'is_nsclc' in df.columns:
        df = df[df['is_nsclc'] == 1].copy()
        print(f"  After NSCLC filter (is_nsclc==1): {len(df):,}")
    else:
        # Fallback: check condition text
        if 'condition_text' in df.columns:
            nsclc_keywords = ['non small cell lung cancer', 'nsclc', 'non-small cell lung cancer', 
                            'non-small-cell lung cancer', 'non-small-cell lung carcinoma']
            mask = df['condition_text'].astype(str).str.lower().str.contains(
                '|'.join(nsclc_keywords), na=False, regex=True
            )
            df = df[mask].copy()
            print(f"  After NSCLC filter (condition text): {len(df):,}")
        else:
            print("  WARNING: No NSCLC flag or condition_text found, skipping NSCLC filter")
    
    # 2. Ongoing status only
    if 'overall_status' in df.columns:
        ongoing_statuses = ['RECRUITING', 'ACTIVE_NOT_RECRUITING', 'ENROLLING_BY_INVITATION']
        status_counts = df['overall_status'].value_counts()
        print(f"  Status distribution before filter:")
        for status, count in status_counts.items():
            print(f"    {status}: {count:,}")
        
        df = df[df['overall_status'].isin(ongoing_statuses)].copy()
        print(f"  After Ongoing status filter: {len(df):,}")
        
        if len(df) == 0:
            print("\n  WARNING: No ongoing trials found in dataset!")
            print("  The dataset may only contain historical (completed/terminated) trials.")
            print("  Consider using an earlier pipeline step (e.g., 05_trials_with_pca.parquet)")
            print("  that includes ongoing trials before the historical filtering step.")
    else:
        print("  WARNING: overall_status not found, skipping status filter")
    
    print(f"\n  Final deployment cohort: {len(df):,} trials")
    print(f"  Filtered out: {initial_count - len(df):,} trials")
    
    return df


def prepare_features(df: pd.DataFrame, metadata: Dict, scaler: Optional[any] = None) -> Tuple[np.ndarray, List[str]]:
    """
    Prepare features exactly as during training.
    
    Returns:
        Feature matrix (numpy array) and feature names
    """
    print("\n" + "=" * 80)
    print("Preparing Features")
    print("=" * 80)
    
    input_mode = metadata['input_mode']
    
    print(f"\nMode: {input_mode}")
    print(f"  Input dataframe shape: {df.shape}")
    print(f"  Input dataframe columns: {len(df.columns)}")
    
    # For deployment, we don't have labels, so we need to extract features manually
    # instead of using split_features_labels which drops rows with missing labels
    # Get feature columns based on mode (same logic as split_features_labels but without label requirement)
    if input_mode == "tabular":
        feature_cols = identify_tabular_columns(df, exclude_leakage=True)
    elif input_mode == "tabular_plus_embeddings":
        tabular_cols = identify_tabular_columns(df, exclude_leakage=True)
        pca_cols = identify_pca_columns(df)
        feature_cols = tabular_cols + pca_cols
    else:
        raise ValueError(f"Unknown input mode: {input_mode}")
    
    # Extract features (don't drop rows - we want all ongoing trials)
    X = df[feature_cols].copy()
    print(f"  After feature extraction:")
    print(f"    Features: {X.shape[1]}")
    print(f"    Trials: {X.shape[0]}")
    print(f"  Initial features: {X.shape[1]}")
    
    # Remove constant features
    X = remove_constant_features(X)
    print(f"  After removing constants: {X.shape[1]}")
    
    # Apply feature selection (use only features from metadata)
    expected_features = metadata.get('feature_names', [])
    if expected_features:
        available_features = [f for f in expected_features if f in X.columns]
        missing_features = [f for f in expected_features if f not in X.columns]
        
        if missing_features:
            print(f"  WARNING: {len(missing_features)} features missing from dataset")
            if len(missing_features) <= 10:
                print(f"    Missing: {missing_features}")
            else:
                print(f"    Missing (first 10): {missing_features[:10]}")
            
            # Add missing features as zeros (e.g., disease flags that don't apply to NSCLC-only cohort)
            for feat in missing_features:
                X[feat] = 0
            print(f"  Added {len(missing_features)} missing features as zeros")
        
        # Reorder to match expected feature order
        X = X[expected_features]
        print(f"  Final features: {X.shape[1]} (expected: {len(expected_features)})")
    
    # Apply model-specific preprocessing
    model_name = metadata['model_name']
    
    if model_name == 'XGBoost':
        X_processed = convert_to_numeric(X)
        feature_names = list(X_processed.columns)
        X_array = X_processed.values
        
    elif model_name == 'RandomForest':
        X_processed = X.select_dtypes(include=[np.number, 'bool']).fillna(0)
        for col in X_processed.columns:
            if X_processed[col].dtype == 'bool':
                X_processed[col] = X_processed[col].astype(int)
        feature_names = list(X_processed.columns)
        X_array = X_processed.values
        
    elif model_name == 'SVM':
        X_processed = X.select_dtypes(include=[np.number, 'bool']).fillna(0)
        for col in X_processed.columns:
            if X_processed[col].dtype == 'bool':
                X_processed[col] = X_processed[col].astype(int)
        feature_names = list(X_processed.columns)
        
        if scaler is not None:
            X_array = scaler.transform(X_processed)
        else:
            print("  WARNING: No scaler found for SVM model")
            X_array = X_processed.values
    else:
        # Default
        X_processed = X.select_dtypes(include=[np.number, 'bool']).fillna(0)
        for col in X_processed.columns:
            if X_processed[col].dtype == 'bool':
                X_processed[col] = X_processed[col].astype(int)
        feature_names = list(X_processed.columns)
        X_array = X_processed.values
    
    return X_array, feature_names


def compute_predictions(model: any, X: np.ndarray) -> np.ndarray:
    """Compute completion probabilities."""
    print("\n" + "=" * 80)
    print("Computing Predictions")
    print("=" * 80)
    
    if X.shape[0] == 0:
        print("  ERROR: No trials to predict (X has 0 rows)")
        print(f"  X shape: {X.shape}")
        raise ValueError("Cannot compute predictions: feature matrix has 0 rows. Check feature preparation.")
    
    p_completion = model.predict_proba(X)[:, 1]
    
    print(f"  Predictions computed: {len(p_completion):,} trials")
    if len(p_completion) > 0:
        print(f"  Min probability: {p_completion.min():.4f}")
        print(f"  Median probability: {np.median(p_completion):.4f}")
        print(f"  Max probability: {p_completion.max():.4f}")
    else:
        print("  WARNING: Empty predictions array")
    
    return p_completion


def create_rankings_and_buckets(p_completion: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Create rankings and risk buckets.
    
    Returns:
        rankings, buckets, bucket_info (with thresholds)
    """
    print("\n" + "=" * 80)
    print("Creating Rankings and Risk Buckets")
    print("=" * 80)
    
    # Rank (1 = highest probability)
    rankings = (-p_completion).argsort().argsort() + 1
    
    # Buckets: Low = top 30%, Medium = middle 40%, High = bottom 30%
    p30 = np.percentile(p_completion, 70)  # Top 30% threshold
    p70 = np.percentile(p_completion, 30)  # Bottom 30% threshold
    
    buckets = np.where(
        p_completion >= p30, 'LOW',
        np.where(p_completion <= p70, 'HIGH', 'MEDIUM')
    )
    
    bucket_info = {
        'low_threshold': float(p30),
        'high_threshold': float(p70),
        'low_count': int((buckets == 'LOW').sum()),
        'medium_count': int((buckets == 'MEDIUM').sum()),
        'high_count': int((buckets == 'HIGH').sum()),
    }
    
    print(f"  Bucket thresholds:")
    print(f"    LOW (top 30%): >= {p30:.4f} ({bucket_info['low_count']} trials)")
    print(f"    MEDIUM (middle 40%): {p70:.4f} < p < {p30:.4f} ({bucket_info['medium_count']} trials)")
    print(f"    HIGH (bottom 30%): <= {p70:.4f} ({bucket_info['high_count']} trials)")
    
    return rankings, buckets, bucket_info


def create_output_table(
    df: pd.DataFrame,
    p_completion: np.ndarray,
    rankings: np.ndarray,
    buckets: np.ndarray,
    metadata: Dict
) -> pd.DataFrame:
    """
    Create the master output table with all required columns and all model features.
    
    Includes:
    - All trial characteristics used for modeling (from metadata feature_names)
    - Important metadata columns (ID, status, sponsor, etc.)
    - Scoring outputs (p_completion, rank_nsclc, risk_bucket)
    """
    print("\n" + "=" * 80)
    print("Creating Output Table")
    print("=" * 80)
    
    output_cols = []
    
    # 1. Start with ID column (always first)
    if ID_COL in df.columns:
        output_cols.append(ID_COL)
    
    # 2. Add all model features that exist in the original dataframe
    # (exclude PCA features as they're derived and not interpretable)
    expected_features = metadata.get('feature_names', [])
    model_features_added = 0
    
    if expected_features:
        for feat in expected_features:
            # Skip PCA features (they're derived, not original trial characteristics)
            if feat.startswith('pca_emb'):
                continue
            # Include feature if it exists in original dataframe
            if feat in df.columns and feat not in output_cols:
                output_cols.append(feat)
                model_features_added += 1
    
    print(f"  Added {model_features_added} model features from original dataframe")
    
    # 3. Add important metadata columns (if not already included)
    required_cols = [
        'overall_status',
        'phase_simple',
        'sponsor_name',  # or lead_sponsor_name
        'intervention_model',
        'allocation',
        'start_year',
        'enrollment_planned_num',
        'num_countries',
        'number_of_facilities',
        'has_us_sites',
        'has_eu_sites',
        'has_china_sites',
    ]
    
    for col in required_cols:
        if col in df.columns and col not in output_cols:
            output_cols.append(col)
        elif col not in df.columns:
            # Try alternatives
            if col == 'sponsor_name':
                if 'lead_sponsor_name' in df.columns and 'lead_sponsor_name' not in output_cols:
                    output_cols.append('lead_sponsor_name')
            else:
                print(f"  WARNING: {col} not found, skipping")
    
    # 4. Add additional useful columns if available
    additional_cols = [
        'masking', 'primary_purpose', 'enrollment_actual_num',
        'num_arms', 'has_placebo', 'is_blinded', 'eligibility_complexity_score',
        'start_date', 'primary_completion_date', 'completion_date',
    ]
    
    for col in additional_cols:
        if col in df.columns and col not in output_cols:
            output_cols.append(col)
    
    # 5. Create output dataframe
    output_df = df[output_cols].copy()
    
    # 6. Add predictions and rankings (at the end for easy reference)
    output_df['p_completion'] = p_completion
    output_df['rank_nsclc'] = rankings
    output_df['risk_bucket'] = buckets
    
    # 7. Sort by rank
    output_df = output_df.sort_values('rank_nsclc').reset_index(drop=True)
    
    print(f"  Output table created: {len(output_df):,} rows × {len(output_df.columns)} columns")
    print(f"  Columns: {', '.join(output_df.columns[:10])}...")
    if len(output_df.columns) > 10:
        print(f"  ... and {len(output_df.columns) - 10} more columns")
    
    return output_df


def generate_charts(
    output_df: pd.DataFrame,
    p_completion: np.ndarray,
    bucket_info: Dict
):
    """Generate all required charts."""
    print("\n" + "=" * 80)
    print("Generating Charts")
    print("=" * 80)
    
    # (A) Probability distribution histogram
    print("  (A) Probability distribution histogram...")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(p_completion, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(bucket_info['low_threshold'], color='green', linestyle='--', linewidth=2, label='LOW threshold (70th percentile)')
    ax.axvline(bucket_info['high_threshold'], color='red', linestyle='--', linewidth=2, label='HIGH threshold (30th percentile)')
    ax.set_xlabel('Completion Probability', fontsize=12)
    ax.set_ylabel('Number of Trials', fontsize=12)
    ax.set_title('Distribution of Completion Probabilities', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "nsclc_prob_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved: nsclc_prob_distribution.png")
    
    # (B) Bucket counts bar chart
    print("  (B) Bucket counts bar chart...")
    bucket_counts = output_df['risk_bucket'].value_counts().reindex(['LOW', 'MEDIUM', 'HIGH'])
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['green', 'orange', 'red']
    bars = ax.bar(bucket_counts.index, bucket_counts.values, color=colors, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Risk Bucket', fontsize=12)
    ax.set_ylabel('Number of Trials', fontsize=12)
    ax.set_title('Trial Count by Risk Bucket', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "nsclc_bucket_counts.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved: nsclc_bucket_counts.png")
    
    # (C) Top 25 ranked table as image
    print("  (C) Top 25 ranked table...")
    # Select available columns
    table_cols = ['nct_id', 'rank_nsclc', 'p_completion', 'risk_bucket']
    if 'sponsor_name' in output_df.columns:
        table_cols.insert(1, 'sponsor_name')
    elif 'lead_sponsor_name' in output_df.columns:
        table_cols.insert(1, 'lead_sponsor_name')
    if 'phase_simple' in output_df.columns:
        table_cols.append('phase_simple')
    if 'overall_status' in output_df.columns:
        table_cols.append('overall_status')
    
    top25 = output_df.head(25)[table_cols].copy()
    
    # Rename sponsor column if needed for display
    if 'lead_sponsor_name' in top25.columns:
        top25 = top25.rename(columns={'lead_sponsor_name': 'sponsor_name'})
    
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis('tight')
    ax.axis('off')
    
    table_data = []
    # Build header based on available columns
    header = ['Rank', 'NCT ID']
    if 'sponsor_name' in top25.columns:
        header.append('Sponsor')
    if 'phase_simple' in top25.columns:
        header.append('Phase')
    if 'overall_status' in top25.columns:
        header.append('Status')
    header.extend(['P(Completion)', 'Risk'])
    table_data.append(header)
    
    for idx, row in top25.iterrows():
        row_data = [
            int(row['rank_nsclc']),
            row['nct_id']
        ]
        if 'sponsor_name' in top25.columns:
            sponsor = str(row.get('sponsor_name', 'N/A'))
            row_data.append(sponsor[:30] if len(sponsor) > 30 else sponsor)
        if 'phase_simple' in top25.columns:
            row_data.append(str(row.get('phase_simple', 'N/A')))
        if 'overall_status' in top25.columns:
            row_data.append(str(row.get('overall_status', 'N/A')))
        row_data.extend([
            f"{row['p_completion']:.4f}",
            row['risk_bucket']
        ])
        table_data.append(row_data)
    
    # Calculate column widths dynamically
    n_cols = len(table_data[0])
    col_widths = [0.1] * n_cols  # Default width
    if n_cols >= 3:
        col_widths[1] = 0.15  # NCT ID
    if n_cols >= 4 and 'sponsor_name' in top25.columns:
        col_widths[2] = 0.25  # Sponsor
    if n_cols >= 6:
        col_widths[-2] = 0.12  # P(Completion)
        col_widths[-1] = 0.08  # Risk
    
    table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                     cellLoc='left', loc='center', colWidths=col_widths)
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax.set_title('Top 25 NSCLC Trials by Completion Probability', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "nsclc_top25_table.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved: nsclc_top25_table.png")
    
    # (D) Scatterplots
    # Enrollment vs probability
    print("  (D1) Enrollment vs probability scatterplot...")
    if 'enrollment_planned_num' in output_df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        for bucket in ['LOW', 'MEDIUM', 'HIGH']:
            mask = output_df['risk_bucket'] == bucket
            data = output_df[mask]
            if len(data) > 0:
                ax.scatter(data['enrollment_planned_num'], data['p_completion'],
                          label=bucket, alpha=0.6, s=50)
        
        ax.set_xlabel('Planned Enrollment (log scale)', fontsize=12)
        ax.set_ylabel('Completion Probability', fontsize=12)
        ax.set_title('Enrollment vs Completion Probability', fontsize=14, fontweight='bold')
        ax.set_xscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(CHARTS_DIR / "nsclc_enrollment_vs_prob.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    Saved: nsclc_enrollment_vs_prob.png")
    else:
        print("    WARNING: enrollment_planned_num not found, skipping enrollment scatterplot")
    
    # Facilities vs probability
    print("  (D2) Facilities vs probability scatterplot...")
    facilities_col = None
    for col in ['number_of_facilities', 'num_facilities', 'facility_count']:
        if col in output_df.columns:
            facilities_col = col
            break
    
    if facilities_col:
        fig, ax = plt.subplots(figsize=(10, 6))
        for bucket in ['LOW', 'MEDIUM', 'HIGH']:
            mask = output_df['risk_bucket'] == bucket
            data = output_df[mask]
            if len(data) > 0:
                ax.scatter(data[facilities_col], data['p_completion'],
                          label=bucket, alpha=0.6, s=50)
        
        ax.set_xlabel('Number of Facilities', fontsize=12)
        ax.set_ylabel('Completion Probability', fontsize=12)
        ax.set_title('Number of Facilities vs Completion Probability', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(CHARTS_DIR / "nsclc_facilities_vs_prob.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    Saved: nsclc_facilities_vs_prob.png")
    else:
        print("    WARNING: number_of_facilities not found, skipping facilities scatterplot")


def compute_shap_analysis(
    model: any,
    X: np.ndarray,
    feature_names: List[str],
    metadata: Dict,
    output_df: pd.DataFrame,
    max_samples: int = 500
) -> Optional[pd.DataFrame]:
    """
    Compute SHAP values for the scored cohort (optional, only if fast).
    
    Returns:
        DataFrame with top 15 non-PCA features by mean(|SHAP|), or None if skipped
    """
    print("\n" + "=" * 80)
    print("SHAP Analysis (Optional)")
    print("=" * 80)
    
    model_name = metadata['model_name']
    
    # Only do SHAP for tree-based models (fast)
    if model_name not in ['XGBoost', 'RandomForest']:
        print(f"  Skipping SHAP: {model_name} not supported (only XGBoost/RandomForest)")
        return None
    
    try:
        import shap
        
        # Sample if too large
        if len(X) > max_samples:
            print(f"  Sampling {max_samples} trials from {len(X)} for SHAP")
            sample_idx = np.random.choice(len(X), max_samples, replace=False)
            X_shap = X[sample_idx]
        else:
            sample_idx = np.arange(len(X))
            X_shap = X
        
        print(f"  Computing SHAP values for {len(X_shap)} trials...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_shap)
        
        # For binary classification, get class 1 SHAP values
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # Compute mean absolute SHAP per feature
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'mean_abs_shap': mean_abs_shap
        })
        
        # Exclude PCA features
        importance_df = importance_df[~importance_df['feature'].str.startswith('pca_emb')]
        
        # Get top 15
        top15 = importance_df.nlargest(15, 'mean_abs_shap').sort_values('mean_abs_shap', ascending=False)
        
        # Save CSV
        top15.to_csv(NSCLC_DIR / "nsclc_shap_top15.csv", index=False)
        print(f"  Saved: nsclc_shap_top15.csv")
        
        # Plot bar chart
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(range(len(top15)), top15['mean_abs_shap'], color='steelblue', alpha=0.7)
        ax.set_yticks(range(len(top15)))
        # Replace underscores with spaces and increase font size
        feature_labels = [feat.replace('_', ' ') for feat in top15['feature']]
        ax.set_yticklabels(feature_labels, fontsize=16, fontweight='medium')
        ax.set_xlabel('Mean(|SHAP|)', fontsize=14, fontweight='bold')
        ax.set_title('Top 15 Non-PCA Features by SHAP Importance (NSCLC Cohort)', fontsize=16, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
        # Increase tick label font size
        ax.tick_params(axis='x', labelsize=12)
        plt.tight_layout()
        plt.savefig(CHARTS_DIR / "nsclc_shap_top15.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: nsclc_shap_top15.png")
        
        # Generate beeswarm plot
        print("  Generating beeswarm plot...")
        
        # Get indices of non-PCA features
        non_pca_mask = ~np.array([f.startswith('pca_emb') for f in feature_names])
        non_pca_indices = np.where(non_pca_mask)[0]
        
        # Filter SHAP values and feature names to non-PCA only
        shap_values_non_pca = shap_values[:, non_pca_indices]
        feature_names_non_pca = [feature_names[i] for i in non_pca_indices]
        
        # Format feature names for display (replace underscores, title case)
        feature_names_display = [name.replace('_', ' ').title() for name in feature_names_non_pca]
        
        # Set larger font sizes for SHAP plots
        plt.rcParams.update({'font.size': 14, 'axes.labelsize': 14, 'axes.titlesize': 16, 
                             'xtick.labelsize': 12, 'ytick.labelsize': 14})
        
        # Generate beeswarm plot (dot plot)
        shap.summary_plot(
            shap_values_non_pca,
            X_shap[:, non_pca_indices],
            feature_names=feature_names_display,
            show=False,
            plot_type="dot",
            max_display=20  # Show top 20 features
        )
        plt.tight_layout()
        plt.savefig(CHARTS_DIR / "nsclc_shap_beeswarm.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: nsclc_shap_beeswarm.png")
        
        # Reset font size to default
        plt.rcParams.update(plt.rcParamsDefault)
        
        return top15
        
    except ImportError:
        print("  Skipping SHAP: shap library not installed")
        return None
    except Exception as e:
        print(f"  Skipping SHAP: Error occurred - {str(e)}")
        return None


def main():
    """Main execution function."""
    print("=" * 80)
    print("NSCLC Ongoing Trials Scoring - Deployment Script")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. Load dataset
    dataset_path = find_dataset()
    print(f"\nLoading dataset from: {dataset_path}")
    df = pd.read_parquet(dataset_path)
    print(f"  Total trials loaded: {len(df):,}")
    
    # 2. Load model
    model, metadata, scaler = load_model_and_metadata()
    print(f"\nModel loaded:")
    print(f"  Type: {metadata['model_name']}")
    print(f"  Input mode: {metadata['input_mode']}")
    print(f"  ROC-AUC: {metadata['roc_auc']:.4f}")
    
    # 3. Filter deployment cohort
    df_cohort = filter_deployment_cohort(df)
    
    if len(df_cohort) == 0:
        print("\n" + "!" * 80)
        print("ERROR: No trials match the deployment cohort criteria!")
        print("!" * 80)
        return
    
    # 4. Prepare features
    X, feature_names = prepare_features(df_cohort, metadata, scaler)
    
    # 5. Compute predictions
    p_completion = compute_predictions(model, X)
    
    # 6. Create rankings and buckets
    rankings, buckets, bucket_info = create_rankings_and_buckets(p_completion)
    
    # 7. Create output table
    output_df = create_output_table(df_cohort, p_completion, rankings, buckets, metadata)
    
    # 8. Save outputs
    print("\n" + "=" * 80)
    print("Saving Outputs")
    print("=" * 80)
    
    csv_path = NSCLC_DIR / "nsclc_ongoing_scored.csv"
    parquet_path = NSCLC_DIR / "nsclc_ongoing_scored.parquet"
    
    output_df.to_csv(csv_path, index=False)
    print(f"  Saved CSV: {csv_path}")
    
    output_df.to_parquet(parquet_path, index=False)
    print(f"  Saved Parquet: {parquet_path}")
    
    # 9. Generate charts
    generate_charts(output_df, p_completion, bucket_info)
    
    # 10. Optional SHAP analysis
    shap_df = compute_shap_analysis(model, X, feature_names, metadata, output_df)
    
    # 11. Summary statistics
    print("\n" + "=" * 80)
    print("Summary Statistics")
    print("=" * 80)
    print(f"  Total in deployment cohort: {len(df_cohort):,}")
    print(f"  Completion probability stats:")
    print(f"    Min: {p_completion.min():.4f}")
    print(f"    Median: {np.median(p_completion):.4f}")
    print(f"    Max: {p_completion.max():.4f}")
    print(f"    Mean: {p_completion.mean():.4f}")
    print(f"    Std: {p_completion.std():.4f}")
    print(f"\n  Bucket thresholds:")
    print(f"    LOW (top 30%): >= {bucket_info['low_threshold']:.4f}")
    print(f"    HIGH (bottom 30%): <= {bucket_info['high_threshold']:.4f}")
    
    print(f"\n  Top 10 trials:")
    top10 = output_df.head(10)
    for idx, row in top10.iterrows():
        sponsor = row.get('sponsor_name', row.get('lead_sponsor_name', 'N/A'))
        if isinstance(sponsor, str) and len(sponsor) > 30:
            sponsor = sponsor[:30] + "..."
        print(f"    {int(row['rank_nsclc'])}. {row['nct_id']} | {sponsor} | P={row['p_completion']:.4f} | {row['risk_bucket']}")
    
    # Check for missing critical columns
    critical_cols = ['nct_id', 'overall_status', 'phase_simple', 'p_completion', 'rank_nsclc', 'risk_bucket']
    missing = [col for col in critical_cols if col not in output_df.columns]
    if missing:
        print(f"\n  WARNING: Missing critical columns: {missing}")
    else:
        print(f"\n  [OK] All critical columns present")
    
    print("\n" + "=" * 80)
    print("DEPLOYMENT SCORING COMPLETED")
    print("=" * 80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nOutput files:")
    print(f"  - {csv_path}")
    print(f"  - {parquet_path}")
    print(f"  - Charts in: {CHARTS_DIR}")


if __name__ == "__main__":
    main()

