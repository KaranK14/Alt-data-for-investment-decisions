"""
Step 6: Combine all features into final dataset.

This is the final preprocessing step before model training.
Creates the dataset that will be used for training.
"""

import pandas as pd
from pathlib import Path
import sys

# Add Project 2 directory to path so we can import scripts
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.config import DATA_ENHANCED_DIR


def main():
    """Main feature combination function."""
    print("=" * 80)
    print("Step 6: Combining All Features")
    print("=" * 80)
    
    # Load data
    input_path = DATA_ENHANCED_DIR / "05_trials_with_pca.parquet"
    if not input_path.exists():
        print(f"\nERROR: Input file not found: {input_path}")
        print("Please run step 5 (05_apply_pca.py) first.")
        return
    
    df = pd.read_parquet(input_path)
    print(f"\nLoaded {len(df):,} trials from step 5")
    print(f"  Total columns: {len(df.columns)}")
    
    # Drop original embedding columns (keep only PCA embeddings)
    embedding_cols = [col for col in df.columns if col.startswith('embedding_')]
    if embedding_cols:
        print(f"\nDropping {len(embedding_cols)} original embedding columns (keeping PCA embeddings)")
        df = df.drop(columns=embedding_cols)
    
    # Recreate label_feasible to ensure correct labeling (excludes ongoing trials)
    # IMPORTANT: Only label trials with FINAL status (completed or terminated).
    # Exclude ongoing trials because their final outcome is unknown.
    print("\nCreating/updating label_feasible from overall_status...")
    print("  Only final-status trials will be labeled (ongoing trials excluded)")
    
    status = df.get('overall_status', '').astype(str).str.upper()
    df['label_feasible'] = None  # Initialize as None
    df.loc[status == 'COMPLETED', 'label_feasible'] = 1
    df.loc[status.isin(['TERMINATED', 'WITHDRAWN', 'SUSPENDED']), 'label_feasible'] = 0
    # All other statuses remain None (will be excluded)
    
    # Drop rows with missing labels
    before = len(df)
    df = df[df['label_feasible'].notna()].copy()
    after = len(df)
    
    if before > after:
        print(f"\nDropped {before - after:,} trials with missing labels (ongoing/unknown status)")
        print(f"  These trials were excluded because their final outcome is unknown")
    
    # Validation: Verify that ongoing trials are excluded
    if 'overall_status' in df.columns:
        ongoing_statuses = ['RECRUITING', 'ACTIVE_NOT_RECRUITING', 'ENROLLING_BY_INVITATION', 
                           'NOT_YET_RECRUITING', 'UNKNOWN']
        ongoing_in_final = df[df['overall_status'].isin(ongoing_statuses)]
        if len(ongoing_in_final) > 0:
            print(f"\nWARNING: {len(ongoing_in_final)} ongoing trials found in final dataset!")
            print(f"  Statuses: {ongoing_in_final['overall_status'].value_counts().to_dict()}")
        else:
            print(f"\n[OK] Validation passed: No ongoing trials in final dataset")
    
    # Save final dataset
    output_path = DATA_ENHANCED_DIR / "oncology_phase23_enhanced_hist.parquet"
    df.to_parquet(output_path, index=False)
    
    print(f"\n[OK] Saved final dataset to: {output_path}")
    print(f"  Shape: {df.shape[0]:,} trials × {df.shape[1]:,} columns")
    
    # Print summary
    print(f"\nSummary:")
    print(f"  Label distribution:")
    if 'label_feasible' in df.columns:
        label_counts = df['label_feasible'].value_counts()
        completed = label_counts.get(1, 0)
        not_completed = label_counts.get(0, 0)
        print(f"    Completed (1): {completed:,} ({completed/len(df)*100:.1f}%)")
        print(f"    Not Completed (0): {not_completed:,} ({not_completed/len(df)*100:.1f}%)")
    
    # Count feature types
    pca_cols = [c for c in df.columns if c.startswith('pca_emb_')]
    disease_flags = [c for c in df.columns if c.startswith('is_') and any(d in c for d in ['nsclc', 'sclc', 'breast', 'prostate', 'colorectal', 'gastric', 'pancreatic', 'other_cancer', 'other'])]
    intervention_flags = [c for c in df.columns if c.startswith('is_') and any(i in c for i in ['immuno', 'tki', 'adc', 'chemo', 'hormone', 'cell', 'targeted', 'radio', 'surgery'])]
    complexity_cols = [c for c in df.columns if 'eligibility' in c or 'inclusion' in c or 'exclusion' in c]
    
    print(f"\n  Feature breakdown:")
    print(f"    PCA embeddings: {len(pca_cols)}")
    print(f"    Disease flags: {len(disease_flags)}")
    print(f"    Intervention flags: {len(intervention_flags)}")
    print(f"    Complexity features: {len(complexity_cols)}")
    print(f"    Other features: {len(df.columns) - len(pca_cols) - len(disease_flags) - len(intervention_flags) - len(complexity_cols) - 2}")  # -2 for nct_id and label
    
    print("\n[OK] Pipeline complete! Ready for model training.")


if __name__ == "__main__":
    main()

