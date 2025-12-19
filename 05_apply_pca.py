"""
Step 5: Apply PCA dimensionality reduction to embeddings.

Reduces embedding dimensions while retaining 95% of variance.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib

# Add Project 2 directory to path so we can import scripts
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.config import (
    DATA_ENHANCED_DIR,
    PCA_VARIANCE_THRESHOLD,
    MODELS_DIR,
)


def main():
    """Main PCA function."""
    print("=" * 80)
    print("Step 5: Applying PCA to Embeddings")
    print("=" * 80)
    
    # Load data
    input_path = DATA_ENHANCED_DIR / "04_trials_with_embeddings.parquet"
    if not input_path.exists():
        print(f"\nERROR: Input file not found: {input_path}")
        print("Please run step 4 (04_generate_embeddings.py) first.")
        return
    
    df = pd.read_parquet(input_path)
    print(f"\nLoaded {len(df):,} trials from step 4")
    
    # Identify embedding columns
    embedding_cols = [col for col in df.columns if col.startswith('embedding_')]
    print(f"\nFound {len(embedding_cols)} embedding columns")
    
    if len(embedding_cols) == 0:
        print("ERROR: No embedding columns found!")
        return
    
    # Extract embeddings
    X_embed = df[embedding_cols].values
    
    # Handle missing values (trials with no text)
    valid_mask = ~np.isnan(X_embed).all(axis=1)
    X_valid = X_embed[valid_mask]
    ids_valid = df.index[valid_mask]
    ids_null = df.index[~valid_mask]
    
    print(f"\nValid embeddings: {len(X_valid):,} trials")
    print(f"Missing embeddings: {len(ids_null):,} trials")
    
    if len(X_valid) == 0:
        print("ERROR: No valid embeddings found!")
        return
    
    # Standardize embeddings
    print("\nStandardizing embeddings...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_valid)
    
    print(f"  Mean: {X_scaled.mean(axis=0).mean():.6f} (should be ~0)")
    print(f"  Std: {X_scaled.std(axis=0).mean():.6f} (should be ~1)")
    
    # Apply PCA
    print(f"\nApplying PCA (retaining {PCA_VARIANCE_THRESHOLD*100:.1f}% variance)...")
    pca = PCA(n_components=PCA_VARIANCE_THRESHOLD)
    X_pca = pca.fit_transform(X_scaled)
    
    n_components = X_pca.shape[1]
    explained_variance = pca.explained_variance_ratio_.sum()
    
    print(f"  Original dimension: {X_valid.shape[1]}")
    print(f"  Reduced dimension: {n_components}")
    print(f"  Compression ratio: {X_valid.shape[1] / n_components:.2f}x")
    print(f"  Variance explained: {explained_variance*100:.2f}%")
    
    # Create PCA embedding columns
    print("\nCreating PCA embedding DataFrame...")
    pca_df = pd.DataFrame(index=df.index)
    
    # Add PCA embeddings for valid trials
    for i in range(n_components):
        pca_df[f'pca_emb_{i}'] = np.nan
    
    pca_df.loc[ids_valid, [f'pca_emb_{i}' for i in range(n_components)]] = X_pca
    
    # Add PCA columns to main dataframe
    for col in pca_df.columns:
        df[col] = pca_df[col].values
    
    # Save PCA model and scaler
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    pca_path = MODELS_DIR / "pca_model.pkl"
    scaler_path = MODELS_DIR / "pca_scaler.pkl"
    
    joblib.dump(pca, pca_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"\n✓ Saved PCA model to: {pca_path}")
    print(f"✓ Saved scaler to: {scaler_path}")
    
    # Save results
    DATA_ENHANCED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DATA_ENHANCED_DIR / "05_trials_with_pca.parquet"
    df.to_parquet(output_path, index=False)
    
    print(f"\n✓ Saved to: {output_path}")
    print(f"  Shape: {df.shape[0]:,} trials × {df.shape[1]:,} columns")
    print(f"  PCA embedding columns: {n_components}")


if __name__ == "__main__":
    main()

