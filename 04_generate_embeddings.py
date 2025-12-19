"""
Step 4: Generate BioLinkBERT embeddings for text fields.

Embeds the following text fields:
- brief_title
- official_title
- brief_summary
- detailed_description
- eligibility_criteria_text

Uses the BioLinkBERT-base model.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# Add Project 2 directory to path so we can import scripts
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.config import (
    DATA_ENHANCED_DIR,
    BIOLINKBERT_MODEL_NAME,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DEVICE,
)


def generate_embeddings(texts: pd.Series, tokenizer, model, device, batch_size: int = 32):
    """Generate embeddings for a series of texts."""
    embeddings = []
    
    # Filter out NaN texts
    valid_indices = texts.notna()
    valid_texts = texts[valid_indices].fillna('').astype(str).tolist()
    
    if len(valid_texts) == 0:
        return np.array([])
    
    model.eval()
    
    # Process in batches
    with torch.no_grad():
        for i in tqdm(range(0, len(valid_texts), batch_size), desc="  Generating embeddings"):
            batch_texts = valid_texts[i:i + batch_size]
            
            # Tokenize
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(device)
            
            # Generate embeddings
            outputs = model(**encoded)
            
            # Use [CLS] token embedding (first token)
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(batch_embeddings)
    
    # Concatenate all batches
    all_embeddings = np.vstack(embeddings)
    
    # Create full array with NaN for missing texts
    full_embeddings = np.full((len(texts), all_embeddings.shape[1]), np.nan)
    full_embeddings[valid_indices] = all_embeddings
    
    return full_embeddings


def main():
    """Main embedding generation function."""
    print("=" * 80)
    print("Step 4: Generating BioLinkBERT Embeddings")
    print("=" * 80)
    
    # Load data
    input_path = DATA_ENHANCED_DIR / "03_trials_with_eligibility.parquet"
    if not input_path.exists():
        print(f"\nERROR: Input file not found: {input_path}")
        print("Please run step 3 (03_extract_eligibility_complexity.py) first.")
        return
    
    df = pd.read_parquet(input_path)
    print(f"\nLoaded {len(df):,} trials from step 3")
    
    # Set device
    device = EMBEDDING_DEVICE if torch.cuda.is_available() and EMBEDDING_DEVICE == "cuda" else "cpu"
    print(f"\nUsing device: {device}")
    
    # Load model and tokenizer
    print(f"\nLoading BioLinkBERT model: {BIOLINKBERT_MODEL_NAME}")
    print("  (This may take a few minutes on first run...)")
    
    tokenizer = AutoTokenizer.from_pretrained(BIOLINKBERT_MODEL_NAME)
    model = AutoModel.from_pretrained(BIOLINKBERT_MODEL_NAME).to(device)
    
    print(f"  Model loaded. Embedding dimension: {model.config.hidden_size}")
    
    # Text fields to embed
    text_fields = {
        'brief_title': 'brief_title',
        'official_title': 'official_title',
        'brief_summary': 'brief_summary',
        'detailed_description': 'detailed_description',
        'eligibility_criteria_text': 'eligibility_criteria_text',
    }
    
    # Combine text fields for embedding (row-wise)
    print("\nCombining text fields...")
    text_parts = []
    for field in text_fields.values():
        if field in df.columns:
            text_parts.append(df[field].fillna('').astype(str))
    
    if text_parts:
        # Combine Series row-wise using pandas
        df['description_text'] = text_parts[0]
        for part in text_parts[1:]:
            df['description_text'] = df['description_text'] + ' ' + part
    else:
        df['description_text'] = ''
    
    # Generate embeddings
    print("\nGenerating embeddings for combined text...")
    embeddings = generate_embeddings(
        df['description_text'],
        tokenizer,
        model,
        device,
        batch_size=EMBEDDING_BATCH_SIZE
    )
    
    # Save embeddings separately
    embedding_cols = [f'embedding_{i}' for i in range(embeddings.shape[1])]
    embedding_df = pd.DataFrame(embeddings, columns=embedding_cols, index=df.index)
    
    # Add embedding columns to main dataframe
    for col in embedding_cols:
        df[col] = embedding_df[col].values
    
    print(f"\n✓ Generated embeddings: {embeddings.shape[1]} dimensions")
    
    # Count missing embeddings
    n_missing = df[embedding_cols].isna().all(axis=1).sum()
    print(f"  Missing embeddings: {n_missing:,} trials ({n_missing/len(df)*100:.1f}%)")
    
    # Save results
    DATA_ENHANCED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DATA_ENHANCED_DIR / "04_trials_with_embeddings.parquet"
    df.to_parquet(output_path, index=False)
    
    print(f"\n✓ Saved to: {output_path}")
    print(f"  Shape: {df.shape[0]:,} trials × {df.shape[1]:,} columns")
    print(f"  Embedding columns: {len(embedding_cols)}")


if __name__ == "__main__":
    main()

