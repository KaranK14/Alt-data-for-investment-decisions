"""
Step 2: Extract MeSH/disease features and intervention flags.

Creates disease-specific flags (NSCLC, SCLC, breast cancer, etc.) and
intervention type flags (immunotherapy, TKI, ADC, etc.).
"""

import pandas as pd
from pathlib import Path
import sys
from typing import List

# Add Project 2 directory to path so we can import scripts
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.config import DATA_DIR, DATA_ENHANCED_DIR


def flag_from_keywords(text: str, keywords: List[str]) -> int:
    """Check if any keyword appears in text (case-insensitive)."""
    if not text or pd.isna(text):
        return 0
    text_lower = str(text).lower()
    return 1 if any(kw.lower() in text_lower for kw in keywords) else 0


def extract_disease_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Extract disease-specific flags from condition text."""
    print("\nExtracting disease flags...")
    
    result = df.copy()
    # Get condition_text column (should exist after step 1 parsing)
    if 'condition_text' not in df.columns:
        print("  WARNING: condition_text column not found! Creating empty flags.")
        condition_text = pd.Series([''] * len(df), index=df.index)
    else:
        condition_text = df['condition_text'].fillna('')
    
    # Disease-specific keywords
    disease_keywords = {
        'is_nsclc': ['non-small cell lung cancer', 'nsclc', 'non small cell lung cancer', 'non-small-cell lung cancer'],
        'is_sclc': ['small cell lung cancer', 'sclc'],
        'is_lung_cancer': ['lung cancer', 'lung carcinoma', 'lung neoplasm', 'lung tumor'],
        'is_breast_cancer': ['breast cancer', 'breast carcinoma', 'breast neoplasm'],
        'is_prostate_cancer': ['prostate cancer', 'prostate carcinoma', 'prostate neoplasm'],
        'is_colorectal_cancer': ['colorectal cancer', 'colon cancer', 'rectal cancer'],
        'is_gastric_cancer': ['gastric cancer', 'stomach cancer', 'gastric carcinoma'],
        'is_pancreatic_cancer': ['pancreatic cancer', 'pancreas cancer', 'pancreatic carcinoma'],
        'is_hepatocellular': ['hepatocellular', 'liver cancer', 'hcc'],
        'is_renal': ['renal cell', 'kidney cancer', 'rcc'],
        'is_bladder': ['bladder cancer', 'bladder carcinoma'],
        'is_ovarian': ['ovarian cancer', 'ovary cancer'],
        'is_cervical': ['cervical cancer', 'cervix cancer'],
        'is_endometrial': ['endometrial cancer', 'endometrium cancer'],
        'is_esophageal': ['esophageal cancer', 'oesophageal cancer'],
        'is_head_neck': ['head and neck cancer', 'head neck cancer'],
        'is_brain': ['brain cancer', 'brain tumor', 'glioma'],
        'is_thyroid': ['thyroid cancer', 'thyroid carcinoma'],
        'is_sarcoma': ['sarcoma'],
        'is_lymphoma': ['lymphoma'],
        'is_myeloma': ['myeloma', 'multiple myeloma'],
        'is_aml': ['acute myeloid leukemia', 'aml'],
        'is_all': ['acute lymphoblastic leukemia', 'all'],
        'is_cll': ['chronic lymphocytic leukemia', 'cll'],
        'is_cml': ['chronic myeloid leukemia', 'cml'],
    }
    
    # Extract flags
    for flag_name, keywords in disease_keywords.items():
        result[flag_name] = condition_text.apply(lambda x: flag_from_keywords(x, keywords))
    
    # is_other_cancer: contains "cancer" but none of the specific flags
    cancer_keywords = ["cancer", "carcinoma", "neoplasm", "tumor", "tumour", "malignancy", "malignant"]
    has_cancer_term = condition_text.apply(lambda x: flag_from_keywords(x, cancer_keywords))
    
    specific_flags = [col for col in result.columns if col.startswith('is_') and col != 'nct_id']
    has_specific_flag = result[specific_flags].sum(axis=1) > 0
    
    result['is_other_cancer'] = ((has_cancer_term == 1) & (has_specific_flag == 0)).astype(int)
    
    # is_other: catch-all for trials that don't match any specific or general cancer flag
    all_flags = [col for col in result.columns if col.startswith('is_') and col not in ['nct_id', 'is_multicountry', 'is_randomized', 'is_blinded']]
    result['is_other'] = (result[all_flags].sum(axis=1) == 0).astype(int)
    
    print(f"  Created {len(disease_keywords) + 2} disease flags")
    
    return result


def extract_intervention_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Extract intervention type flags."""
    print("\nExtracting intervention flags...")
    
    result = df.copy()
    # Get intervention text columns (should exist after step 1 parsing)
    intervention_text = df['intervention_text'].fillna('') if 'intervention_text' in df.columns else pd.Series([''] * len(df), index=df.index)
    interventions_str = df['interventions'].fillna('') if 'interventions' in df.columns else pd.Series([''] * len(df), index=df.index)
    
    combined_text = intervention_text.astype(str) + ' ' + interventions_str.astype(str)
    
    # Intervention keywords
    intervention_keywords = {
        'is_immunotherapy': ['immunotherapy', 'immune checkpoint', 'pd-1', 'pd1', 'pd-l1', 'pdl1', 'ctla-4', 'nivolumab', 'pembrolizumab', 'atezolizumab'],
        'is_tki': ['tyrosine kinase inhibitor', 'tki', 'erlotinib', 'gefitinib', 'osimertinib', 'afatinib', 'icotinib'],
        'is_adc': ['antibody drug conjugate', 'adc', 'trastuzumab emtansine', 'trastuzumab deruxtecan'],
        'is_chemo': ['chemotherapy', 'chemo', 'cisplatin', 'carboplatin', 'paclitaxel', 'docetaxel', 'pemetrexed'],
        'is_hormone': ['hormone therapy', 'hormonal therapy', 'tamoxifen', 'aromatase inhibitor'],
        'is_cell_therapy': ['car-t', 'cart', 'cell therapy', 't-cell therapy'],
        'is_targeted': ['targeted therapy', 'targeted treatment', 'molecularly targeted'],
        'is_radiotherapy': ['radiotherapy', 'radiation therapy', 'radiation treatment'],
        'is_surgery': ['surgery', 'surgical', 'resection'],
    }
    
    # Extract flags
    for flag_name, keywords in intervention_keywords.items():
        result[flag_name] = combined_text.apply(lambda x: flag_from_keywords(x, keywords))
    
    print(f"  Created {len(intervention_keywords)} intervention flags")
    
    return result


def main():
    """Main extraction function."""
    print("=" * 80)
    print("Step 2: Extracting MeSH/Disease and Intervention Features")
    print("=" * 80)
    
    # Load parsed data
    input_path = DATA_DIR / "01_parsed_trials.parquet"
    if not input_path.exists():
        print(f"\nERROR: Input file not found: {input_path}")
        print("Please run step 1 (01_parse_raw_data.py) first.")
        return
    
    df = pd.read_parquet(input_path)
    print(f"\nLoaded {len(df):,} trials from step 1")
    
    # Extract disease flags
    df = extract_disease_flags(df)
    
    # Extract intervention flags
    df = extract_intervention_flags(df)
    
    # Save results
    DATA_ENHANCED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DATA_ENHANCED_DIR / "02_trials_with_mesh.parquet"
    df.to_parquet(output_path, index=False)
    
    print(f"\n✓ Saved to: {output_path}")
    print(f"  Shape: {df.shape[0]:,} trials × {df.shape[1]:,} columns")
    
    # Print summary
    disease_flags = [col for col in df.columns if col.startswith('is_') and any(d in col for d in ['nsclc', 'sclc', 'breast', 'prostate', 'colorectal', 'gastric', 'pancreatic', 'other_cancer', 'other'])]
    intervention_flags = [col for col in df.columns if col.startswith('is_') and any(i in col for i in ['immuno', 'tki', 'adc', 'chemo', 'hormone', 'cell', 'targeted', 'radio', 'surgery'])]
    
    print(f"\nSummary:")
    print(f"  Disease flags: {len(disease_flags)}")
    print(f"  Intervention flags: {len(intervention_flags)}")
    
    # Show flag counts
    print(f"\n  Top disease flags:")
    for flag in disease_flags[:5]:
        count = df[flag].sum()
        print(f"    {flag}: {count:,} trials")


if __name__ == "__main__":
    main()

