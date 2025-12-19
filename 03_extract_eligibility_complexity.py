"""
Step 3: Extract eligibility criteria complexity features.

Calculates complexity metrics from eligibility criteria text.
"""

import pandas as pd
from pathlib import Path
import sys
import re

# Add Project 2 directory to path so we can import scripts
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.config import DATA_DIR, DATA_ENHANCED_DIR


def calculate_eligibility_complexity(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate complexity metrics from eligibility criteria text."""
    print("\nCalculating eligibility complexity...")
    
    result = df.copy()
    eligibility_text = df.get('eligibility_criteria_text', '').fillna('')
    
    # Text length features
    result['eligibility_text_len'] = eligibility_text.str.len()
    
    # Count inclusions and exclusions
    result['num_inclusion_criteria'] = eligibility_text.str.count(r'(?i)inclusion', flags=re.IGNORECASE)
    result['num_exclusion_criteria'] = eligibility_text.str.count(r'(?i)exclusion', flags=re.IGNORECASE)
    
    # Extract inclusion and exclusion sections separately
    def extract_section(text, section_name):
        if not text:
            return ''
        pattern = rf'(?i){section_name}.*?(?=(?:exclusion|inclusion|$))'
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        return match.group(0) if match else ''
    
    result['inclusion_section'] = eligibility_text.apply(lambda x: extract_section(x, 'inclusion'))
    result['exclusion_section'] = eligibility_text.apply(lambda x: extract_section(x, 'exclusion'))
    
    result['inclusion_text_len'] = result['inclusion_section'].str.len()
    result['exclusion_text_len'] = result['exclusion_section'].str.len()
    
    # Count conditions in inclusion/exclusion
    result['num_inclusion_items'] = result['inclusion_section'].str.count(r'[•\-\*]')
    result['num_exclusion_items'] = result['exclusion_section'].str.count(r'[•\-\*]')
    
    # Drop intermediate columns
    result = result.drop(columns=['inclusion_section', 'exclusion_section'])
    
    print(f"  Created {len([c for c in result.columns if 'eligibility' in c or 'inclusion' in c or 'exclusion' in c])} complexity features")
    
    return result


def main():
    """Main extraction function."""
    print("=" * 80)
    print("Step 3: Extracting Eligibility Complexity Features")
    print("=" * 80)
    
    # Load data
    input_path = DATA_ENHANCED_DIR / "02_trials_with_mesh.parquet"
    if not input_path.exists():
        print(f"\nERROR: Input file not found: {input_path}")
        print("Please run step 2 (02_extract_mesh_features.py) first.")
        return
    
    df = pd.read_parquet(input_path)
    print(f"\nLoaded {len(df):,} trials from step 2")
    
    # Calculate complexity
    df = calculate_eligibility_complexity(df)
    
    # Save results
    DATA_ENHANCED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DATA_ENHANCED_DIR / "03_trials_with_eligibility.parquet"
    df.to_parquet(output_path, index=False)
    
    print(f"\n✓ Saved to: {output_path}")
    print(f"  Shape: {df.shape[0]:,} trials × {df.shape[1]:,} columns")
    
    # Print summary
    complexity_cols = [c for c in df.columns if 'eligibility' in c or 'inclusion' in c or 'exclusion' in c]
    print(f"\nSummary:")
    print(f"  Complexity features: {len(complexity_cols)}")
    print(f"  Mean eligibility text length: {df['eligibility_text_len'].mean():.0f} characters")
    print(f"  Mean inclusion items: {df['num_inclusion_items'].mean():.1f}")
    print(f"  Mean exclusion items: {df['num_exclusion_items'].mean():.1f}")


if __name__ == "__main__":
    main()

