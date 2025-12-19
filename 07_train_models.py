"""
Step 7: Train and evaluate models.

This step:
1. Loads the final dataset
2. Splits into train/val/test
3. Applies XGBoost-based feature selection (if enabled)
4. Trains models with imbalance handling:
   - SVM (tabular only)
   - Random Forest (tabular + enhanced)
   - XGBoost (tabular + enhanced)
   - Dual-tower Neural Network (enhanced only)
5. Saves results to CSV and Markdown

All the training logic is in run_experiments.py which this script calls.
"""

import subprocess
import sys
from pathlib import Path

# Add scripts directory to path (run_experiments.py is now in scripts folder)
SCRIPTS_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

if __name__ == "__main__":
    print("=" * 80)
    print("Step 7: Training Models")
    print("=" * 80)
    print("\nThis step will:")
    print("  1. Load the final dataset")
    print("  2. Split into train/val/test sets")
    print("  3. Apply XGBoost-based feature selection (if enabled)")
    print("  4. Train models with multiple imbalance strategies:")
    print("     - SVM (tabular only, class_weight)")
    print("     - Random Forest (tabular + enhanced, 3 strategies)")
    print("     - XGBoost (tabular + enhanced, 3 strategies)")
    print("     - Dual-tower NN (enhanced only, oversampling)")
    print("  5. Run 10 iterations per configuration")
    print("  6. Save results to results/model_comparison.csv")
    print("\n" + "=" * 80)
    print("Running full model training pipeline...\n")
    
    # Import and run the main experiment script (now in scripts folder)
    from run_experiments import main
    main()
