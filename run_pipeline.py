"""
Master Pipeline Execution Script

Runs the entire Project 2 pipeline in the correct order.

Usage:
    python scripts/run_pipeline.py [--step START_STEP] [--end END_STEP] [--skip-step STEP]
    
Examples:
    python scripts/run_pipeline.py                    # Run all steps
    python scripts/run_pipeline.py --step 5           # Start from step 5
    python scripts/run_pipeline.py --step 1 --end 4   # Run steps 1-4
    python scripts/run_pipeline.py --skip-step 4      # Skip step 4 (embeddings)
"""

import subprocess
import sys
from pathlib import Path
import argparse

# Pipeline steps in order
PIPELINE_STEPS = [
    ("01_parse_raw_data.py", "Parse raw JSON files"),
    ("02_extract_mesh_features.py", "Extract MeSH/disease and intervention features"),
    ("03_extract_eligibility_complexity.py", "Calculate eligibility complexity"),
    ("04_generate_embeddings.py", "Generate BioLinkBERT embeddings (SLOW)"),
    ("05_apply_pca.py", "Apply PCA to embeddings"),
    ("06_combine_features.py", "Combine all features"),
    ("07_train_models.py", "Train and evaluate models"),
]

# Script directory
SCRIPT_DIR = Path(__file__).parent
PIPELINE_DIR = SCRIPT_DIR / "pipeline"


def run_step(step_file: str, step_name: str, step_num: int) -> bool:
    """
    Run a single pipeline step.
    
    Returns:
        True if successful, False otherwise
    """
    script_path = PIPELINE_DIR / step_file
    
    if not script_path.exists():
        print(f"ERROR: Script not found: {script_path}")
        return False
    
    print(f"\n{'='*80}")
    print(f"STEP {step_num}: {step_name}")
    print(f"{'='*80}\n")
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            check=True,
            cwd=SCRIPT_DIR.parent
        )
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: Step {step_num} failed with return code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print(f"\n\nInterrupted by user at step {step_num}")
        return False


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Run Project 2 ML pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_pipeline.py
  python scripts/run_pipeline.py --step 5
  python scripts/run_pipeline.py --step 1 --end 4
  python scripts/run_pipeline.py --skip-step 4
        """
    )
    
    parser.add_argument(
        '--step',
        type=int,
        help='Start from this step number (1-indexed)'
    )
    
    parser.add_argument(
        '--end',
        type=int,
        help='End at this step number (1-indexed, inclusive)'
    )
    
    parser.add_argument(
        '--skip-step',
        type=int,
        action='append',
        dest='skip_steps',
        help='Skip this step number (can be used multiple times)'
    )
    
    args = parser.parse_args()
    
    # Determine which steps to run
    start_idx = (args.step - 1) if args.step else 0
    end_idx = args.end if args.end else len(PIPELINE_STEPS)
    skip_indices = [(s - 1) for s in (args.skip_steps or [])]
    
    steps_to_run = [
        (i, step) for i, step in enumerate(PIPELINE_STEPS, 1)
        if (i - 1) >= start_idx and (i - 1) < end_idx and (i - 1) not in skip_indices
    ]
    
    # Print plan
    print("=" * 80)
    print("PROJECT 2 ML PIPELINE")
    print("=" * 80)
    print(f"\nSteps to run ({len(steps_to_run)} of {len(PIPELINE_STEPS)}):")
    for step_num, (step_file, step_name) in steps_to_run:
        status = "SKIP" if (step_num - 1) in skip_indices else "RUN"
        print(f"  {step_num}. {step_name} [{status}]")
    print()
    
    # Run steps
    for step_num, (step_file, step_name) in steps_to_run:
        success = run_step(step_file, step_name, step_num)
        
        if not success:
            print(f"\n{'='*80}")
            print(f"PIPELINE STOPPED AT STEP {step_num}")
            print(f"{'='*80}")
            print(f"\nTo resume, run:")
            print(f"  python scripts/run_pipeline.py --step {step_num + 1}")
            sys.exit(1)
    
    print(f"\n{'='*80}")
    print("PIPELINE COMPLETE")
    print(f"{'='*80}")
    print("\nAll steps completed successfully!")
    print(f"\nResults saved to:")
    print(f"  - Final dataset: data_enhanced/oncology_phase23_enhanced_hist.parquet")
    print(f"  - Model results: results/model_comparison.csv")


if __name__ == "__main__":
    main()

