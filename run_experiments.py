"""
Main orchestration script for running all ML experiments.

Usage:
    python run_experiments.py [--no-feature-selection] [--skip-models svm nn]
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime
import sys

# Add parent directory to path so we can import scripts as a package
PROJECT_ROOT_PARENT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT_PARENT))

# Import from scripts package (using absolute imports)
from scripts.config import (
    PROJECT_ROOT,
    RESULTS_DIR,
    RESULTS_CSV,
    RESULTS_MD,
    MODELS_DIR,
    ENABLE_FEATURE_SELECTION,
    RANDOM_SEED,
)
from scripts.data_loading import (
    load_dataset,
    split_features_labels,
    train_val_test_split,
    identify_tabular_columns,
    identify_pca_columns,
    remove_constant_features,
)
from scripts.feature_selection import select_features_with_xgboost, apply_feature_mask
from scripts.models.svm_model import train_evaluate_svm
from scripts.models.rf_model import train_evaluate_rf
from scripts.models.xgb_model import train_evaluate_xgb
from scripts.models.dual_tower_nn import train_evaluate_dual_tower_nn


def run_experiments(
    enable_feature_selection: bool = ENABLE_FEATURE_SELECTION,
    skip_models: list = None
):
    """Run all experiments."""
    if skip_models is None:
        skip_models = []
    
    print("=" * 80)
    print("CLINICAL TRIAL FEASIBILITY PREDICTION - FULL EXPERIMENT SUITE")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Feature selection: {enable_feature_selection}")
    print(f"Skipping models: {skip_models if skip_models else 'None'}")
    
    # ========================================================================
    # LOAD DATA
    # ========================================================================
    print("\n" + "=" * 80)
    print("Step 1: Loading Dataset")
    print("=" * 80)
    
    df = load_dataset()
    
    # ========================================================================
    # SPLIT DATA
    # ========================================================================
    print("\n" + "=" * 80)
    print("Step 2: Splitting Data")
    print("=" * 80)
    
    # Split for tabular mode
    X_tabular, y = split_features_labels(df, mode="tabular")
    X_tabular = remove_constant_features(X_tabular)
    
    X_train_tab, X_val_tab, X_test_tab, y_train, y_val, y_test = train_val_test_split(
        X_tabular, y, random_state=RANDOM_SEED
    )
    
    # Split for tabular+embeddings mode
    X_enhanced, y_enhanced = split_features_labels(df, mode="tabular_plus_embeddings")
    X_enhanced = remove_constant_features(X_enhanced)
    
    # Identify tabular and embedding columns from original dataframe (before splitting)
    # This ensures we get all columns correctly, even if some are removed during feature selection
    tabular_cols_enh = identify_tabular_columns(df)
    pca_cols_enh = identify_pca_columns(df)
    
    # Filter to only columns that exist in X_enhanced (after removing constant features)
    tabular_cols_enh = [c for c in tabular_cols_enh if c in X_enhanced.columns]
    pca_cols_enh = [c for c in pca_cols_enh if c in X_enhanced.columns]
    
    # Use same indices as tabular split to ensure consistent train/val/test splits
    train_idx = X_train_tab.index
    val_idx = X_val_tab.index
    test_idx = X_test_tab.index
    
    X_train_enh = X_enhanced.loc[train_idx]
    X_val_enh = X_enhanced.loc[val_idx]
    X_test_enh = X_enhanced.loc[test_idx]
    
    # Separate tabular and embedding features for enhanced mode (for dual-tower NN)
    # This approach matches run_nn_only.py - split first, then separate
    X_train_enh_tab = X_train_enh[tabular_cols_enh].copy()
    X_train_enh_emb = X_train_enh[pca_cols_enh].copy()
    X_val_enh_tab = X_val_enh[tabular_cols_enh].copy()
    X_val_enh_emb = X_val_enh[pca_cols_enh].copy()
    X_test_enh_tab = X_test_enh[tabular_cols_enh].copy()
    X_test_enh_emb = X_test_enh[pca_cols_enh].copy()
    
    print(f"\nTabular features: {len(tabular_cols_enh)}")
    print(f"PCA embedding features: {len(pca_cols_enh)}")
    
    # ========================================================================
    # FEATURE SELECTION (if enabled)
    # ========================================================================
    if enable_feature_selection:
        print("\n" + "=" * 80)
        print("Step 3: Feature Selection")
        print("=" * 80)
        
        print("\n  Tabular features:")
        feature_mask_tabular, selected_features_tabular, _ = select_features_with_xgboost(
            X_train_tab, y_train, random_state=RANDOM_SEED
        )
        X_train_tab = apply_feature_mask(X_train_tab, feature_mask_tabular)
        X_val_tab = apply_feature_mask(X_val_tab, feature_mask_tabular)
        X_test_tab = apply_feature_mask(X_test_tab, feature_mask_tabular)
        
        print("\n  Enhanced features (tabular + embeddings):")
        feature_mask_enhanced, selected_features_enhanced, _ = select_features_with_xgboost(
            X_train_enh, y_train, random_state=RANDOM_SEED
        )
        X_train_enh = apply_feature_mask(X_train_enh, feature_mask_enhanced)
        X_val_enh = apply_feature_mask(X_val_enh, feature_mask_enhanced)
        X_test_enh = apply_feature_mask(X_test_enh, feature_mask_enhanced)
        
        # Update separated features after feature selection
        # Filter to only columns that still exist after feature selection
        tabular_cols_selected = [c for c in tabular_cols_enh if c in X_train_enh.columns]
        pca_cols_selected = [c for c in pca_cols_enh if c in X_train_enh.columns]
        
        # Re-separate the features using the filtered column lists
        X_train_enh_tab = X_train_enh[tabular_cols_selected].copy()
        X_train_enh_emb = X_train_enh[pca_cols_selected].copy()
        X_val_enh_tab = X_val_enh[tabular_cols_selected].copy()
        X_val_enh_emb = X_val_enh[pca_cols_selected].copy()
        X_test_enh_tab = X_test_enh[tabular_cols_selected].copy()
        X_test_enh_emb = X_test_enh[pca_cols_selected].copy()
        
        # Update the column lists for later use
        tabular_cols_enh = tabular_cols_selected
        pca_cols_enh = pca_cols_selected
    
    # ========================================================================
    # RUN EXPERIMENTS
    # ========================================================================
    print("\n" + "=" * 80)
    print("Step 4: Running Experiments")
    print("=" * 80)
    
    all_results = []
    
    # 1. SVM (tabular only)
    if 'svm' not in skip_models:
        print("\n" + "-" * 80)
        print("MODEL: SVM (Tabular Only)")
        print("-" * 80)
        
        result_svm = train_evaluate_svm(
            X_train_tab, y_train, X_val_tab, y_val, X_test_tab, y_test,
            random_state=RANDOM_SEED
        )
        
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
            all_results.append({
                'model_name': 'SVM',
                'input_mode': 'tabular',
                'imbalance_strategy': 'class_weight',
                'metric_name': metric,
                'mean': result_svm[f'{metric}_mean'],
                'std': result_svm[f'{metric}_std'],
            })
    else:
        print("\n  Skipping SVM model")
    
    # 2. Random Forest
    if 'rf' not in skip_models:
        print("\n" + "-" * 80)
        print("MODEL: Random Forest")
        print("-" * 80)
        
        # Tabular mode
        print("\n  Input mode: tabular")
        result_rf_tab = train_evaluate_rf(
            X_train_tab, y_train, X_val_tab, y_val, X_test_tab, y_test,
            random_state=RANDOM_SEED
        )
        
        for strategy, metrics in result_rf_tab.items():
            for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
                all_results.append({
                    'model_name': 'RandomForest',
                    'input_mode': 'tabular',
                    'imbalance_strategy': strategy,
                    'metric_name': metric,
                    'mean': metrics[f'{metric}_mean'],
                    'std': metrics[f'{metric}_std'],
                })
        
        # Enhanced mode
        print("\n  Input mode: tabular_plus_embeddings")
        result_rf_enh = train_evaluate_rf(
            X_train_enh, y_train, X_val_enh, y_val, X_test_enh, y_test,
            random_state=RANDOM_SEED
        )
        
        for strategy, metrics in result_rf_enh.items():
            for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
                all_results.append({
                    'model_name': 'RandomForest',
                    'input_mode': 'tabular_plus_embeddings',
                    'imbalance_strategy': strategy,
                    'metric_name': metric,
                    'mean': metrics[f'{metric}_mean'],
                    'std': metrics[f'{metric}_std'],
                })
    else:
        print("\n  Skipping Random Forest model")
    
    # 3. XGBoost
    if 'xgb' not in skip_models:
        print("\n" + "-" * 80)
        print("MODEL: XGBoost")
        print("-" * 80)
        
        # Tabular mode
        print("\n  Input mode: tabular")
        result_xgb_tab = train_evaluate_xgb(
            X_train_tab, y_train, X_val_tab, y_val, X_test_tab, y_test,
            random_state=RANDOM_SEED,
            use_feature_selection=False
        )
        
        for strategy, metrics in result_xgb_tab.items():
            for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
                all_results.append({
                    'model_name': 'XGBoost',
                    'input_mode': 'tabular',
                    'imbalance_strategy': strategy,
                    'metric_name': metric,
                    'mean': metrics[f'{metric}_mean'],
                    'std': metrics[f'{metric}_std'],
                })
        
        # Enhanced mode
        print("\n  Input mode: tabular_plus_embeddings")
        result_xgb_enh = train_evaluate_xgb(
            X_train_enh, y_train, X_val_enh, y_val, X_test_enh, y_test,
            random_state=RANDOM_SEED,
            use_feature_selection=False
        )
        
        for strategy, metrics in result_xgb_enh.items():
            for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
                all_results.append({
                    'model_name': 'XGBoost',
                    'input_mode': 'tabular_plus_embeddings',
                    'imbalance_strategy': strategy,
                    'metric_name': metric,
                    'mean': metrics[f'{metric}_mean'],
                    'std': metrics[f'{metric}_std'],
                })
    else:
        print("\n  Skipping XGBoost model")
    
    # 4. Dual-tower Neural Network (tabular+embeddings only)
    if 'nn' not in skip_models:
        print("\n" + "-" * 80)
        print("MODEL: Dual-Tower Neural Network")
        print("-" * 80)
        
        result_nn = train_evaluate_dual_tower_nn(
            X_train_enh_tab, X_train_enh_emb, y_train,
            X_val_enh_tab, X_val_enh_emb, y_val,
            X_test_enh_tab, X_test_enh_emb, y_test,
            random_state=RANDOM_SEED
        )
        
        for ratio_str, metrics in result_nn.items():
            for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
                all_results.append({
                    'model_name': 'DualTowerNN',
                    'input_mode': 'tabular_plus_embeddings',
                    'imbalance_strategy': ratio_str,
                    'metric_name': metric,
                    'mean': metrics[f'{metric}_mean'],
                    'std': metrics[f'{metric}_std'],
                })
    else:
        print("\n  Skipping Dual-Tower NN model")
    
    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    print("\n" + "=" * 80)
    print("Step 5: Saving Results")
    print("=" * 80)
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(RESULTS_CSV, index=False)
    print(f"  Saved results to: {RESULTS_CSV}")
    
    # Create summary Markdown
    with open(RESULTS_MD, 'w') as f:
        f.write("# Model Comparison Results\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Feature Selection:** {enable_feature_selection}\n\n")
        f.write("## Summary\n\n")
        
        # Best model per metric
        for metric in ['roc_auc', 'f1', 'accuracy']:
            metric_df = results_df[results_df['metric_name'] == metric].copy()
            if len(metric_df) > 0:
                best_idx = metric_df['mean'].idxmax()
                best = metric_df.loc[best_idx]
                f.write(f"**Best {metric.upper()}:** {best['model_name']} ({best['input_mode']}, {best['imbalance_strategy']}) = {best['mean']:.4f} ± {best['std']:.4f}\n\n")
        
        # Detailed table
        f.write("## Detailed Results\n\n")
        f.write("| Model | Input Mode | Imbalance Strategy | Metric | Mean | Std |\n")
        f.write("|-------|------------|-------------------|--------|------|-----|\n")
        
        for _, row in results_df.iterrows():
            f.write(f"| {row['model_name']} | {row['input_mode']} | {row['imbalance_strategy']} | "
                   f"{row['metric_name']} | {row['mean']:.4f} | {row['std']:.4f} |\n")
    
    print(f"  Saved summary to: {RESULTS_MD}")
    
    # ========================================================================
    # SAVE BEST MODEL
    # ========================================================================
    print("\n" + "=" * 80)
    print("Step 6: Saving Best Model")
    print("=" * 80)
    
    # Find best model by ROC-AUC
    roc_auc_df = results_df[results_df['metric_name'] == 'roc_auc'].copy()
    best_idx = roc_auc_df['mean'].idxmax()
    best_config = roc_auc_df.loc[best_idx]
    
    print(f"\n  Best model configuration:")
    print(f"    Model: {best_config['model_name']}")
    print(f"    Input mode: {best_config['input_mode']}")
    print(f"    Imbalance strategy: {best_config['imbalance_strategy']}")
    print(f"    ROC-AUC: {best_config['mean']:.4f} ± {best_config['std']:.4f}")
    
    # Retrain and save the best model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"\n  Retraining best model for saving...")
    import joblib
    import json
    
    # Select appropriate data
    if best_config['input_mode'] == 'tabular_plus_embeddings':
        X_train_best = X_train_enh.copy()
        X_test_best = X_test_enh.copy()
    else:
        X_train_best = X_train_tab.copy()
        X_test_best = X_test_tab.copy()
    
    # Apply imbalance strategy
    from scripts.imbalance import apply_imbalance_strategy
    X_train_processed, y_train_processed = apply_imbalance_strategy(
        X_train_best, y_train, best_config['imbalance_strategy'],
        random_state=RANDOM_SEED,
        target_ratio=None
    )
    
    # Train and save model based on type
    if best_config['model_name'] == 'XGBoost':
        from scripts.models.xgb_model import convert_to_numeric
        import xgboost as xgb
        
        X_train_clean = convert_to_numeric(X_train_processed)
        X_test_clean = convert_to_numeric(X_test_best)
        
        final_model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            use_label_encoder=False,
            random_state=RANDOM_SEED
        )
        final_model.fit(X_train_clean, y_train_processed)
        feature_names = list(X_train_clean.columns)
        
    elif best_config['model_name'] == 'RandomForest':
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        
        # Convert to numeric
        X_train_clean = X_train_processed.select_dtypes(include=[np.number]).fillna(0)
        X_test_clean = X_test_best.select_dtypes(include=[np.number]).fillna(0)
        
        # Align columns
        common_cols = X_train_clean.columns.intersection(X_test_clean.columns)
        X_train_clean = X_train_clean[common_cols]
        X_test_clean = X_test_clean[common_cols]
        
        final_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=RANDOM_SEED,
            class_weight='balanced' if best_config['imbalance_strategy'] == 'class_weight' else None
        )
        final_model.fit(X_train_clean, y_train_processed)
        feature_names = list(X_train_clean.columns)
        
    elif best_config['model_name'] == 'SVM':
        from sklearn.svm import SVC
        from sklearn.preprocessing import StandardScaler
        
        # Convert to numeric
        X_train_clean = X_train_processed.select_dtypes(include=[np.number]).fillna(0)
        X_test_clean = X_test_best.select_dtypes(include=[np.number]).fillna(0)
        
        # Align columns
        common_cols = X_train_clean.columns.intersection(X_test_clean.columns)
        X_train_clean = X_train_clean[common_cols]
        X_test_clean = X_test_clean[common_cols]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_clean)
        
        final_model = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,
            random_state=RANDOM_SEED,
            class_weight='balanced'
        )
        final_model.fit(X_train_scaled, y_train_processed)
        feature_names = list(X_train_clean.columns)
        
        # Save scaler too
        scaler_path = MODELS_DIR / "best_model_scaler.pkl"
        joblib.dump(scaler, scaler_path)
        print(f"  Saved scaler to: {scaler_path}")
        
    else:
        print(f"  Warning: Model saving not yet implemented for {best_config['model_name']}")
        print(f"  Only XGBoost, RandomForest, and SVM models can be saved currently.")
        return
    
    # Save model
    model_path = MODELS_DIR / "best_model.pkl"
    joblib.dump(final_model, model_path)
    
    # Save metadata
    metadata = {
        'model_name': best_config['model_name'],
        'input_mode': best_config['input_mode'],
        'imbalance_strategy': best_config['imbalance_strategy'],
        'roc_auc': float(best_config['mean']),
        'roc_auc_std': float(best_config['std']),
        'feature_names': feature_names,
        'random_state': RANDOM_SEED,
    }
    metadata_path = MODELS_DIR / "best_model_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  Saved best model to: {model_path}")
    print(f"  Saved metadata to: {metadata_path}")
    
    print("\n" + "=" * 80)
    print("EXPERIMENTS COMPLETED")
    print("=" * 80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total configurations: {len(results_df)}")
    print(f"\nResults saved to: {RESULTS_DIR}")
    print(f"Best model saved to: {MODELS_DIR}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run ML experiments for clinical trial feasibility prediction')
    parser.add_argument(
        '--no-feature-selection',
        action='store_true',
        help='Disable XGBoost-based feature selection'
    )
    parser.add_argument(
        '--skip-models',
        nargs='+',
        choices=['svm', 'rf', 'xgb', 'nn'],
        help='Skip specific models (e.g., --skip-models svm nn)'
    )
    
    args = parser.parse_args()
    
    enable_feature_selection = not args.no_feature_selection
    skip_models = args.skip_models or []
    
    run_experiments(
        enable_feature_selection=enable_feature_selection,
        skip_models=skip_models
    )


if __name__ == "__main__":
    main()

