"""
Central configuration for ML pipeline.

Defines paths, column names, hyperparameters, and experiment settings.
"""

from pathlib import Path

# ============================================================================
# PROJECT STRUCTURE
# ============================================================================

# Project root (two levels up from scripts/)
PROJECT_ROOT = Path(__file__).parent.parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = PROJECT_ROOT.parent / "data_raw" / "ctgov_api" / "trials"
DATA_ENHANCED_DIR = PROJECT_ROOT / "data_enhanced"

# Create data directories if they don't exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
DATA_ENHANCED_DIR.mkdir(parents=True, exist_ok=True)

# Main dataset file (will be created by pipeline)
DATASET_PATH = DATA_ENHANCED_DIR / "oncology_phase23_enhanced_hist.parquet"

# Output paths
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = RESULTS_DIR / "models"
RESULTS_CSV = RESULTS_DIR / "model_comparison.csv"
RESULTS_MD = RESULTS_DIR / "model_comparison.md"

# ============================================================================
# DATA PARSING FILTERS
# ============================================================================

# Phase filter (used for reference - actual filtering done in parsing script)
PHASES = ["PHASE2", "PHASE3", "PHASE4", "PHASE2/PHASE3"]

# Study type filter
STUDY_TYPE = "INTERVENTIONAL"

# Sponsor type filter (company-sponsored)
SPONSOR_TYPES = ["INDUSTRY"]

# Condition filters - REMOVED (now including all cancer types)
# Previously filtered for NSCLC only, but now expanding to all cancers
# NSCLC_KEYWORDS kept for reference but no longer used in parsing

# ============================================================================
# COLUMN NAMES
# ============================================================================

# Label column
LABEL_COL = "label_feasible"

# Identifier column
ID_COL = "nct_id"

# PCA embedding column prefix
PCA_EMB_PREFIX = "pca_emb_"

# Text columns (to exclude from tabular features)
TEXT_COLUMNS = [
    "brief_title",
    "official_title",
    "brief_summary",
    "detailed_description",
    "eligibility_criteria_text",
    "condition_text",
    "intervention_text",
    "description_text",
]

# JSON columns (contain JSON-encoded lists/dicts - exclude from tabular features)
JSON_COLUMNS = [
    "condition_list",
    "condition_mesh_terms",
    "intervention_mesh_terms",
    "interventions",
    "intervention_names",
    "intervention_types",
    "arm_groups",
    "phases_list",
    "primary_outcomes",
    "primary_outcome_measures",
    "secondary_outcomes",
    "secondary_outcome_measures",
    "other_outcome_measures",
    "collaborators",
    "references",
    "overall_officials",
    "central_contacts",
    "secondary_ids",
    "keywords",
    "facility_countries",
    "facility_cities",
    "facility_states",
    "std_ages",
    "see_also_links",
    "avail_ipds",
    "ipd_sharing_info_types",
]

# Data leakage columns (to exclude - contain post-hoc information)
LEAKAGE_COLUMNS = [
    "is_historical",
    "feasibility_label",
    "overall_status",
    "last_known_status",
    "why_stopped",  # Explains why trial was terminated (post-hoc)
    "has_results",  # Indicates if results are available (post-hoc)
    "completion_date",
    "primary_completion_date",
    "last_update_posted_date",
    "last_update_submit_date",
    "first_posted_date",
    "results_first_posted_date",
    "results_first_submit_date",
    "results_first_submit_qc_date",
    "study_first_submit_date",
    "study_first_submit_qc_date",
    "status_verified_date",
    "enrollment_actual",
    "enrollment_actual_type",
    "enrollment_actual_num",
]

# Manually excluded columns (excluded before feature selection)
# Excluded based on: high missingness, redundancy, or lack of predictive value
MANUALLY_EXCLUDED_COLUMNS = [
    "acronym",  # High missingness (75%), not predictive
    "biospec_description",  # 100% missing
    "biospec_retention",  # 100% missing
    "gender_based",  # 97% missing, redundant with gender/sex
    "gender_description",  # 99% missing, redundant with gender/sex
    "intervention_model_description",  # 92% missing, redundant with intervention_model
    "ipd_sharing_access_criteria",  # 90% missing, not predictive
    "ipd_sharing_description",  # 78% missing, not predictive
    "ipd_sharing_time_frame",  # 91% missing, not predictive
    "ipd_sharing_url",  # 85% missing, not predictive
    "is_ppsd",  # 100% missing
    "is_unapproved_device",  # 100% missing (only 33 non-missing)
    "is_us_export",  # 92% missing, not predictive
    "lead_sponsor_type",  # Constant (all "Industry"), redundant
    "observational_model",  # 100% missing
    "patient_registry",  # 100% missing
    "responsible_party_investigator_affiliation",  # 100% missing
    "responsible_party_investigator_full_name",  # 100% missing
    "responsible_party_investigator_title",  # 100% missing
    "responsible_party_name",  # 94% missing, redundant with lead_sponsor_name
    "responsible_party_organization",  # 94% missing, redundant with lead_sponsor_name
    "sampling_method",  # 100% missing
    "sex",  # Redundant with gender (both contain same information)
    "study_population",  # 100% missing
    "study_type",  # Constant (all "INTERVENTIONAL"), redundant
    "target_duration",  # 100% missing
    "time_perspective",  # 100% missing
    "who_masked",  # 75% missing, redundant with masking
    # Note: lead_sponsor_name is KEPT (available in 100% of cases, useful for prediction)
]

# ============================================================================
# RANDOM SEED
# ============================================================================

RANDOM_SEED = 42

# ============================================================================
# DATA SPLITTING
# ============================================================================

TEST_SIZE = 0.2
VAL_SIZE = 0.15  # Fraction of training data for validation
TRAIN_SIZE = 1.0 - TEST_SIZE

# ============================================================================
# PCA SETTINGS
# ============================================================================

PCA_VARIANCE_THRESHOLD = 0.95  # Retain 95% of variance

# ============================================================================
# FEATURE SELECTION
# ============================================================================

# Enable/disable feature selection
ENABLE_FEATURE_SELECTION = True

# XGBoost parameters for feature selection
FEATURE_SELECTION_XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "n_estimators": 100,
    "max_depth": 5,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_SEED,
    "use_label_encoder": False,
}

# Minimum number of features to retain (safety check)
MIN_FEATURES_AFTER_SELECTION = 10

# ============================================================================
# CLASS IMBALANCE HANDLING
# ============================================================================

# Downsampling ratios (minority:majority)
DOWNSAMPLE_RATIOS = [1.0, 1.5, 2.0]

# Oversampling ratios for neural networks (minority:majority)
# Per thesis: 0.6, 0.8, 1.0 (note: 0.6 may require capping if minority already large)
OVERSAMPLE_RATIOS = [0.6, 0.8, 1.0]

# ============================================================================
# MODEL HYPERPARAMETERS
# ============================================================================

# SVM (baseline, tabular only)
SVM_PARAMS = {
    "kernel": "linear",
    "probability": True,
    "class_weight": "balanced",
    "random_state": RANDOM_SEED,
}

# Random Forest grid search parameters
RF_GRID_PARAMS = {
    "n_estimators": [100, 200, 300],
    "max_depth": [10, 20, None],
    "max_features": ["sqrt", "log2"],
    "class_weight": ["balanced"],
    "random_state": [RANDOM_SEED],
}

# XGBoost grid search parameters
XGB_GRID_PARAMS = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.8, 0.9],
    "colsample_bytree": [0.8, 0.9],
    "random_state": [RANDOM_SEED],
}

# Dual-tower NN parameters
NN_PARAMS = {
    "tabular_tower": [32, 64],
    "embedding_tower": [64, 64],
    "combined_units": 128,
    "classifier_units": 64,
    "dropout_rate": 0.3,
    "learning_rate": 0.001,
    "weight_decay": 1e-4,
    "batch_size": 32,
    "epochs": 100,
    "early_stopping_patience": 5,
}

# ============================================================================
# EXPERIMENT SETTINGS
# ============================================================================

# Number of repeated runs for each configuration
N_ITERATIONS = 10

# Cross-validation folds for hyperparameter tuning
CV_FOLDS = 5

# Metrics to compute
METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]

# ============================================================================
# IMBALANCE STRATEGIES
# ============================================================================

IMBALANCE_STRATEGY_CLASS_WEIGHT = "class_weight"
IMBALANCE_STRATEGY_DOWNSAMPLE_BALANCED = "downsample_balanced"
IMBALANCE_STRATEGY_DOWNSAMPLE_1_5 = "downsample_1_5"  # 1:1.5 ratio
IMBALANCE_STRATEGY_PARTIAL = "partial_downsample"  # 1:2 ratio

ALL_IMBALANCE_STRATEGIES = [
    IMBALANCE_STRATEGY_CLASS_WEIGHT,
    IMBALANCE_STRATEGY_DOWNSAMPLE_BALANCED,  # 1:1
    IMBALANCE_STRATEGY_DOWNSAMPLE_1_5,  # 1:1.5
    IMBALANCE_STRATEGY_PARTIAL,  # 1:2
]

# ============================================================================
# BIOLINKBERT / EMBEDDING SETTINGS
# ============================================================================

BIOLINKBERT_MODEL_NAME = "michiyasunaga/BioLinkBERT-base"
EMBEDDING_BATCH_SIZE = 32
EMBEDDING_DEVICE = "cuda"  # Will fallback to CPU if CUDA unavailable
