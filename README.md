# Data Processing Pipeline - Detailed Documentation

This folder contains the **executable pipeline steps** that process raw ClinicalTrials.gov data and prepare it for machine learning model training.

**These scripts are meant to be RUN directly** (unlike utility modules in the parent `scripts/` directory which are imported).

---

## Pipeline Overview

The complete pipeline consists of 7 sequential steps, each building upon the previous step's output:

```
Raw CTGov JSON → Step 1 → Step 2 → Step 3 → Step 4 → Step 5 → Step 6 → Final Dataset → Step 7 (Model Training)
```

---

## Step-by-Step Details

### Step 1: Parse Raw Data (`01_parse_raw_data.py`)

**Purpose**: Extract structured variables from raw ClinicalTrials.gov JSON files.

**Input**: 
- Raw JSONL files from CTGov v2 API download:
  - `oncology_phase234_interventional_industry_full_raw.jsonl` (preferred - full records)
  - Falls back to `oncology_phase234_interventional_industry_search_raw.jsonl`

**Filtering Criteria**:
- **Phase**: Phase 2, 3, or 4 (including Phase 2/3 combos)
- **Study Type**: Interventional only
- **Sponsor Type**: Industry-sponsored (company-sponsored) only
- **Disease Domain**: All oncology/cancer indications (no specific cancer type filter)

**Process**:
1. Read JSONL files line by line
2. Parse each trial record using CTGov v2 API structure
3. Apply filters (phase, study type, sponsor type)
4. Extract variables from all available protocol modules:
   - Identification, Status, Design, Arms/Interventions
   - Conditions, Outcomes, Eligibility
   - Sponsor/Collaborators, Contacts/Locations
   - Description, Oversight, IPD Sharing
5. Derive binary label: `label_feasible`
   - 1 = COMPLETED
   - 0 = TERMINATED, WITHDRAWN, SUSPENDED, or other non-completed statuses

**Output**: 
- **File**: `data/01_parsed_trials.parquet`
- **Shape**: ~10,408 trials × 108 columns
- **Format**: Parquet (columnar, compressed)

**Key Variables Extracted** (108 total):
- Identifiers: `nct_id`, `brief_title`, `official_title`, `acronym`
- Text fields: `brief_summary`, `detailed_description`, `eligibility_criteria_text`, `condition_text`, `intervention_text`
- Design: `phase`, `study_type`, `allocation`, `intervention_model`, `masking`, `number_of_arms`
- Eligibility: `min_age`, `max_age`, `gender`, `healthy_volunteers`, `study_population`
- Enrollment: `enrollment_planned`, `enrollment_planned_type`
- Sponsor/Location: `lead_sponsor_name`, `lead_sponsor_type`, `number_of_facilities`, location flags
- Regulatory: `has_dmc`, `is_fda_regulated_drug`, `is_fda_regulated_device`
- Dates: `start_date` (excludes post-hoc dates to prevent leakage)
- Label: `label_feasible` (derived from `overall_status`)

**Execution Time**: ~2-5 minutes (depends on file size)

---

### Step 2: Extract MeSH/Disease and Intervention Features (`02_extract_mesh_features.py`)

**Purpose**: Create disease-specific and intervention type flags from text fields using keyword matching.

**Input**: `data/01_parsed_trials.parquet`

**Process**:

#### Disease Flags Extraction
- Extracts text from `condition_text` column
- Uses case-insensitive keyword matching
- Creates binary flags for:
  - **NSCLC-specific**: `is_nsclc`, `is_sclc`, `is_lung_cancer`
  - **Other cancers**: `is_breast_cancer`, `is_prostate_cancer`, `is_colorectal_cancer`, `is_gastric_cancer`, `is_pancreatic_cancer`, `is_hepatocellular`, `is_renal`, `is_bladder`, `is_ovarian`, `is_cervical`, `is_endometrial`, `is_esophageal`, `is_head_neck`, `is_brain`, `is_thyroid`, `is_sarcoma`, `is_lymphoma`, `is_myeloma`, `is_aml`, `is_all`, `is_cll`, `is_cml`
  - **Catch-alls**: `is_other_cancer` (contains "cancer" but doesn't match specific types), `is_other` (doesn't match any specific or general cancer category)

#### Intervention Flags Extraction
- Extracts text from `intervention_text` and `interventions` columns
- Creates binary flags for:
  - `is_immunotherapy` (PD-1/PD-L1 inhibitors, checkpoint inhibitors)
  - `is_tki` (tyrosine kinase inhibitors)
  - `is_adc` (antibody-drug conjugates)
  - `is_chemo` (chemotherapy)
  - `is_hormone` (hormone therapy)
  - `is_cell_therapy` (CAR-T, cell therapy)
  - `is_targeted` (targeted therapy)
  - `is_radiotherapy` (radiation therapy)
  - `is_surgery` (surgical interventions)

**Output**: 
- **File**: `data_enhanced/02_trials_with_mesh.parquet`
- **New Columns**: ~35 binary flags (~25 disease + ~10 intervention)
- **Shape**: ~10,408 trials × ~143 columns (108 original + ~35 new)

**Execution Time**: ~1-2 minutes

---

### Step 3: Extract Eligibility Complexity Features (`03_extract_eligibility_complexity.py`)

**Purpose**: Quantify the complexity and stringency of eligibility criteria through text analysis.

**Input**: `data_enhanced/02_trials_with_mesh.parquet`

**Process**:

Extracts multiple metrics from `eligibility_criteria_text`:

**Text Length Metrics**:
- `eligibility_text_len`: Total character count

**Structural Metrics**:
- `num_inclusion_criteria`: Count of "inclusion" mentions
- `num_exclusion_criteria`: Count of "exclusion" mentions

**Section-Specific Metrics**:
- `inclusion_text_len`: Character count of inclusion section
- `exclusion_text_len`: Character count of exclusion section
- `num_inclusion_items`: Bullet points in inclusion section
- `num_exclusion_items`: Bullet points in exclusion section


**Rationale**: More complex eligibility criteria may indicate:
- More selective patient populations
- More sophisticated trial design
- Higher safety standards
- Better planning and execution

**Output**: 
- **File**: `data_enhanced/03_trials_with_eligibility.parquet`
- **New Columns**: 6 complexity metrics
- **Shape**: ~10,408 trials × ~163 columns

**Execution Time**: ~1-2 minutes

---

### Step 4: Generate Text Embeddings (`04_generate_embeddings.py`)

**Purpose**: Generate semantic embeddings from textual trial descriptions using BioLinkBERT.

**Input**: `data_enhanced/03_trials_with_eligibility.parquet`

**Process**:

1. **Text Combination**:
   - Combines 5 text fields into single string:
     - `brief_title` + `official_title` + `brief_summary` + `detailed_description` + `eligibility_criteria_text`
   - Handles missing text by replacing with empty strings

2. **Embedding Generation**:
   - Model: **BioLinkBERT-base** (`michiyasunaga/BioLinkBERT-base`)
   - Architecture: BERT-base (12 layers, 768 hidden dimensions, 12 attention heads)
   - Pre-training: PubMed abstracts, clinical trial descriptions, biomedical literature
   - Tokenization: Maximum 512 tokens (BERT standard), padding/truncation as needed
   - Embedding extraction: `[CLS]` token (768 dimensions per trial)

3. **Batch Processing**:
   - Processes trials in batches (configurable, default: 32)
   - GPU acceleration recommended (optional but ~10x faster)

**Why BioLinkBERT?**
- Domain-specific pre-training captures biomedical terminology
- Trained on clinical trial descriptions (better than general-purpose models)
- Captures nuanced relationships between medical concepts

**Output**: 
- **File**: `data_enhanced/04_trials_with_embeddings.parquet`
- **New Columns**: 768 embedding dimensions (`embedding_0` through `embedding_767`)
- **Shape**: ~10,408 trials × ~931 columns (163 previous + 768 embeddings)

**Execution Time**: 
- CPU: ~30-60 minutes (for 10,000+ trials)
- GPU: ~5-10 minutes (highly recommended)

**Note**: This is the slowest step. Consider using GPU if available.

---

### Step 5: Apply PCA Dimensionality Reduction (`05_apply_pca.py`)

**Purpose**: Reduce embedding dimensionality while preserving information (768 → ~295 dimensions).

**Input**: `data_enhanced/04_trials_with_embeddings.parquet`

**Process**:

1. **Preprocessing**:
   - Extract embedding columns (`embedding_0` through `embedding_767`)
   - Apply StandardScaler (mean=0, std=1) - PCA is scale-sensitive
   - Handle missing embeddings (should be rare)

2. **PCA Fitting** (ONLY on training data - prevents data leakage):
   - Loads previously split data or performs train/test split
   - Fits PCA on training embeddings only
   - Retains components explaining 95% of variance
   - Automatically determines number of components needed

3. **Transformation**:
   - Transforms train/val/test sets using fitted PCA (no refitting)
   - Creates new columns: `pca_emb_0` through `pca_emb_N` (typically ~295)

4. **Model Persistence**:
   - Saves fitted PCA model: `results/models/pca_model.pkl`
   - Saves fitted scaler: `results/models/pca_scaler.pkl`
   - Enables consistent transformation for future inference

**PCA Results**:
- **Original dimension**: 768
- **Reduced dimension**: ~295 components
- **Compression ratio**: ~2.6x reduction
- **Variance explained**: 95.0%
- **Top 10 components**: Explain substantial variance (concentrated information)

**Why PCA?**
- Reduces dimensionality to prevent overfitting
- Removes multicollinearity among embedding dimensions
- Speeds up model training and inference
- Filters noise while preserving semantic information

**Output**: 
- **File**: `data_enhanced/05_trials_with_pca.parquet`
- **New Columns**: ~295 PCA embedding columns (replaces original 768)
- **Shape**: ~10,408 trials × ~1,228 columns (163 base + 768 original embeddings + 295 PCA embeddings)
- **Saved Models**: `results/models/pca_model.pkl`, `results/models/pca_scaler.pkl`

**Execution Time**: ~1-2 minutes

---

### Step 6: Combine All Features (`06_combine_features.py`)

**Purpose**: Create final dataset ready for model training by combining all features and cleaning data.

**Input**: `data_enhanced/05_trials_with_pca.parquet`

**Process**:

1. **Drop Original Embeddings**:
   - Removes 768 original embedding columns (`embedding_0` through `embedding_767`)
   - Keeps only PCA-reduced embeddings (`pca_emb_*`)

2. **Label Verification**:
   - Ensures `label_feasible` column exists
   - Creates from `overall_status` if missing (shouldn't happen)

3. **Drop Missing Labels**:
   - Removes trials with missing or ambiguous labels
   - **Excludes all ongoing trials** (RECRUITING, ACTIVE_NOT_RECRUITING, ENROLLING_BY_INVITATION, etc.) because their final outcome is unknown
   - Final dataset contains only trials with **final status** (completed or terminated)

4. **Final Quality Checks**:
   - Verifies data integrity
   - Prints summary statistics

**Output**: 
- **File**: `data_enhanced/oncology_phase23_enhanced_hist.parquet` (final dataset)
- **Shape**: ~7,444 trials × 460 columns (after dropping trials with missing labels and excluding ongoing trials)
- **Columns**: 
  - ~112 tabular features (55 base + disease flags + intervention flags + complexity)
  - 295 PCA embeddings
  - Metadata (IDs, labels, text columns - excluded from model training)

**Summary Statistics**:
- Label distribution: Completed (75.8%), Not Completed (24.2%)
- Excluded: 2,964 ongoing/unknown status trials
- Feature breakdown: PCA embeddings (295), Disease flags (~11), Intervention flags (~10), Complexity features (~9), Other features (~133)

**Execution Time**: < 1 minute

---

### Step 7: Train and Evaluate Models (`07_train_models.py`)

**Purpose**: Train and evaluate all model configurations with comprehensive evaluation protocol.

**Input**: `data_enhanced/oncology_phase23_enhanced_hist.parquet`

**Process**: (See TRAINING_FLOW.md for detailed explanation)

1. **Load Dataset** (`scripts/data_loading.py`)
2. **Split Data** into train/val/test sets (68%/12%/20%)
3. **Feature Selection** (`scripts/feature_selection.py`) - XGBoost-based
4. **Train Models**:
   - SVM (tabular only)
   - Random Forest (tabular + enhanced, 4 strategies)
   - XGBoost (tabular + enhanced, 4 strategies)
   - Dual-tower NN (enhanced only, 3 oversampling ratios)
5. **Evaluate**: 10 iterations per configuration
6. **Save Results** to CSV and Markdown

**Output**: 
- **File**: `results/model_comparison.csv`, `results/model_comparison.md`

**Execution Time**: ~2-4 hours (depends on hardware and number of models)

**Note**: This script calls `run_experiments.py` which contains all the training logic.

---

## Usage

### Run All Steps

Use the master script:
```bash
# From Project 2 directory
python scripts/run_pipeline.py
```

Or run steps individually:
```bash
python scripts/pipeline/01_parse_raw_data.py
python scripts/pipeline/02_extract_mesh_features.py
python scripts/pipeline/03_extract_eligibility_complexity.py
python scripts/pipeline/04_generate_embeddings.py
python scripts/pipeline/05_apply_pca.py
python scripts/pipeline/06_combine_features.py
python scripts/pipeline/07_train_models.py
```

### Run Specific Steps

The master script supports flexible execution:
```bash
# Start from step 5
python scripts/run_pipeline.py --step 5

# Run steps 1-4 only
python scripts/run_pipeline.py --step 1 --end 4

# Skip step 4 (embeddings - slowest step)
python scripts/run_pipeline.py --skip-step 4

# See all options
python scripts/run_pipeline.py --help
```

### Resuming After Interruption

If a step fails, you can resume from the next step:
```bash
# If step 5 failed, resume from step 5
python scripts/run_pipeline.py --step 5
```

Each script checks if its required input exists before running.

---

## Performance Considerations

### Step 4 (Embedding Generation) - SLOWEST STEP

**Performance Tips**:
1. **Use GPU**: ~10x speedup (CUDA-enabled GPU required)
2. **Adjust batch size**: Larger batches = faster (if memory allows)
3. **Configuration**: Set `EMBEDDING_BATCH_SIZE` in `scripts/config.py`

**Estimated Time (10,000 trials)**:
- CPU (4 cores): ~45-60 minutes
- GPU (NVIDIA, CUDA): ~5-10 minutes

### Step 7 (Model Training) - SECOND SLOWEST

**Performance Factors**:
- Number of models (4 model families)
- Number of imbalance strategies (3-4 per model)
- Hyperparameter tuning (grid search with 5-fold CV)
- 10 iterations per configuration

**Estimated Time**:
- Full suite: ~2-4 hours (CPU) or ~1-2 hours (GPU for neural network)
- Skip specific models: Use `--skip-models` flag in `run_experiments.py`

---

## Output File Locations

All intermediate and final outputs are organized as follows:

```
Project 2/
├── data/                                    # Step 1 output
│   └── 01_parsed_trials.parquet
│
├── data_enhanced/                           # Steps 2-6 outputs
│   ├── 02_trials_with_mesh.parquet
│   ├── 03_trials_with_eligibility.parquet
│   ├── 04_trials_with_embeddings.parquet
│   ├── 05_trials_with_pca.parquet
│   └── oncology_phase23_enhanced_hist.parquet  # Final dataset
│
└── results/                                 # Step 7 outputs
    ├── model_comparison.csv
    ├── model_comparison.md
    └── models/                              # Saved models
        ├── pca_model.pkl
        └── pca_scaler.pkl
```

---

## Data Flow Summary

```
Raw JSONL Files (CTGov v2 API)
    ↓
Step 1: Parse → 10,408 trials × 108 columns
    ↓
Step 2: MeSH Features → +35 binary flags
    ↓
Step 3: Eligibility Complexity → +20 complexity metrics
    ↓
Step 4: Embeddings → +768 embedding dimensions
    ↓
Step 5: PCA → -768 original, +295 PCA embeddings
    ↓
Step 6: Combine → Final: 9,456 trials × 460 columns
    ↓
Step 7: Train Models → Results CSV/MD
```

---

## Troubleshooting

### Common Issues

1. **Missing input file**: Each step checks if its required input exists. Run previous steps first.

2. **Out of memory**: 
   - Reduce batch size in Step 4 (embedding generation)
   - Process in chunks if necessary

3. **GPU not detected**: 
   - Step 4 will fall back to CPU (slower but works)
   - Check CUDA installation if GPU expected

4. **Long execution time**: 
   - Step 4 (embeddings) and Step 7 (training) are the slowest
   - Use GPU for Step 4 if available
   - Consider skipping specific models in Step 7

### Verification

After each step, check:
- Output file exists and has expected shape
- No error messages in console
- Summary statistics printed look reasonable

---

## Dependencies

Each step requires:
- **Step 1**: Raw JSONL files from CTGov download
- **Steps 2-6**: Output from previous step
- **Step 7**: Final dataset from Step 6

All steps require Python packages:
- pandas, numpy
- scikit-learn (for PCA, preprocessing)
- transformers (for Step 4 - BioLinkBERT)
- torch (optional, for GPU acceleration in Step 4)

See `requirements.txt` for complete list.

---

**Last Updated**: 2025-12-12  
**Pipeline Version**: v2.0  
**Dataset**: Oncology Phase 2/3/4, Interventional, Industry-Sponsored
