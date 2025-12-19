# Clinical Trial Feasibility Prediction

Machine learning pipeline to predict clinical trial completion/non-completion (feasibility) using publicly available ClinicalTrials.gov data.

**Key Innovation**: Integration of BioLinkBERT embeddings with traditional tabular features to capture semantic information from trial descriptions and eligibility criteria.

---

## 📁 Repository Structure

```
.
├── docs/                          # Project documentation
│   ├── METHODS_AND_RESULTS.md    # Detailed methods and results
│   ├── PROJECT_OVERVIEW.md       # High-level project overview
│   └── Research_Report_Clinical_Trial_Feasibility.md
│
├── scripts/                       # Source code
│   ├── pipeline/                 # Data processing pipeline (steps 01-07)
│   ├── deployment/               # Scoring ongoing trials (NSCLC)
│   ├── reports/                  # Report generation utilities
│   ├── models/                   # Model implementations
│   ├── utils/                    # Utility scripts
│   ├── config.py                 # Configuration
│   ├── data_loading.py           # Data loading utilities
│   ├── feature_selection.py      # Feature selection
│   ├── imbalance.py             # Class imbalance handling
│   ├── run_pipeline.py          # Run full pipeline
│   ├── run_experiments.py       # Run model training experiments
│   └── shap_analysis.py         # SHAP interpretability analysis
│
├── results/                      # Generated outputs (mostly gitignored)
│   ├── model_comparison.csv      # Model performance comparison (tracked)
│   ├── models/                   # Trained models (tracked: metadata only)
│   └── nsclc/                    # NSCLC deployment results (tracked: summaries only)
│
├── data/                         # Raw data (gitignored)
├── data_enhanced/                # Processed data (gitignored)
├── deliverables/                 # Generated reports/charts (gitignored)
│
├── .gitignore
├── LICENSE
├── README.md                     # This file
└── requirements.txt              # Python dependencies
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- GPU recommended (for embedding generation)
- ~10GB disk space for data

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd Project-2
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Download/Prepare Data:**
   - Place raw ClinicalTrials.gov JSON files in `data/` directory
   - Or use pre-processed data in `data_enhanced/` if available

---

## 🔄 How to Reproduce Results

### Option 1: Full Pipeline (from raw data)

If you have raw ClinicalTrials.gov data:

```bash
# Step 1: Parse raw JSON data
python scripts/pipeline/01_parse_raw_data.py

# Step 2: Extract MeSH features
python scripts/pipeline/02_extract_mesh_features.py

# Step 3: Extract eligibility complexity features
python scripts/pipeline/03_extract_eligibility_complexity.py

# Step 4: Generate BioLinkBERT embeddings (SLOW - requires GPU)
python scripts/pipeline/04_generate_embeddings.py

# Step 5: Apply PCA dimensionality reduction
python scripts/pipeline/05_apply_pca.py

# Step 6: Combine all features
python scripts/pipeline/06_combine_features.py

# Step 7: Train models
python scripts/pipeline/07_train_models.py
```

**Or run all steps at once:**
```bash
python scripts/run_pipeline.py
```

### Option 2: Model Training Only (if data already processed)

If you have the processed dataset (`data_enhanced/oncology_phase23_enhanced_hist.parquet`):

```bash
# Run model training experiments
python scripts/run_experiments.py
```

This will:
- Load the processed dataset
- Apply feature selection
- Train multiple models (XGBoost, Random Forest, SVM, Dual NN)
- Evaluate with cross-validation
- Save best model to `results/models/`

### Option 3: SHAP Analysis (if model already trained)

```bash
# Generate SHAP interpretability analysis
python scripts/shap_analysis.py
```

This generates:
- SHAP importance plots
- Feature importance rankings
- Dependence plots
- Excel exports with detailed SHAP values

### Option 4: Deploy to Ongoing NSCLC Trials

```bash
# Score ongoing NSCLC Phase II-III trials
python scripts/deployment/score_ongoing_nsclc.py
```

This will:
- Load the trained model
- Filter ongoing NSCLC trials
- Generate completion probability predictions
- Create risk buckets (LOW/MEDIUM/HIGH)
- Generate visualizations and output tables

### Option 5: Generate Reports

```bash
# Generate PowerPoint presentation
python scripts/reports/create_presentation.py

# Generate Word document
python scripts/reports/create_word_document.py

# Prepare Power BI data
python scripts/reports/prepare_powerbi_data.py
```

---

## 📊 Key Results

### Model Performance

**Best Model**: XGBoost (tabular_plus_embeddings, class_weight)
- **ROC-AUC**: 0.6803 ± 0.0069 (cross-validated)
- **Features**: 219 total (14 non-PCA tabular + 205 PCA embeddings)

### Top Predictors (SHAP Analysis)

1. **Number of Facilities** (SHAP = 0.6165) - Strongest predictor
2. **Has US Sites** (SHAP = 0.3124)
3. **Has DMC** (SHAP = 0.1666)
4. **Eligibility Complexity** (text length metrics)

See `docs/METHODS_AND_RESULTS.md` for complete details.

### Deployment Results

- **287 ongoing NSCLC Phase II-III trials** scored
- **Risk buckets**: 86 LOW / 115 MEDIUM / 86 HIGH
- **Output**: `results/nsclc/nsclc_ongoing_scored.csv`

---

## 📝 Documentation

- **`docs/METHODS_AND_RESULTS.md`**: Complete methodology, results, and interpretation
- **`docs/PROJECT_OVERVIEW.md`**: High-level project overview
- **`docs/Research_Report_Clinical_Trial_Feasibility.md`**: Academic-style research report

---

## 🔧 Configuration

Key settings in `scripts/config.py`:

- `ENABLE_FEATURE_SELECTION`: Enable/disable XGBoost feature selection
- `RANDOM_STATE`: Random seed for reproducibility
- `N_PCA_COMPONENTS`: Number of PCA components (default: 295, 95% variance)

---

## 📦 Data Requirements

### Input Data Format

- **Raw data**: ClinicalTrials.gov JSON files (`.jsonl` format)
- **Processed data**: Parquet file with all features (`data_enhanced/oncology_phase23_enhanced_hist.parquet`)

### Data Filters Applied

- Phase 2, 3, or 4 trials
- Interventional studies only
- Industry-sponsored only
- Oncology/cancer indications

### Label Definition

- **Label = 1**: COMPLETED trials (final status)
- **Label = 0**: TERMINATED/WITHDRAWN/SUSPENDED trials (final status)
- **Label = NaN**: Ongoing trials (excluded from training)

**Critical**: Only trials with final status are used for training. Ongoing trials are excluded because their outcome is unknown.

---

## 🧪 Model Training Details

### Models Tested

1. **XGBoost** (selected - best performance)
2. **Random Forest**
3. **SVM** (baseline)
4. **Dual Tower Neural Network**

### Input Modes

- **Tabular-only**: Structured features only
- **Tabular + Embeddings**: Structured + PCA-reduced BioLinkBERT embeddings

### Class Imbalance Handling

- **Class weight balancing** (for tree-based models)
- **Oversampling** (for neural networks)
- **Downsampling** (tested but performed poorly)

### Evaluation

- **10-fold cross-validation** with different random seeds
- **Metrics**: ROC-AUC, accuracy, precision, recall, F1
- **Results**: Mean ± standard deviation across iterations

---

## 🎯 Usage Examples

### Train a model and evaluate:

```python
from scripts.run_experiments import run_experiments

# Run full experiment suite
results = run_experiments(
    input_mode='tabular_plus_embeddings',
    enable_feature_selection=True
)
```

### Score new trials:

```python
from scripts.deployment.score_ongoing_nsclc import score_trials

# Score ongoing trials
predictions = score_trials(
    model_path='results/models/best_model.pkl',
    data_path='data/new_trials.parquet'
)
```

---

## ⚠️ Important Notes

1. **Data Privacy**: This project uses publicly available ClinicalTrials.gov data. No patient-level data is included.

2. **Reproducibility**: 
   - Set `RANDOM_STATE=42` in `scripts/config.py` for reproducibility
   - Results may vary slightly due to random sampling in cross-validation

3. **Computational Requirements**:
   - Embedding generation (Step 4) requires GPU and takes several hours
   - Model training takes ~30-60 minutes on CPU
   - Full pipeline: ~4-6 hours with GPU

4. **Model Limitations**:
   - ROC-AUC of 0.68 indicates moderate discriminative ability
   - Predictions are probabilistic, not guarantees
   - Suitable for risk stratification, not perfect prediction

---

## 📄 License

MIT License - see `LICENSE` file for details.
```

---

## 📧 Contact

For questions or issues, please open a GitHub issue.

---

## 🔗 Related Resources

- **Data Source**: [ClinicalTrials.gov](https://clinicaltrials.gov/)
- **BioLinkBERT**: [Hugging Face Model](https://huggingface.co/michiyasunaga/BioLinkBERT-base)
- **SHAP**: [SHAP Documentation](https://shap.readthedocs.io/)
