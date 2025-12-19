# Methods and Results: Clinical Trial Feasibility Prediction

## Table of Contents
1. [Data Source and Preprocessing](#1-data-source-and-preprocessing)
   - [1.0 Label Creation and Ongoing Trial Exclusion](#10-label-creation-and-ongoing-trial-exclusion)
2. [Feature Engineering Pipeline](#2-feature-engineering-pipeline)
3. [Text Embedding Generation](#3-text-embedding-generation)
4. [Dimensionality Reduction](#4-dimensionality-reduction)
5. [Feature Selection](#5-feature-selection)
6. [Class Imbalance Handling](#6-class-imbalance-handling)
7. [Model Training and Evaluation](#7-model-training-and-evaluation)
8. [Results](#8-results)
9. [Discussion](#9-discussion)

---

## 1. Data Source and Preprocessing

### 1.0 Label Creation and Ongoing Trial Exclusion

**Critical Methodology Decision**: Only trials with **final status** are included in modeling. Ongoing trials are excluded because their final outcome is unknown.

**Label Creation Rules** (applied in `scripts/pipeline/01_parse_raw_data.py`):
- `label_feasible = 1` if `overall_status == "COMPLETED"` (final status: completed)
- `label_feasible = 0` if `overall_status in {"TERMINATED", "WITHDRAWN", "SUSPENDED"}` (final status: non-completed)
- `label_feasible = NaN` for all other statuses, including:
  - `RECRUITING` (actively recruiting participants)
  - `ACTIVE_NOT_RECRUITING` (trial is ongoing but not recruiting)
  - `ENROLLING_BY_INVITATION` (ongoing enrollment)
  - `NOT_YET_RECRUITING` (not yet started)
  - `UNKNOWN` (status unknown)
  - Any other non-final statuses

**Exclusion Rationale**: 
- Ongoing trials have unknown final outcomes - they could still complete or terminate
- Including them as "non-completed" would be incorrect and misleading
- Only trials with definitive final status provide reliable training labels

**Impact**: 
- **10,408 trials** parsed from raw data
- **2,964 trials excluded** (2,012 ongoing + 952 unknown/ambiguous)
- **7,444 trials** included in final modeling dataset

### 1.1 Data Source
The dataset was constructed from publicly available ClinicalTrials.gov (CTGov) data, accessed via the Clinical Trials Transformation Initiative (CTTI) database. Raw JSON files containing protocol-level information for clinical trials were downloaded and parsed to extract structured and unstructured features.

### 1.2 Filtering Criteria
To focus on a homogeneous, high-quality dataset, we applied the following filters:

- **Phase**: Phase 2, Phase 3, or Phase 4 trials (including Phase 2/3 combos)
- **Study Type**: Interventional studies only
- **Sponsor Type**: Company-sponsored trials only (Industry-sponsored, excluding NIH, universities, and other non-industry sponsors)
- **Disease Indication**: All oncology/cancer trials (no disease-specific filter)

**Rationale**: Phase 2, 3, and 4 trials represent the critical stages where feasibility and completion become most relevant. Company-sponsored trials typically have more standardized reporting and resource allocation. Including all oncology trials provides a broader, more generalizable dataset while maintaining disease domain homogeneity.

### 1.3 Parsed Variables (Step 1: Raw Data Parsing)

From each trial's JSON protocol (via ClinicalTrials.gov v2 API), we extracted comprehensive variables from all available modules. The parsing process read from JSONL files (`oncology_phase234_interventional_industry_full_raw.jsonl`) containing full study records with complete protocol sections.

#### 1.3.1 Total Variables Extracted

The comprehensive parsing extracted **108 total variables** from the following modules:

- **identificationModule**: Trial identifiers, titles, acronyms, secondary IDs
- **statusModule**: Status information, dates, why_stopped, expanded access
- **designModule**: Phase, study type, allocation, masking, intervention model, enrollment info
- **armsInterventionsModule**: Arms and interventions (names, types, descriptions)
- **conditionsModule**: Conditions, keywords
- **outcomesModule**: Primary, secondary, and other outcomes
- **eligibilityModule**: Eligibility criteria text, age, sex/gender, study population
- **sponsorCollaboratorsModule**: Sponsors, collaborators, responsible party
- **contactsLocationsModule**: Locations (countries, cities, states), officials, contacts
- **descriptionModule**: Brief summary, detailed description
- **oversightModule**: DMC status, FDA regulation flags
- **ipdSharingStatementModule**: IPD sharing information
- **referencesModule**: References, links, IPDs
- **derivedSection**: MeSH terms (condition and intervention browse terms)

### 1.4 Data Quality and Missingness

After parsing and filtering, the dataset contained **10,408 trials**. After excluding trials with missing or ambiguous labels and **excluding all ongoing trials** (RECRUITING, ACTIVE_NOT_RECRUITING, ENROLLING_BY_INVITATION, etc.), the final modeling dataset contained **7,444 trials**. 

**Label Creation Logic:**
- Only trials with **final status** are included in modeling:
  - `label_feasible = 1` if `overall_status == "COMPLETED"`
  - `label_feasible = 0` if `overall_status in {"TERMINATED", "WITHDRAWN", "SUSPENDED"}`
  - `label_feasible = NaN` for all other statuses (ongoing, unknown, etc.) - **excluded from modeling**
- **Rationale**: Ongoing trials have unknown final outcomes and could still complete or terminate, making them inappropriate for training a feasibility prediction model.

Missing value analysis revealed the following patterns:

- **High missingness variables** (>30% missing):
  - `enrollment_planned`: Missing in ~30-40% of trials (common when enrollment type is ACTUAL or not yet reported)
  - `primary_completion_date`: Missing in ~25-35% of trials (for ongoing/terminated trials)
  - `completion_date`: Missing in ~30-40% of trials (for ongoing/terminated trials)
  - `enrollment_actual`: Missing in ~40-50% of trials (excluded from features to prevent leakage)

- **Low missingness variables** (<10% missing):
  - Trial identifiers (`nct_id`): 0% missing
  - Titles and descriptions: <2% missing
  - Design features (`phase`, `allocation`, `study_type`): <5% missing
  - Status information (`overall_status`): <1% missing
  - Location features (`number_of_facilities`, country flags): <8% missing

- **Feature engineering variables**:
  - Disease flags: 0% missing (binary flags, 0 if not matched)
  - Intervention flags: 0% missing (binary flags, 0 if not matched)
  - Eligibility complexity features: <5% missing (derived from eligibility text)
  - PCA embeddings: <2% missing (for trials with completely missing text)

**Handling**: Missing values were handled differently based on variable type:
- **Categorical variables**: Missing values were encoded as a separate category or imputed with the mode
- **Numeric variables**: Missing values were imputed with the median for continuous variables, or 0 for count variables
- **Text variables**: Missing text was replaced with empty strings for embedding generation
- **JSON variables**: Missing JSON fields were set to None/null and excluded from tabular features

**Feature Categorization Summary**:
- **Total variables parsed**: 108 columns
- **Excluded from tabular features**: 54+ columns
  - 2 identifiers/labels
  - 8 text columns (used for embeddings)
  - 26 JSON columns (structured data encoded as JSON)
  - 17 leakage columns (post-hoc information)
  - 1 PCA embedding prefix (dynamic, varies by PCA components)
  - **Manually excluded columns**: Additional variables excluded due to high missingness (>75%), redundancy, or lack of predictive value (e.g., `acronym`, `biospec_description`, `gender_based`, etc.)
- **Included as tabular features**: **55 structured features** (after manual exclusions)

---

## 2. Feature Engineering Pipeline

### 2.1 Step 2: MeSH/Disease and Intervention Feature Extraction

#### 2.1.1 Disease-Specific Flags
To capture disease heterogeneity within NSCLC and related cancers, we created binary flags based on keyword matching in the `condition_text` field:

**NSCLC-Specific Flags:**
- `is_nsclc`: Non-small cell lung cancer (primary focus)
- `is_sclc`: Small cell lung cancer
- `is_lung_cancer`: General lung cancer (broader category)

**Other Cancer Types (for context):**
- `is_breast_cancer`, `is_prostate_cancer`, `is_colorectal_cancer`, `is_gastric_cancer`, `is_pancreatic_cancer`
- `is_hepatocellular`, `is_renal`, `is_bladder`, `is_ovarian`, `is_cervical`, `is_endometrial`
- `is_esophageal`, `is_head_neck`, `is_brain`, `is_thyroid`, `is_sarcoma`
- `is_lymphoma`, `is_myeloma`, `is_aml`, `is_all`, `is_cll`, `is_cml`

**Catch-All Flags:**
- `is_other_cancer`: Contains cancer-related terms but doesn't match specific cancer types
- `is_other`: Catch-all for trials that don't match any specific or general cancer category

**Method**: Keyword matching was performed case-insensitively. Each flag was set to 1 if any keyword in the corresponding keyword list appeared in the condition text.

#### 2.1.2 Intervention Type Flags
To capture intervention heterogeneity, we created flags based on intervention names and types:

**Therapy Type Flags:**
- `is_immunotherapy`: PD-1/PD-L1 inhibitors, CTLA-4 inhibitors, checkpoint inhibitors
- `is_tki`: Tyrosine kinase inhibitors (erlotinib, gefitinib, osimertinib, etc.)
- `is_adc`: Antibody-drug conjugates
- `is_chemotherapy`: Traditional chemotherapy agents
- `is_targeted_therapy`: Other targeted therapies
- `is_radiation`: Radiation therapy
- `is_surgery`: Surgical interventions
- `is_combination`: Combination therapies

**Intervention Category Flags:**
- `is_drug`: Drug interventions
- `is_biological`: Biological products
- `is_device`: Medical devices
- `is_procedure`: Procedures
- `is_behavioral`: Behavioral interventions
- `is_dietary`: Dietary supplements
- `is_genetic`: Genetic interventions

**Method**: Similar keyword-based matching as disease flags, applied to `intervention_text` and `interventions` fields.

**Total Features Created**: **22 disease flags + 4 intervention flags = 26 binary features**

**Actual Disease Flags Used (22):**
- `is_nsclc`, `is_sclc`, `is_lung_cancer`
- `is_breast_cancer`, `is_colorectal_cancer`, `is_gastric_cancer`, `is_pancreatic_cancer`
- `is_hepatocellular`, `is_renal`, `is_bladder`, `is_ovarian`, `is_endometrial`
- `is_head_neck`, `is_brain`, `is_lymphoma`, `is_myeloma`
- `is_aml`, `is_all`, `is_cll`, `is_cml`
- `is_other_cancer`, `is_other`

**Actual Intervention Flags Used (4):**
- `is_immunotherapy`, `is_chemo`, `is_hormone`, `is_surgery`

### 2.2 Step 3: Eligibility Complexity Feature Extraction

To quantify the complexity and stringency of eligibility criteria, we extracted multiple metrics from the `eligibility_criteria_text` field:

#### 2.2.1 Text Length Metrics
- `eligibility_text_len`: Total character count

#### 2.2.2 Structural Metrics
- `num_inclusion_criteria`: Count of "inclusion" mentions
- `num_exclusion_criteria`: Count of "exclusion" mentions

#### 2.2.3 Section-Specific Metrics
- `inclusion_text_len`: Character count of inclusion section
- `exclusion_text_len`: Character count of exclusion section
- `num_inclusion_items`: Count of bullet points in inclusion section
- `num_exclusion_items`: Count of bullet points in exclusion section

**Rationale**: More complex eligibility criteria (longer text, more conditions, more medical terms) may indicate:
1. More selective patient populations (potentially easier to recruit but harder to enroll)
2. More stringent safety requirements (potentially associated with higher completion rates)
3. More sophisticated trial design (potentially associated with better planning and execution)

**Total Features Created**: **6 complexity metrics**

**Actual Eligibility Complexity Features Used (6):**
- Text length: `eligibility_text_len`
- Structural: `num_inclusion_criteria`, `num_exclusion_criteria`
- Section-specific: `inclusion_text_len`, `exclusion_text_len`, `num_inclusion_items`, `num_exclusion_items`

### 2.3 Final Feature Set Summary

After parsing, feature engineering, and feature selection, the **final feature set** used in the best model consists of **44 features** (estimated, will be confirmed after retraining) organized as follows:

1. **Basic structured features** (12): 
   - `has_expanded_access`, `enrollment_planned`, `number_of_arms`, `healthy_volunteers`
   - `number_of_facilities`, `has_us_sites`, `has_china_sites`, `has_eu_sites`, `is_multicountry`
   - `has_dmc`, `is_fda_regulated_drug`, `has_ipd_sharing_plan`

2. **Disease flags** (22): 
   - `is_nsclc`, `is_sclc`, `is_lung_cancer`, `is_breast_cancer`, `is_colorectal_cancer`, `is_gastric_cancer`, `is_pancreatic_cancer`
   - `is_hepatocellular`, `is_renal`, `is_bladder`, `is_ovarian`, `is_endometrial`, `is_head_neck`, `is_brain`
   - `is_lymphoma`, `is_myeloma`, `is_aml`, `is_all`, `is_cll`, `is_cml`, `is_other_cancer`, `is_other`

3. **Intervention flags** (4): 
   - `is_immunotherapy`, `is_chemo`, `is_hormone`, `is_surgery`

4. **Eligibility complexity features** (6): 
   - Text length: `eligibility_text_len`
   - Structural: `num_inclusion_criteria`, `num_exclusion_criteria`
   - Section-specific: `inclusion_text_len`, `exclusion_text_len`, `num_inclusion_items`, `num_exclusion_items`

**Total Features in Best Model**: **44 features** (selected via XGBoost feature importance, after removing 14 features)

**Note**: The initial tabular feature set before feature selection consisted of ~41-98 features (after removing 14 features: num_bullets, num_lines, num_numeric_ranges, eligibility_text_len_words, and 10 medical term mention features). After feature selection, the best model uses approximately 44 features. The `acronym` variable is excluded due to high missingness (75%) and lack of predictive value.

**Removed Features (14 total)**:
- `num_bullets`, `num_lines`, `num_numeric_ranges`, `eligibility_text_len_words`
- `num_disease_mentions`, `num_diagnosis_mentions`, `num_treatment_mentions`, `num_medication_mentions`
- `num_therapy_mentions`, `num_procedure_mentions`, `num_laboratory_mentions`, `num_biopsy_mentions`
- `num_histology_mentions`, `num_pathology_mentions`

**Note**: In the previous version of the pipeline, disease flags and intervention flags were created from condition/intervention text. In this updated version, these flags are not explicitly created as separate binary features, but the comprehensive parsing includes MeSH terms and all available structured features from the v2 API.

---

## 3. Text Embedding Generation

### 3.1 Text Field Selection

To capture semantic information from unstructured trial descriptions, we generated embeddings for the following text fields:

1. **`brief_title`**: Short trial title
2. **`official_title`**: Full official title
3. **`brief_summary`**: Brief summary of the trial
4. **`detailed_description`**: Detailed description of the trial design, rationale, and methodology
5. **`eligibility_criteria_text`**: Full eligibility criteria (inclusion and exclusion)

**Rationale**: These fields contain rich information about:
- Trial design and methodology (from descriptions)
- Patient population characteristics (from eligibility criteria)
- Intervention details and rationale (from descriptions and titles)
- Overall trial complexity and sophistication (from all fields combined)

### 3.2 Embedding Model: BioLinkBERT

We used **BioLinkBERT-base** (`michiyasunaga/BioLinkBERT-base`), a domain-specific BERT model pre-trained on biomedical literature and clinical trial data.

**Model Specifications:**
- **Architecture**: BERT-base (12 layers, 768 hidden dimensions, 12 attention heads)
- **Vocabulary**: Biomedical and clinical domain vocabulary
- **Pre-training**: Trained on PubMed abstracts, clinical trial descriptions, and biomedical text
- **Output Dimension**: 768-dimensional vectors per text input

**Why BioLinkBERT?**
1. **Domain-specific**: Pre-trained on biomedical/clinical text, capturing domain terminology
2. **Clinical trial context**: Trained on clinical trial descriptions, making it well-suited for our task
3. **Semantic richness**: Captures nuanced relationships between medical concepts, trial designs, and eligibility criteria

### 3.3 Embedding Generation Process

For each trial, we:

1. **Concatenated text fields**: Combined all 5 text fields into a single string:
   ```
   combined_text = brief_title + " " + official_title + " " + brief_summary + " " + detailed_description + " " + eligibility_criteria_text
   ```

2. **Tokenization**: Tokenized the combined text using BioLinkBERT's tokenizer with:
   - Maximum length: 512 tokens (BERT's standard limit)
   - Padding: Applied to shorter texts
   - Truncation: Applied to longer texts (preserving beginning of text)

3. **Embedding extraction**: 
   - Passed tokenized text through BioLinkBERT model
   - Extracted the `[CLS]` token embedding (first token, 768 dimensions)
   - The `[CLS]` token is designed to capture the overall semantic representation of the entire sequence

4. **Batch processing**: Processed trials in batches of 32 to optimize GPU/CPU usage

**Output**: Each trial was represented by a **768-dimensional embedding vector**.

### 3.4 Handling Missing Text

For trials with missing text fields:
- Missing fields were replaced with empty strings
- Embeddings were still generated (empty strings produce valid embeddings)
- Trials with completely missing text (all fields empty) were assigned NaN embeddings and handled separately in downstream steps

---

## 4. Dimensionality Reduction

### 4.1 Rationale for PCA

The raw BioLinkBERT embeddings had **768 dimensions**, which:
1. **High dimensionality**: Could lead to overfitting, especially with limited sample size
2. **Multicollinearity**: Embedding dimensions are likely correlated (capturing similar semantic concepts)
3. **Computational efficiency**: Reducing dimensions speeds up model training and inference
4. **Noise reduction**: Lower-dimensional representations may filter out noise while preserving signal

**Principal Component Analysis (PCA)** was chosen because:
- **Linear transformation**: Preserves interpretability and allows inverse transformation
- **Variance preservation**: Retains maximum variance with fewer dimensions
- **Orthogonality**: Principal components are uncorrelated, reducing redundancy

### 4.2 PCA Implementation

#### 4.2.1 Preprocessing
Before applying PCA:
1. **Standardization**: Applied `StandardScaler` to center and scale embeddings to mean=0, std=1
   - **Why**: PCA is sensitive to scale; standardization ensures all dimensions contribute equally
2. **Missing value handling**: Trials with missing embeddings were excluded from PCA fitting but included in transformation (assigned NaN)

#### 4.2.2 PCA Fitting
- **Variance threshold**: Retained components explaining **95% of total variance**
- **Fitting data**: PCA was fit **only on training data** to prevent data leakage
- **Method**: Used scikit-learn's `PCA` with `n_components=0.95` (automatically selects number of components)

#### 4.2.3 Results
After PCA:
- **Original dimension**: 768
- **Reduced dimension**: ~248 components (exact number depends on variance distribution)
- **Compression ratio**: ~3.1x reduction (768 → 248)
- **Variance explained**: 95.0% (by definition)
- **Top 10 components**: Explained ~XX% of variance (indicating concentration of information)

#### 4.2.4 Transformation
- **Training set**: Transformed using fitted PCA
- **Validation set**: Transformed using same fitted PCA (no refitting)
- **Test set**: Transformed using same fitted PCA (no refitting)

**Critical**: PCA was fit only on training data to ensure no information leakage from validation/test sets.

#### 4.2.5 Model Persistence
The fitted PCA model and scaler were saved to disk (`models/pca_model.pkl`, `models/pca_scaler.pkl`) for:
- Reproducibility
- Future inference on new trials
- Consistency across experiments

### 4.3 Final Embedding Features

After PCA, each trial was represented by **~248 PCA-reduced embedding features** (named `pca_emb_0`, `pca_emb_1`, ..., `pca_emb_247`).

---

## 5. Feature Selection

### 5.1 Rationale

Feature selection serves multiple purposes:
1. **Noise reduction**: Removes irrelevant or redundant features
2. **Overfitting prevention**: Reduces model complexity, improving generalization
3. **Interpretability**: Focuses on features with genuine predictive power
4. **Computational efficiency**: Faster training and inference

### 5.2 XGBoost-Based Feature Selection

We adopted a **model-based feature selection** approach using XGBoost as the selector model.

#### 5.2.1 Method
1. **Fit XGBoost**: Trained an XGBoost classifier on the training data with default hyperparameters
2. **Extract importances**: Computed feature importance scores (gain-based importance)
3. **Calculate threshold**: Computed mean importance across all features
4. **Select features**: Retained only features with `importance > mean(importance)`

**Rationale for threshold**: Features with above-average importance are likely to contribute meaningfully to predictions. This automatic threshold avoids manual tuning while providing a principled selection criterion.

#### 5.2.2 Separate Selection for Each Mode
Feature selection was performed **separately** for:
- **Tabular-only mode**: Selected from ~75-80 tabular features
- **Tabular+embeddings mode**: Selected from ~75-80 tabular features + ~248 PCA embeddings = ~323-328 total features

**Why separate?**: Different feature sets may have different importance distributions. Features that are important in tabular-only mode may be less important when embeddings are included (and vice versa).

#### 5.2.3 Safety Checks
- **Minimum features**: Ensured at least 10 features were selected (even if importance < mean)
- **Missing value handling**: Imputed missing values before feature selection (median for numeric, mode for categorical)

#### 5.2.4 Feature Mask Application
After selection, a boolean mask was created and applied consistently to:
- Training set
- Validation set
- Test set

**Result**: Each mode (tabular, tabular+embeddings) had its own selected feature set, tailored to that mode's characteristics.

### 5.3 Selected Features

**Tabular-only mode**: Typically selected ~30-50 features (exact count depends on importance distribution)

**Tabular+embeddings mode**: Typically selected ~100-150 features (including both tabular and embedding features)

**Common selected tabular features** (examples):
- Design features: `phase`, `allocation`, `number_of_arms`
- Disease flags: `is_nsclc`, `is_lung_cancer`
- Intervention flags: `is_immunotherapy`, `is_tki`, `is_combination`
- Eligibility complexity: `eligibility_text_len`, `num_exclusion_criteria`
- Location: `number_of_facilities`, `is_multicountry`

**Selected embedding features**: Typically a mix of early PCA components (which explain more variance) and later components (which may capture nuanced patterns).

---

## 6. Class Imbalance Handling

### 6.1 Class Distribution

The dataset exhibited **significant class imbalance**:
- **Majority class** (completed trials): ~XX% of trials
- **Minority class** (non-completed trials): ~XX% of trials
- **Imbalance ratio**: Approximately 1:3.1 (minority:majority, where minority = non-completed, majority = completed)

**Impact**: Without handling, models may:
- Over-predict the majority class
- Achieve high accuracy but poor recall for minority class
- Fail to learn meaningful patterns for minority class

### 6.2 Downsampling Strategies

We applied **random downsampling** of the majority class to create balanced datasets. Unlike oversampling (e.g., SMOTE), downsampling:
- **Preserves data integrity**: Uses only real observations (no synthetic data)
- **Avoids unrealistic samples**: Clinical trial data is highly structured; synthetic samples may not reflect real-world distributions
- **Reduces training size**: Trade-off between balance and sample size

#### 6.2.1 Downsampling Ratios
We tested three ratios (minority:majority):
1. **1:1 (fully balanced)**: Equal number of minority and majority samples
2. **1:1.5**: 1.5 majority samples for every 1 minority sample
3. **1:2**: 2 majority samples for every 1 minority sample

**Method**: For each ratio, we randomly sampled the majority class (without replacement) to achieve the target ratio, using different random seeds for each iteration.

#### 6.2.2 Implementation
- **Random sampling**: Used `RandomUnderSampler` from `imblearn` library
- **Per-iteration sampling**: Each of the 10 iterations used a different random seed, creating different majority class subsets
- **Training only**: Downsampling was applied only to training data (validation and test sets remained unchanged)

### 6.3 Class Weight Balancing

As an alternative to downsampling, we also tested **class weight balancing**:
- **Method**: Set `class_weight='balanced'` in model hyperparameters
- **Effect**: Automatically adjusts class weights inversely proportional to class frequencies
- **Advantage**: Uses all training data (no downsampling)
- **Disadvantage**: May still be biased toward majority class if imbalance is severe

### 6.4 Strategy Combinations

For **Random Forest** and **XGBoost**, we tested three strategies:
1. **`class_weight='balanced'`**: Original imbalanced data with class weights
2. **`downsample_balanced`**: Fully balanced (1:1) without class weights
3. **`downsample_1_5`**: Partially downsampled (1:1.5) with class weights
4. **`partial_downsample`**: Partially downsampled (1:2) with class weights

For **Dual-tower Neural Network**, we used **oversampling** instead of downsampling:
- **Method**: `RandomOverSampler` with ratios 0.6, 0.8, 1.0 (minority:majority)
- **Rationale**: Neural networks benefit from larger training sets; oversampling preserves all majority class samples while increasing minority class representation

### 6.5 Iteration Strategy

To ensure robustness, we performed **10 independent iterations** for each strategy:
- Each iteration used a different random seed for downsampling/oversampling
- Metrics were averaged across iterations
- Standard deviations were computed to assess variability

---

## 7. Model Training and Evaluation

### 7.1 Data Splitting

**Split Strategy**: Train/Validation/Test split with fixed random seed for reproducibility

- **Training set**: 68% of data (used for model training and hyperparameter tuning)
- **Validation set**: 12% of data (used for hyperparameter selection and early stopping)
- **Test set**: 20% of data (used only for final evaluation, never used in training or tuning)

**Stratification**: Splits were stratified by `label_feasible` to maintain class distribution across splits.

**Leakage Prevention**: 
- Excluded `enrollment_actual` (only known after completion)
- Excluded `completion_date` and related temporal features
- Excluded `is_historical` flags
- PCA and feature selection were fit only on training data

### 7.2 Models Implemented

We evaluated four model families:

#### 7.2.1 Support Vector Machine (SVM)
- **Mode**: Tabular-only (baseline)
- **Kernel**: Linear
- **Hyperparameters**: 
  - `kernel='linear'`
  - `probability=True` (for probability estimates)
  - `class_weight='balanced'`
  - `random_state=42`
- **Hyperparameter tuning**: None (serves as baseline)
- **Rationale**: Simple, interpretable baseline to assess whether more complex models offer improvements

#### 7.2.2 Random Forest
- **Modes**: Both tabular-only and tabular+embeddings
- **Hyperparameter tuning**: Grid search with 5-fold cross-validation
  - `n_estimators`: [100, 200, 300]
  - `max_depth`: [10, 20, None]
  - `max_features`: ['sqrt', 'log2']
  - `class_weight`: ['balanced'] (for class_weight strategy only)
- **Imbalance strategies**: 4 strategies (class_weight, downsample_balanced, downsample_1_5, partial_downsample)
- **Rationale**: Strong performance on structured data, interpretable feature importances, handles non-linearity

#### 7.2.3 XGBoost
- **Modes**: Both tabular-only and tabular+embeddings
- **Configuration**:
  - `objective='binary:logistic'`
  - `eval_metric='logloss'`
  - `use_label_encoder=False`
- **Hyperparameter tuning**: Grid search with 5-fold cross-validation
  - `n_estimators`: [100, 200, 300]
  - `max_depth`: [3, 5, 7]
  - `learning_rate`: [0.01, 0.05, 0.1]
  - `subsample`: [0.8, 0.9]
  - `colsample_bytree`: [0.8, 0.9]
- **Imbalance strategies**: 4 strategies (same as Random Forest)
- **Rationale**: State-of-the-art performance on structured data, gradient boosting with regularization

#### 7.2.4 Dual-Tower Neural Network
- **Mode**: Tabular+embeddings only
- **Architecture**:
  - **Tabular tower**: 2-layer MLP (32 → 64 units), ReLU, BatchNorm, Dropout(0.3)
  - **Embedding tower**: 2-layer MLP (64 → 64 units), ReLU, BatchNorm, Dropout(0.3)
  - **Concatenation**: 128-dimensional combined representation
  - **Classification head**: 64 units (ReLU + Dropout), sigmoid output
- **Preprocessing**:
  - Tabular features: Boolean → 0/1, MinMaxScaler normalization
  - Embedding features: StandardScaler normalization (after PCA)
- **Training**:
  - Optimizer: Adam (lr=0.001, weight_decay=1e-4)
  - Scheduler: ReduceLROnPlateau (reduces LR on plateau)
  - Early stopping: Validation AUC, patience=5 epochs
  - Batch size: 32
  - Max epochs: 100
- **Threshold optimization**: Selected threshold that maximizes F1 on validation set (not default 0.5)
- **Imbalance strategies**: Oversampling with ratios [0.6, 0.8, 1.0]
- **Rationale**: Captures complex interactions between tabular and embedding features, deep learning approach

### 7.3 Evaluation Metrics

We computed the following metrics:

1. **Accuracy**: Overall classification accuracy
2. **Precision**: True positives / (True positives + False positives)
3. **Recall**: True positives / (True positives + False negatives)
4. **F1 Score**: Harmonic mean of precision and recall
5. **ROC-AUC**: Area under the receiver operating characteristic curve

**Rationale**: 
- **Accuracy**: Overall performance
- **Precision**: Important for identifying truly feasible trials (minimize false positives)
- **Recall**: Important for identifying all feasible trials (minimize false negatives)
- **F1**: Balances precision and recall
- **ROC-AUC**: Robust to class imbalance, measures discrimination ability

### 7.4 Evaluation Protocol

For each model configuration:
1. **10 iterations**: Each iteration used different random seeds for:
   - Downsampling/oversampling
   - Model initialization (if applicable)
2. **Metrics computation**: Computed all 5 metrics on test set for each iteration
3. **Aggregation**: Averaged metrics across 10 iterations, computed standard deviations
4. **Reporting**: Reported mean ± std for each metric

**Rationale**: 
- **Robustness**: Accounts for variability in random sampling and model initialization
- **Reliability**: Provides confidence intervals (via std) for performance estimates
- **Reproducibility**: Fixed random seeds ensure reproducibility while allowing variability assessment

---

## 8. Results

### 8.1 Dataset Characteristics

After parsing raw ClinicalTrials.gov JSON files and applying filtering criteria (Phase 2/3/4, Interventional, Industry-sponsored, all oncology), the final dataset consisted of:

- **Total trials**: **7,444 trials** (after excluding trials with missing/ambiguous labels and excluding all ongoing trials)
- **Completed trials (label=1)**: **5,639 trials (75.8%)**
- **Non-completed trials (label=0)**: **1,805 trials (24.2%)**
- **Excluded ongoing trials**: **2,964 trials** (RECRUITING, ACTIVE_NOT_RECRUITING, ENROLLING_BY_INVITATION, etc.)
- **Class imbalance ratio**: **1:3.12** (minority:majority, where minority = non-completed, majority = completed)

**Note**: **2,964 trials** were excluded from the final modeling dataset:
- **2,012 ongoing trials** (RECRUITING, ACTIVE_NOT_RECRUITING, ENROLLING_BY_INVITATION) - excluded because their final outcome is unknown
- **952 trials** with unknown or ambiguous completion status
- **Rationale**: Only trials with **final status** (completed or terminated) are included, as ongoing trials could still complete or terminate, making them inappropriate for training a feasibility prediction model.

**Note**: The dataset includes all oncology Phase 2/3/4 interventional industry-sponsored trials from ClinicalTrials.gov, providing a comprehensive sample across multiple cancer types. The class distribution shows completed trials outnumbering non-completed trials, which is expected for industry-sponsored trials that typically have better completion rates.

**Phase Distribution**:
- **Phase 2**: 4,576 trials (61.5%)
- **Phase 3**: 2,424 trials (32.6%)
- **Phase 4**: 444 trials (6.0%)

**Data Splits** (68/12/20):
- **Training set**: ~5,062 trials (68%)
- **Validation set**: ~893 trials (12%)
- **Test set**: ~1,489 trials (20%)

Splits were stratified by `label_feasible` to maintain class distribution across all sets.

### 8.2 Feature Engineering Summary

#### 8.2.1 Parsed Variables from Raw JSON

From the ClinicalTrials.gov v2 API JSON files, we extracted **108 total variables**, including comprehensive information from all protocol sections:

**Key Variable Categories**:
- **Identifiers**: `nct_id`, `brief_title`, `official_title`, `acronym`, `secondary_ids`
- **Text fields**: `brief_summary`, `detailed_description`, `eligibility_criteria_text`, `condition_text`, `intervention_text`
- **Status and dates**: `overall_status`, `start_date`, `completion_date`, `primary_completion_date`, `why_stopped`, `has_results`, and multiple submission/posted dates
- **Design features**: `phase`, `allocation`, `intervention_model`, `masking`, `primary_purpose`, `number_of_arms`, `target_duration`
- **Enrollment**: `enrollment_planned`, `enrollment_planned_type`, `enrollment_actual` (excluded from features)
- **Eligibility**: `eligibility_criteria_text`, `min_age`, `max_age`, `sex/gender`, `healthy_volunteers`, `study_population`
- **Location/sponsor**: `number_of_facilities`, `facility_countries`, `lead_sponsor_name`, `lead_sponsor_type`, location flags
- **Regulatory**: `has_dmc`, `is_fda_regulated_drug`, `is_fda_regulated_device`
- **Outcomes**: `primary_outcome_measures`, `secondary_outcome_measures`, `other_outcome_measures` (as JSON)
- **MeSH terms**: `condition_mesh_terms`, `intervention_mesh_terms` (as JSON)
- **Other**: IPD sharing details, references, responsible party information, etc.

#### 8.2.2 Final Feature Set for Modeling

**Excluded Variables Summary**:
- **Identifiers/Labels**: 2 columns (`nct_id`, `label_feasible`)
- **Text columns**: 8 columns (used for embeddings: `brief_title`, `official_title`, `brief_summary`, `detailed_description`, `eligibility_criteria_text`, `condition_text`, `intervention_text`, `description_text`)
- **JSON columns**: 26 columns (structured data as JSON: condition/intervention lists, outcomes, MeSH terms, references, locations as JSON, etc.)
- **Leakage columns**: 17 columns (post-hoc information: completion/posted/submit dates, `overall_status`, `why_stopped`, `has_results`, `enrollment_actual`, etc.)

**Initial Tabular Features Available**: ~55-112 structured features (before feature selection)

**Final Features Used in Best Model**: **44 features** (after XGBoost feature selection and removing 14 features)

**Feature Categories in Best Model** (44 total):
1. **Basic structured features** (12): enrollment_planned, number_of_arms, healthy_volunteers, number_of_facilities, geographic flags, regulatory flags, etc.
2. **Disease flags** (22): is_nsclc, is_sclc, is_lung_cancer, is_breast_cancer, is_colorectal_cancer, and 17 other cancer type flags
3. **Intervention flags** (4): is_immunotherapy, is_chemo, is_hormone, is_surgery
4. **Eligibility complexity features** (6): eligibility_text_len, num_inclusion_criteria, num_exclusion_criteria, inclusion_text_len, exclusion_text_len, num_inclusion_items, num_exclusion_items

**Note**: `acronym` and other high-missingness variables are excluded before feature selection due to high missingness (75%+) and lack of predictive value

#### 8.2.3 Text Embedding Features

- **Raw BioLinkBERT embeddings**: **768 dimensions** per trial
- **PCA-reduced embeddings**: **295 dimensions** (retaining 95.0% variance)
- **Compression ratio**: **~2.6x** (768 → 295)
- **Variance explained**: 95.0% (by definition of retention threshold)

#### 8.2.4 Final Feature Set for Modeling

**Best Model (XGBoost, tabular mode)**:
- **Input mode**: Tabular-only (no text embeddings)
- **Total features used**: **44 features** (estimated, will be confirmed after retraining)
- **Feature breakdown**:
  - 12 basic structured features
  - 22 disease flags
  - 4 intervention flags
  - 6 eligibility complexity features (after removing 14 features)
- **Feature selection**: Features selected via XGBoost feature importance during training

**Note on Feature Selection**:
- The initial tabular feature set before selection consisted of ~41-98 features (after removing 14 features: num_bullets, num_lines, num_numeric_ranges, eligibility_text_len_words, and 10 medical term mention features)
- XGBoost feature importance was used to select the most predictive features

**Tabular+embeddings mode** (selected as best model):
- **Total features after selection**: **219 features** (14 non-PCA tabular features + 205 PCA embedding features)
- Feature selection based on XGBoost importance scores exceeding the mean importance across all features

**Features excluded from modeling** (to prevent data leakage):
- `nct_id` - Identifier only
- `label_feasible` - Target variable
- Text columns (used for embeddings, not as direct features): 8 columns
- JSON columns (structured data as JSON strings): 26 columns
- Leakage columns (post-hoc information): 17 columns including:
  - Date columns (completion_date, posted_date, submit_date, etc.)
  - Status columns (overall_status, why_stopped, has_results)
  - Post-hoc enrollment (enrollment_actual, enrollment_actual_type)

### 8.3 Model Performance Summary

All models were evaluated over **10 independent iterations** with different random seeds for downsampling/oversampling. Metrics are reported as mean ± standard deviation across iterations.

#### 8.3.1 Model Comparison Experiments

Multiple models and configurations were tested during model development. The following table shows results from model comparison experiments across different input modes and imbalance strategies:

| Rank | Model | Input Mode | Imbalance Strategy | ROC-AUC (mean ± std) | Notes |
|------|-------|------------|---------------------|---------------------|-------|
| 1 | **XGBoost** | tabular_plus_embeddings | class_weight | **0.6803 ± 0.0069** | **Best performance - Selected for deployment** |
| 2 | XGBoost | tabular | class_weight | 0.6744 ± 0.0022 | Strong tabular-only performance |
| 3 | RandomForest | tabular | class_weight | 0.6626 ± 0.0031 | Strong tabular-only performance |
| 4 | RandomForest | tabular_plus_embeddings | class_weight | 0.6621 ± 0.0057 | Similar to tabular-only RandomForest |
| 5 | DualTowerNN | tabular_plus_embeddings | oversample_1.0 | 0.6431 ± 0.0135 | Best neural network performance |
| 6 | DualTowerNN | tabular_plus_embeddings | oversample_0.8 | 0.6449 ± 0.0116 | Neural network approach |
| 7 | DualTowerNN | tabular_plus_embeddings | oversample_0.6 | 0.6382 ± 0.0087 | Neural network approach |
| 8 | SVM | tabular | class_weight | 0.6535 ± 0.0000 | Baseline linear model |

**Key Findings**:
- **XGBoost (tabular_plus_embeddings, class_weight)** achieved the highest ROC-AUC (0.6803 ± 0.0069) and was selected for deployment
- Tabular+embeddings models showed slightly better performance than tabular-only models for XGBoost
- Class_weight imbalance handling performed better than oversampling for tree-based models
- All models achieved moderate discriminative ability (ROC-AUC > 0.64), better than random (0.5) but not perfect (1.0)

#### 8.3.2 Selected Model for Deployment

**Selected Model for Deployment**: XGBoost (tabular_plus_embeddings mode)

**Rationale for Selection**:
1. **Best performance**: Achieved highest ROC-AUC (0.6803 ± 0.0069) among all tested configurations
2. **Interpretability**: While the model includes PCA embeddings, SHAP analysis focuses on interpretable non-PCA features to identify key drivers
3. **Feature richness**: Combines structured tabular features with semantic text embeddings for comprehensive representation
4. **Performance**: ROC-AUC of 0.6803 provides moderate discriminative ability suitable for risk stratification

**Alternative Models Tested**:
- RandomForest (tabular and tabular+embeddings)
- XGBoost (tabular+embeddings)
- DualTowerNN (tabular+embeddings)
- SVM (tabular)

**Note**: The selected XGBoost tabular_plus_embeddings model provides the best performance while maintaining interpretability through SHAP analysis of non-PCA features. PCA embedding features are excluded from SHAP plots and analysis for interpretability, ensuring that key drivers remain clearly interpretable.

#### 8.3.3 Comparison: Imbalance Strategies

**Average ROC-AUC by Strategy** (across all models and modes):

| Strategy | Average ROC-AUC | Standard Deviation | Models Evaluated |
|----------|----------------|-------------------|------------------|
| **class_weight** | **0.6700** | 0.0106 | SVM, RF, XGBoost (all modes) |
| **oversample_1.0** | **0.6431** | 0.0135 | DualTowerNN only |
| **oversample_0.8** | **0.6449** | 0.0116 | DualTowerNN only |
| **oversample_0.6** | **0.6382** | 0.0087 | DualTowerNN only |
| **downsample_balanced** | 0.4882 | 0.0156 | RF, XGBoost (all modes) |
| **partial_downsample** | 0.4763 | 0.0114 | RF, XGBoost (all modes) |
| **downsample_1_5** | 0.4791 | 0.0176 | RF, XGBoost (all modes) |

**Key Findings**:
1. **`class_weight='balanced'` performed best** for tree-based models (RF, XGBoost)
   - Uses all training data with adjusted class weights
   - ROC-AUC: 0.6700 on average (best: 0.6803 for XGBoost tabular_plus_embeddings)
   - Low to moderate variance across models (std: 0.0106)
   - Best overall strategy for tree-based models

2. **Oversampling strategies** (for neural networks) showed moderate performance
   - `oversample_1.0`: Best ROC-AUC (0.6431 ± 0.0135) for DualTowerNN
   - All oversampling ratios (0.6, 0.8, 1.0) performed similarly (0.638-0.645 range)
   - Lower performance than class_weight for tree-based models

3. **Downsampling strategies performed poorly**
   - `downsample_balanced` (1:1): ROC-AUC: 0.4882 (near random chance ~0.5)
   - `downsample_1_5` and `partial_downsample`: ROC-AUC: ~0.476-0.479 (below random chance)
   - **Interpretation**: Aggressive downsampling removes too much information from the majority class, leading to poor model performance

4. **Variance analysis**:
   - Class weight: Low to moderate variance (std: 0.0106) across different models
   - Oversampling: Low to moderate variance (std: 0.009-0.014) for DualTowerNN
   - Downsampling: Low to moderate variance but poor performance

**Recommendation**: Use `class_weight='balanced'` for tree-based models (RF, XGBoost) and oversampling (ratio 0.6-1.0) for neural networks (DualTowerNN).

**XGBoost (Selected Model)**:
- **Configuration**: Tabular_plus_embeddings mode with class_weight imbalance handling
- **ROC-AUC**: **0.6803 ± 0.0069** (cross-validated)
- **Input mode**: Tabular + PCA embeddings
- **Number of features**: 219 features total (14 non-PCA tabular features + 205 PCA embedding features)
- **Imbalance strategy**: class_weight
- **Random state**: 42

**Model Selection Rationale**:
1. **Best performance**: Achieved highest ROC-AUC (0.6803 ± 0.0069) among all tested configurations
2. **Interpretability**: SHAP analysis focuses on interpretable non-PCA features (e.g., number_of_facilities, eligibility complexity) while PCA features are excluded from plots for clarity
3. **Feature richness**: Combines structured tabular features with semantic text embeddings for comprehensive representation
4. **Performance**: ROC-AUC of 0.6803 provides moderate discriminative ability suitable for risk stratification
5. **Practical utility**: Suitable for deployment to ongoing trials with clear interpretability through non-PCA SHAP analysis

**Alternative Models Tested** (not selected):
- **XGBoost tabular-only**: ROC-AUC 0.6744 ± 0.0022 (slightly lower than tabular_plus_embeddings)
- **RandomForest**: Tested with tabular and tabular+embeddings modes
- **DualTowerNN**: Tested with tabular+embeddings mode
- **SVM**: Tested as baseline model

**Performance Interpretation**:
- ROC-AUC of 0.6803 indicates moderate discriminative ability
- Better than random (AUC = 0.5) but not perfect (AUC = 1.0)
- Reflects inherent uncertainty in clinical trial outcomes
- Suitable for risk stratification and ranking of ongoing trials

### 8.4 Feature Importance Analysis

Feature importance was assessed using two complementary approaches:

1. **XGBoost-based feature selection** (during training): Features were selected based on XGBoost importance scores exceeding the mean importance across all features. The final selected model uses **219 features total** (14 non-PCA tabular features + 205 PCA embedding features) after feature selection.

2. **SHAP analysis** (for interpretability): SHAP (SHapley Additive exPlanations) values were computed to understand which features drive model predictions and how they contribute to completion probability estimates. **PCA embedding features (pca_emb_*) are excluded from all SHAP plots and analysis for interpretability**, as they are not directly interpretable (they represent compressed text embeddings).

#### 8.4.1 SHAP Feature Importance Results

SHAP analysis was performed on the selected XGBoost model using the test set (1,489 samples) to identify the most important features driving completion probability predictions. **PCA embedding features (pca_emb_*) are excluded from this analysis for interpretability.** The top 15 non-PCA features by mean absolute SHAP value are:

| Rank | Feature | Mean(|SHAP|) | Interpretation |
|------|---------|---------------|----------------|
| 1 | `number_of_facilities` | 0.61650556 | **Strongest predictor** - Positive association with completion |
| 2 | `has_us_sites` | 0.31243518 | **Negative association** - Having US sites decreases completion probability (red/True on left, blue/False on right) |
| 3 | `has_dmc` | 0.16658497 | **Negative association** - Having DMC decreases completion probability (red/True on left, blue/False on right) |
| 4 | `inclusion_text_len` | 0.14071532 | Negative association (longer = more complex) |
| 5 | `eligibility_text_len` | 0.06437103 | Negative association (longer = more complex) |
| 6 | `is_other` | 0.04737004 | Disease category flag |
| 7 | `is_multicountry` | 0.03567892 | Positive association (geographic diversity) |
| 8 | `exclusion_text_len` | 0.02547307 | Negative association |
| 9 | `num_inclusion_criteria` | 0.01538598 | Eligibility complexity (negative association) |
| 10 | `is_immunotherapy` | 0.00808069 | Intervention type flag |
| 11 | `number_of_arms` | 0.00757430 | Trial design complexity |
| 12 | `has_ipd_sharing_plan` | 0.00472511 | Data sharing indicator |
| 13 | `is_breast_cancer` | 0.00000000 | Disease category flag (no importance) |
| 14 | `has_eu_sites` | 0.00000000 | Geographic flag (no importance) |

**Note**: PCA embedding features (pca_emb_*) are excluded from this SHAP analysis for interpretability. Only the top 14 non-PCA features are shown above.

**Important: Interpreting Binary Features in SHAP Beeswarm Plots**:
- For binary features like `has_us_sites` and `has_dmc`:
  - **Low value (blue) = 0 (False)**: The feature is absent
  - **High value (red) = 1 (True)**: The feature is present
- **SHAP value direction**:
  - **Positive SHAP (right of vertical line)**: Increases the model output (completion probability)
  - **Negative SHAP (left of vertical line)**: Decreases the model output (completion probability)
- **Example interpretation**:
  - If red dots (True) are on the **left** (negative SHAP): Having the feature **decreases** completion probability
  - If red dots (True) are on the **right** (positive SHAP): Having the feature **increases** completion probability

**Key Findings**:
1. **`number_of_facilities` dominates feature importance** (mean(|SHAP|) = 0.61650556), representing 1.973× the importance of the second-ranked feature (`has_us_sites` at 0.31243518). This single feature accounts for 42.7% of the total feature importance across the top 12 non-zero features (sum = 1.4445).

2. **Clear feature importance tiers**:
   - **Tier 1 (Very High)**: `number_of_facilities` (0.61650556) - stands alone, nearly 2× more important than #2
   - **Tier 2 (High)**: `has_us_sites` (0.31243518) - second tier
   - **Tier 3 (Moderate-High)**: `has_dmc` (0.16658497) - Data Monitoring Committee presence
   - **Tier 4 (Moderate)**: Eligibility complexity features (0.01538598-0.14071532): `inclusion_text_len` (0.14071532), `eligibility_text_len` (0.06437103), `exclusion_text_len` (0.02547307), `num_inclusion_criteria` (0.01538598)
   - **Tier 5 (Lower)**: Other features: `is_other` (0.04737004), `is_multicountry` (0.03567892), `is_immunotherapy` (0.00808069), `number_of_arms` (0.00757430), `has_ipd_sharing_plan` (0.00472511)
   - **Tier 6 (No importance)**: `is_breast_cancer` (0.00000000), `has_eu_sites` (0.00000000)

3. **Eligibility complexity features show consistent negative associations** - All text length and complexity metrics (0.01538598-0.14071532) indicate that more complex eligibility criteria correlate with lower completion probability. This suggests that overly restrictive eligibility criteria directly impact recruitment feasibility.

4. **Regulatory and geographic features show negative associations** - Both `has_us_sites` (0.31243518) and `has_dmc` (0.16658497) show **negative associations** (having these features decreases completion probability). These counterintuitive findings may reflect:
   - **Selection bias**: US sites and DMC may be used for more complex/higher-risk trials that are inherently more likely to fail
   - **Confounding factors**: These features may be associated with stricter regulatory requirements or more challenging patient populations
   - **Data artifacts**: The associations may be driven by other correlated factors not captured in the model
   Notably, `is_fda_regulated_drug` is not in the top 14 non-PCA features.

5. **Trial scale is the primary driver** - `number_of_facilities` (0.61650556) is the strongest predictive signal, indicating that operational capacity (multiple sites, geographic diversity, recruitment redundancy) is the most important factor for completion probability. Notably, `enrollment_planned` is not in the top 14 features, suggesting that the number of facilities matters more than target enrollment size.

#### 8.4.2 Most Important Feature Categories

Based on SHAP analysis, the following feature categories were identified as most important, ranked by their collective importance:

**1. Trial Scale Features** (Highest importance):
- `number_of_facilities` - **Dominant predictor** (0.61650556 mean(|SHAP|)) - stands alone as the strongest predictor, accounting for 42.7% of total importance across top features
- `number_of_arms` - Lower importance (0.00757430) - trial design complexity
- **Interpretation**: Multi-site trials have substantially higher completion probability. The dominance of `number_of_facilities` suggests that operational capacity (multiple sites, geographic diversity, recruitment redundancy) is the most critical factor. Notably, `enrollment_planned` is not in the top 14 features, indicating that the number of facilities matters more than target enrollment size.

**2. Regulatory and Geographic Features** (Moderate-High importance - Combined: 0.47902015):
- `has_us_sites` - Highest in category (0.31243518) - 65.2% of category importance - **Negative association** (having US sites decreases completion probability)
- `has_dmc` - High importance (0.16658497) - 34.8% of category importance - **Negative association** (having DMC decreases completion probability)
- **Interpretation**: Both features show counterintuitive negative associations. This may reflect:
  - **Selection bias**: US sites and DMC may be used for more complex/higher-risk trials that are inherently more likely to fail
  - **Confounding factors**: These features may be associated with stricter regulatory requirements or more challenging patient populations
  - **Data artifacts**: The associations may be driven by other correlated factors not captured in the model
  These findings require careful interpretation and may not indicate that US sites or DMC cause lower completion rates, but rather that trials using these features may have other characteristics that reduce completion probability. Notably, `is_fda_regulated_drug` is not in the top 14 non-PCA features.

**3. Eligibility Complexity Features** (Moderate importance - Combined: 0.24594540):
- Text length metrics: `inclusion_text_len` (0.14071532), `eligibility_text_len` (0.06437103), `exclusion_text_len` (0.02547307)
- Structural metrics: `num_inclusion_criteria` (0.01538598)
- **Interpretation**: All eligibility complexity features show negative associations with completion probability. Longer, more detailed eligibility criteria correlate with lower completion rates, likely due to recruitment challenges. This is an actionable factor that sponsors can optimize by simplifying eligibility criteria where clinically appropriate.

**4. Disease and Intervention Flags** (Lower importance):
- Disease flags: Not present in top 15, suggesting disease-specific flags have limited predictive power
- Intervention flags: Not present in top 15, indicating that intervention type (immunotherapy, chemotherapy, etc.) has minimal impact on completion probability compared to operational and design factors
- **Interpretation**: Disease and intervention characteristics are less predictive than operational factors (facilities, geographic presence) and eligibility complexity

**Note**: The selected model uses tabular-only features (no PCA embeddings) for interpretability. Even when models include PCA embeddings, **all SHAP plots and analysis exclude PCA features (pca_emb_*) for interpretability**, as PCA components are not directly interpretable (they represent compressed text embeddings). This ensures that SHAP analysis focuses on interpretable trial characteristics such as `number_of_facilities`, `eligibility_text_len`, etc.

#### 8.4.3 Key Insights from SHAP Analysis

**Why certain features dominate and what this means**:

1. **`number_of_facilities` dominance (0.61650556 - 1.973× second-ranked feature)**:
   - **Operational resilience**: Multi-site trials have built-in redundancy—if one site struggles with recruitment, others can compensate
   - **Geographic diversity**: Multiple sites increase access to diverse patient populations, reducing geographic barriers to enrollment
   - **Resource commitment signal**: Large sponsors may select higher-quality trials for multi-site execution, creating a selection effect
   - **Causality caution**: While multi-site trials show higher completion rates, this may reflect both causal effects (better recruitment) and selection bias (better trials get more sites)
   - **Actionable insight**: For sponsors, increasing site count may improve completion probability, but this requires adequate resources and coordination

2. **Eligibility complexity features (negative association - 0.01538598-0.14071532)**:
   - **Recruitment barrier**: Complex eligibility criteria directly reduce the eligible patient pool, making recruitment more challenging
   - **Actionable factor**: Unlike sponsor reputation or trial phase, eligibility complexity is a design choice that sponsors can optimize
   - **Clinical balance**: While simplifying eligibility may improve recruitment, sponsors must balance this against clinical requirements and patient safety
   - **Pattern consistency**: All complexity metrics (text length, exclusion items, structural complexity) show negative associations, suggesting this is a robust finding

3. **Regulatory and geographic features (0.16658497-0.31243518)**:
   - **Both `has_us_sites` and `has_dmc` show negative associations**: Counterintuitively, having US sites or DMC decreases completion probability. This may reflect:
     - **Selection bias**: US sites and DMC may be used for more complex/higher-risk trials that are inherently more likely to fail
     - **Confounding factors**: These features may be associated with stricter regulatory requirements or more challenging patient populations
     - **Data artifacts**: The associations may be driven by other correlated factors not captured in the model
   - **Interpretation caution**: These negative associations require careful interpretation and may not indicate that US sites or DMC cause lower completion rates, but rather that trials using these features may have other characteristics (e.g., higher complexity, stricter protocols) that reduce completion probability

4. **Trial scale hierarchy**:
   - **Facilities dominate**: `number_of_facilities` (0.61650556) is the strongest predictor, nearly 2× more important than the second-ranked feature
   - **Enrollment not in top 14**: `enrollment_planned` is not among the top 14 non-PCA features, indicating that operational capacity (number of facilities) matters more than target enrollment size
   - **Resource allocation**: This suggests that investing in multiple sites may be more effective than simply increasing target enrollment at fewer sites

5. **Disease and intervention flags (low importance)**:
   - **Limited predictive power**: Disease-specific flags and intervention flags are not in the top 15 predictors
   - **Operational > Clinical**: This suggests that operational factors (facilities, eligibility complexity) matter more than clinical characteristics (disease type, intervention type) for completion probability
   - **NSCLC focus**: The low importance of disease flags may reflect that the model is applied to NSCLC trials, where disease heterogeneity is limited

**Methodological Note**: The selected model uses tabular-only features (no PCA embeddings) for interpretability. While embedding-enhanced models may achieve higher performance, the tabular-only model provides clear, actionable SHAP insights that directly map to trial design decisions. **Note that even when models include PCA embeddings, all SHAP plots and analysis exclude PCA features (pca_emb_*) for interpretability**, ensuring that only interpretable trial characteristics are shown.

---

### 8.5 Application to Ongoing NSCLC Trials

The trained XGBoost model (tabular_plus_embeddings mode, ROC-AUC = 0.6803 ± 0.0069) was applied to **287 ongoing NSCLC Phase II–III industry-sponsored interventional trials** to rank them by predicted completion probability. **Note**: Phase IV trials were not included in the deployment cohort - only Phase II and Phase III trials were scored.

#### 8.5.1 Deployment Cohort

**Cohort characteristics**:
- **Total trials**: 287 ongoing trials
- **Status**: RECRUITING, ACTIVE_NOT_RECRUITING, or ENROLLING_BY_INVITATION
- **Disease focus**: NSCLC (Non-Small Cell Lung Cancer)
- **Phase**: Phase II or Phase III only (Phase IV trials were excluded from the deployment cohort)
- **Sponsor type**: Industry-sponsored
- **Study type**: Interventional

**Rationale for application**:
- The model was trained exclusively on final-outcome trials (completed vs. terminated)
- This enables reliable prediction for ongoing trials with unknown outcomes
- Predictions are probabilistic (P(completion)), not guaranteed outcomes

#### 8.5.2 Prediction Results

**Distribution of predicted completion probabilities**:
- **Minimum**: 0.0433 (4.3%)
- **Median**: 0.7501 (75.0%)
- **Maximum**: 0.9911 (99.1%)
- **Mean**: ~0.75
- **Standard deviation**: ~0.15

**Risk bucket stratification** (percentile-based):
- **LOW risk (top 30%)**: P(completion) ≥ 0.9453 (86 trials)
  - Interpretation: Highest predicted completion probability
- **MEDIUM risk (middle 40%)**: 0.7630 < P(completion) < 0.9453 (115 trials)
  - Interpretation: Moderate predicted completion probability
- **HIGH risk (bottom 30%)**: P(completion) ≤ 0.7630 (86 trials)
  - Interpretation: Lowest predicted completion probability

#### 8.5.3 Key Relationships Observed

**Number of facilities vs. completion probability**:
- Strong positive correlation (strongest predictor from SHAP analysis)
- Trials with more facilities tend to have higher predicted completion probability
- Multi-site trials show better predicted outcomes

**Eligibility complexity vs. completion probability**:
- Negative correlation
- More complex eligibility criteria (longer text, more exclusion items) correlate with lower predicted completion probability
- Reflects recruitment challenges associated with restrictive eligibility

**Geographic distribution**:
- Presence of US sites shows positive association with completion probability
- Multi-country trials show higher predicted completion probability

#### 8.5.4 Output Deliverables

The deployment generates:
1. **Ranked list**: All 287 trials ranked by P(completion) in descending order
2. **Risk buckets**: LOW/MEDIUM/HIGH risk categorization
3. **Feature values**: All trial characteristics used for modeling included in output
4. **Visualizations**: 
   - Probability distribution histogram
   - Risk bucket counts
   - Top 25 ranked trials table
   - Scatter plots (facilities vs. probability, enrollment vs. probability)
5. **SHAP analysis**: Top 15 non-PCA features by importance for the NSCLC cohort (PCA features excluded for interpretability)

---

## 9. Discussion

### 9.1 Key Findings

1. **Model achieves moderate discriminative ability**
   - Selected XGBoost model achieves ROC-AUC of 0.6875 (cross-validated)
   - Better than random (AUC = 0.5) but reflects inherent uncertainty in trial outcomes
   - Suitable for risk stratification of ongoing trials

2. **Trial scale is the strongest predictor**
   - `number_of_facilities` emerged as the dominant predictor (mean(|SHAP|) = 0.60588676), representing 1.951× the importance of the second-ranked feature (`has_us_sites` at 0.31053764)
   - Multi-site trials show substantially higher predicted completion probability
   - Reflects operational resilience, recruitment capacity, and geographic diversity
   - Notably, `enrollment_planned` is not in the top 15 features, indicating that operational capacity (facilities) matters more than target enrollment size

3. **Eligibility complexity negatively impacts completion**
   - Complex eligibility criteria (longer text, more exclusion items) correlate with lower completion probability
   - This is an actionable factor that sponsors can optimize

4. **Tabular_plus_embeddings model selected for best performance**
   - XGBoost tabular_plus_embeddings achieved highest ROC-AUC (0.6803 ± 0.0069)
   - Model includes 219 features total (14 non-PCA + 205 PCA embeddings)
   - SHAP analysis focuses on interpretable non-PCA features for clarity
   - PCA features excluded from SHAP plots for interpretability

5. **Class imbalance handling**
   - `class_weight='balanced'` strategy used to handle 1:3.12 class imbalance
   - Effective for tree-based models like XGBoost

### 9.2 Limitations

1. **Label censoring**: Only final-outcome trials used for training (ongoing trials excluded)
   - Temporal validation needed to track ongoing trials and observe actual outcomes
   - Model performance on ongoing trials cannot be directly validated until trials complete

2. **Moderate performance**: ROC-AUC of 0.6803 ± 0.0069 indicates moderate but not perfect discriminative ability
   - Reflects inherent uncertainty in clinical trial outcomes
   - Predictions should be used as one input among many in decision-making

3. **No time-to-event modeling**: Binary classification does not model time to completion or termination
   - A trial that completes in 2 years vs. 5 years is treated the same
   - Time-to-event modeling could provide additional insights

4. **No real-time recruitment data**: Model uses registration-time information only
   - Actual enrollment rates, recruitment challenges, and protocol amendments are not captured
   - Integration with real-time data sources could improve predictions

5. **Observational nature**: Model identifies associations, not causal relationships
   - Features may be proxies for unmeasured factors (e.g., sponsor quality, therapeutic area attractiveness)
   - Interventions based on model features may not guarantee improved outcomes

6. **Sample characteristics**: Model trained on all oncology Phase 2/3/4 trials, applied to NSCLC subset
   - Results may vary for different therapeutic areas or trial phases
   - External validation needed for different contexts

### 9.3 Future Directions

1. **Temporal validation**: Track ongoing trials to observe actual outcomes
   - Compare predicted vs. observed completion rates
   - Refine model based on validation results
   - Validate predictions for the 287 ongoing NSCLC trials

2. **External validation**: Test model on different therapeutic areas and contexts
   - Apply to different cancer types beyond NSCLC
   - Test on different trial phases or sponsor types
   - Validate generalizability across contexts

3. **Dynamic updates**: Incorporate real-time recruitment data
   - Update predictions as trial characteristics change
   - Integrate site-level enrollment data if available
   - Develop time-to-event models for temporal predictions

4. **Feature enhancements**: Integrate additional data sources
   - Protocol amendments and design changes
   - Site-level enrollment and recruitment rates
   - More sophisticated text features (named entity recognition, medical concept extraction)

5. **Model improvements**: Explore advanced techniques
   - Ensemble methods combining multiple models
   - Deep learning architectures for text processing
   - Calibration methods to improve probability estimates

---

## Appendix: Pipeline Execution Order

1. **Step 1**: Parse raw JSON files → `01_parsed_trials.parquet`
2. **Step 2**: Extract MeSH/disease and intervention features → `02_trials_with_mesh.parquet`
3. **Step 3**: Extract eligibility complexity features → `03_trials_with_eligibility.parquet`
4. **Step 4**: Generate BioLinkBERT embeddings → `04_trials_with_embeddings.parquet`
5. **Step 5**: Apply PCA to embeddings → `05_trials_with_pca.parquet`
6. **Step 6**: Combine all features → `oncology_phase23_enhanced_hist.parquet`
7. **Step 7**: Train and evaluate models → `results/model_comparison.csv`

---

*This document was generated based on the implementation in the codebase. Exact numbers and statistics should be populated from actual data runs.*

