# Breast Cancer Prediction using MRI Image Features

## Overview
This repository contains a machine learning project aimed at predicting breast cancer characteristics, specifically the **Estrogen Receptor (ER) status**, using features extracted from breast MRI images.  
The dataset is based on the work by Saha et al., which includes pre-extracted imaging features and clinical data for breast cancer cases.  

The project involves:
- Data preprocessing (focused on image-derived features)
- Exploratory data analysis (EDA)
- Feature engineering
- Model training and evaluation

**Goal:**  
Develop predictive models for ER status (binary classification: ER-positive vs. ER-negative), which is crucial for personalized treatment in breast cancer patients.  

Key challenges identified:
- Class imbalance
- High feature correlation
- Non-linear separability (suggesting non-linear models for better performance)

---

## Key Components

- **Data Preprocessing:** Missing value handling, outlier removal, and feature engineering tailored to MRI-derived features (e.g., texture, morphology, enhancement rates).
- **EDA:** Statistical and visual analysis to understand feature distributions and relationships.
- **Modeling:** Training and evaluating classification models using modular Python scripts.
- **Reports:** Automatically generated PDFs and Excel files summarizing EDA, model results, and metrics.

The project is structured for **reproducibility** with:
- Jupyter notebooks for experimentation
- Python modules for core functionality
- Unit tests
- Separate directories for datasets, models, and reports

---

## Dataset
Saha and colleagues [1] collected a large set of breast MRI cases that
were used to detect breast cancer. The dataset also contains detailed
information about the characteristic of the patient and subtype of tumors
that were eventually diagnosed. This information can be used to determine
the best treatment.
Basic image processing was performed to extract features from the MRI
cases which were used to predict the molecular tumor subtypes from MRI.
In this assignment, we will try to predict the estrogen receptor (ER) status
from the MRI image features that were provided. The datasets can be downloaded from the following website: https://www.cancerimagingarchive.net/collection/duke-breast-cancer-mri.  
The dataset consists of features derived from breast MRI scans, including:
- Tumor enhancement textures  
- Size/morphology metrics  
- Enhancement variations  

**Stats:**  
- 922 samples  
- Target: ER status (74.4% ER-positive, 25.6% ER-negative)

### Source
- Derived from Saha et al. dataset (public medical imaging dataset; see raw Excel files for details).
- **Raw data files (Excel format) in `dataset/raw/`:**
  - `Clinical_and_Other_Features.xlsx` → Clinical & non-imaging data
  - `Imaging_Features.xlsx` → Pre-extracted MRI features (e.g., `WashinRate_map_Cluster_Prominence_tumor`, `Volume_cu_mm_Tumor`)
  - `test_set.xlsx` → Dedicated test set

### Processed Data
- Located in `dataset/processed/`
- Includes different preprocessing/feature-selection approaches:
  - `train_processed_full.xlsx`, `test_processed_full.xlsx` → Baseline datasets
  - `train_processed_approach_2.xlsx`, `test_processed_approach_2.xlsx` → After correlation filtering & outlier removal
  - `train_processed_approach_3.xlsx`, `test_processed_approach_3.xlsx` → After statistical (KS test) feature selection
  - `approach_4/` → Scaling + UMAP dimensionality reduction  

**Note:** Raw data may contain missing values (up to 4.77%) and outliers, handled in preprocessing.

---

## Usage

The project workflow is primarily driven by Jupyter notebooks in the `notebooks/` directory. Run them in sequence for end-to-end execution.

---

## Step-by-Step Workflow

### 1. Data Preprocessing
- **Notebook:** `notebooks/1. data_preprocessing_image_features.ipynb`  
- **Purpose:**  
  - Load raw Excel files  
  - Merge clinical and imaging features  
  - Handle missing values (using **MICE imputation** to avoid data leakage)  
  - Split into train/test sets  
  - Apply preprocessing specific to image features (e.g., normalization of enhancement rates, texture metrics)  
- **Output:** Saves processed datasets to `dataset/processed/`

---

### 2. Exploratory Data Analysis (EDA)
- **Notebook:** `notebooks/2. EDA.ipynb`  
- **Purpose:** Perform descriptive statistics, visualize distributions, and identify patterns.  

- **Key Analyses:**  
  - **Missing values:** Analyzed percentages and imputed with MICE.  
  - **Outliers:** Detected using Isolation Forest (more frequent in ER-negative class).  
  - **Correlations:** Found high multicollinearity (341+ features with >0.8 correlation).  
  - **Target distribution:** Moderate imbalance.  

- **Feature Selection Approaches:**  
  - **Approach 1:** Baseline (no processing)  
  - **Approach 2:** Remove correlated features (>0.8) and outliers  
  - **Approach 3:** KS test for significant features (dropped 380 non-significant)  
  - **Approach 4:** Robust scaling + UMAP (100 components, trustworthiness ~0.86)  

- **Visualizations:**  
  - Box plots for top features  
  - KDE plots  
  - UMAP/t-SNE projections (showing overlap, no linear separability)  

- **Output:** Generates `reports/1. EDA.pdf` with visualizations and findings.

---

### 3. Modeling
- **Notebook:** `notebooks/3. modeling.ipynb`  
- **Purpose:** Train and evaluate models on processed data.  

- **Modular Scripts in `modeling/`:**  
  - `data_preprocessing.py`: Core preprocessing functions  
  - `feature_engineering.py`: Implements approaches 2–4 (correlation filtering, KS test, UMAP)  
  - `manufacturer_analysis.py`: Analyzes features by MRI manufacturer (if applicable)  
  - `model_training.py`: Trains classifiers (logistic regression, random forest, XGBoost for non-linear handling)  
  - `visualisation.py`: Generates plots for model results  

- **Evaluation:** Metrics include accuracy, precision, recall, F1-score, ROC-AUC (stored in `reports/model_metrics.xlsx`).  

- **Output:**  
  - Saves trained models to `models/saved_models/`  
  - Logs to `models/logs/`  
  - Generates `reports/2. Model results.pdf`

---

## Testing

- **test_data_preprocessing.py**: Tests preprocessing functions.  
- **test_feature_engineering.py**: Tests feature selection.  
- **test_model_training.py**: Tests model fitting and predictions.  

---

## Reports

View generated reports in `reports/`:

- **EDA.pdf**: Detailed EDA visuals and stats.  
- **Model results.pdf**: Model performance summaries.  
- **model_metrics.xlsx**: Tabular metrics across approaches.  
- **Subdirectories (`approach_1/` to `approach_4/`)**: Approach-specific reports.  

## Project Structure

```plaintext
breast_cancer/
├── .gitignore              # Git ignore file
├── .python-version         # Specified Python version
├── .vscode/                # VS Code settings
├── LICENSE                 # Project license (MIT assumed; check file for details)
├── README.md               
├── dataset/                # Data files
│   ├── intermediate/       # Intermediate processing files
│   ├── processed/          # Processed train/test sets
│   └── raw/                # Raw Excel datasets
├── modeling/               # Core Python modules
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── manufacturer_analysis.py
│   ├── model_training.py
│   └── visualisation.py
├── models/                 # Saved models and logs
│   ├── logs/
│   └── saved_models/
├── notebooks/              # Jupyter notebooks
│   ├── 1. data_preprocessing_image_features.ipynb
│   ├── 2. EDA.ipynb
│   └── 3. modeling.ipynb
├── pyproject.toml          # Project config (dependencies, tools)
├── reports/                # Generated reports and metrics
│   ├── 1. EDA.pdf
│   ├── 2. Model results.pdf
│   ├── model_metrics.xlsx
│   ├── approach_1/
│   ├── approach_2/
│   ├── approach_3/
│   └── approach_4/
├── requirements.txt        
├── tests/                  # Unit tests
│   ├── test_data_preprocessing.py
│   ├── test_feature_engineering.py
│   └── test_model_training.py
└── uv.lock                 # UV dependency lock file
