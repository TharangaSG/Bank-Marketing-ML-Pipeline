# Bank Marketing ML Pipeline

A production-grade machine learning pipeline to predict if a client will subscribe to a term deposit based on the Bank Marketing dataset.

## Features

- **Modular Pipeline Architecture**: Separated pipelines for data processing, model training, and batch inference.
- **Robust Data Processing**: 
  - Missing value imputation
  - Outlier detection (IQR method with capping)
  - Feature encoding (one-hot, label encoding, custom ordinal mappings)
  - Feature scaling
  - Stratified data splitting
- **Model Training & Evaluation**: 
  - Support for Random Forest, Gradient Boosting, XGBoost, and Logistic Regression
  - Automated cross-validation and hyperparameter handling
  - Comprehensive metric tracking (Accuracy, Precision, Recall, F1-score, ROC AUC)
- **Experiment Tracking**: Full integration with MLflow for tracking parameters, metrics, and model artifacts.
- **Configuration Driven**: Centralized `config.yaml` to manage all pipeline parameters without modifying code.

## Tech Stack

- **Python**: >= 3.9 managed by `uv`
- **Core ML**: `scikit-learn`, `xgboost`, `pandas`, `numpy`
- **Experiment Tracking**: `mlflow`
- **Configuration**: `pyyaml`

## Project Structure

```text
├── artifacts/           # Stored models, data splits, and predictions
├── data/                # Raw and processed datasets
├── mlruns/              # MLflow artifacts
├── notebooks/           # Jupyter notebooks for EDA and experimentation
├── pipelines/           # Execution scripts for ML pipelines
│   ├── data_pipeline.py
│   ├── inference_pipeline.py
│   └── training_pipeline.py
├── src/                 # Core modular source code
│   ├── data_ingestion.py
│   ├── data_splitter.py
│   ├── feature_encoding.py
│   ├── feature_scaling.py
│   ├── handle_missing_values.py
│   ├── mlflow_utils.py
│   ├── model_building.py
│   ├── model_evaluation.py
│   ├── model_inference.py
│   ├── model_training.py
│   └── outlier_detection.py
├── config.yaml          # Central pipeline configuration
├── Makefile             # CLI commands for running pipelines
├── pyproject.toml       # Python package dependencies
└── README.md
```

## Getting Started

### 1. Prerequisites
Ensure you have `uv` installed for dependency management. If not, install it via pip:
```bash
pip install uv
```

### 2. Installation
Install all project dependencies using the provided Makefile command:
```bash
make install
```

### 3. Usage

The project utilizes a `Makefile` to simplify running different stages of the ML lifecycle.

**Run Data Processing Pipeline**
Processes the raw data, handles missing values, outliers, encoding, scaling, and splits into train/test sets.
```bash
make data
```

**Run Model Training Pipeline**
Trains the machine learning model using the processed data and logs the experiment to MLflow.
```bash
make train
```

**Run End-to-End Pipeline**
Runs both data processing and model training sequentially.
```bash
make all
```

**Run Batch Inference Pipeline**
Runs inference on unseen data using the registered model.
```bash
make inference
```

**Start MLflow UI**
Launch the MLflow tracking server to view experiments, metrics, and models. The UI will be available at http://localhost:5000.
```bash
make mlflow
```

**Clean Artifacts**
Removes all generated artifacts, caches, and the MLflow database.
```bash
make clean
```

## Configuration
All aspects of the pipeline are controlled via `config.yaml`. You can modify:
- Data paths and column types
- Outlier detection thresholds
- Feature binning & encoding rules
- Model hyperparameters
- MLflow tracking details
