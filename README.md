# Polyhouse Monitoring Project

## Problem Statement

This project focuses on monitoring polyhouse environmental conditions such as temperature, humidity, CO₂ levels, and crop yield. The objective is to build a reproducible data science workflow for collecting, processing, analyzing, and modeling sensor data for agritech applications.

## Folder Structure

project/
│
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── docs/
├── reports/
│   └── figures/
├── src/
├── models/
├── smoke_test.py
├── requirements.txt
├── .gitignore
└── README.md

## Environment Setup

### 1. Create a Virtual Environment

python -m venv myenv

### 2. Activate the Environment

myenv\Scripts\activate

### 3. Install Dependencies

pip install -r requirements.txt

### 4. Run the Smoke Test

python smoke_test.py

## Dependencies

The project dependencies are listed in requirements.txt.

## Version Control

Git is used for version control. The repository is hosted on GitHub and all project changes are tracked through commits.

# Dataset Columns

### timestamp

Date and time when sensor reading was recorded.

### temperature_c

Polyhouse temperature in degrees Celsius.

### humidity_pct

Relative humidity percentage.

### co2_ppm

Carbon dioxide concentration in parts per million.

### yield_kg

Harvested mushroom yield in kilograms.

# Data Cleaning

Cleaning operations performed:

* Missing value handling
* Data validation checks
* Yield validation
* Duplicate removal
* Timestamp parsing
* Data type correction

## Dataset Summary

### Raw Dataset

Rows: 1005

### Cleaned Dataset

Rows: 979

### Date Range

Start: 2024-01-01 00:00:00

End: 2024-02-11 15:00:00

## Output Files

data/interim/01_loaded_1000.parquet

data/interim/02_cleaned_1000.parquet

docs/cleaning_log_1000.md

reports/data_quality_1000.md

# Exploratory Data Analysis (EDA)

EDA was performed to analyze:

* Feature distributions
* Correlation between variables
* Yield relationships
* Potential anomalies

## Generated Visualizations

reports/figures/corr_heatmap_1000.png

reports/figures/scatter_yield_1000.png

## EDA Notes

reports/figures/ were generated using Matplotlib and Pandas to understand relationships among environmental variables and crop yield.

# Feature Engineering

## Features Used

1. temperature_c
2. humidity_pct
3. co2_ppm

## Target

yield_kg

## Scaler

MinMaxScaler

## Outputs

data/processed/features_1000.parquet

models/minmax_scaler_1000.joblib

# Train/Test Split

## Split Method

A chronological 80/20 train-test split was used to preserve temporal order and prevent future information from influencing model training.

## Features

* temperature_c
* humidity_pct
* co2_ppm

## Target

* yield_kg

## Dataset Size

* Total Rows: 979
* Training Rows: 783
* Testing Rows: 196

## Date Range

* Training Period: Earliest records from cleaned dataset
* Testing Period: Most recent 20% of records

## Scaling

MinMaxScaler was fitted on the training data only and then used to transform both training and testing datasets. This prevents data leakage from the test set.

## Saved Scaler

models/minmax_scaler_train_1000.joblib

## Saved Split Artifacts

data/processed/X_train_1000.parquet

data/processed/X_test_1000.parquet

data/processed/y_train_1000.parquet

data/processed/y_test_1000.parquet

# Linear Regression Baseline Model

A Linear Regression model was trained as an interpretable baseline model for yield prediction.

## Model Features

* temperature_c
* humidity_pct
* co2_ppm

## Saved Model

models/linear_regression_1000.joblib

## Model Performance

| Metric | Value    |
| ------ | -------- |
| MAE    | 0.206 kg |
| RMSE   | 0.253 kg |
| R²     | 0.870    |

## Coefficients

| Feature       | Coefficient |
| ------------- | ----------- |
| temperature_c | 2.8370      |
| humidity_pct  | 1.7381      |
| co2_ppm       | -2.3961     |

## Interpretation

* Temperature shows a positive influence on yield.
* Humidity shows a positive influence on yield.
* CO₂ shows a negative influence on yield in the scaled feature space.
* Linear Regression provides a strong and interpretable baseline model.

## Generated Reports

reports/metrics_linear_1000.json

reports/coefficient_interpretation_1000.md

reports/baseline_evaluation_1000.md

# Model Diagnostics

Residual analysis was performed to evaluate prediction errors.

## Generated Figure

reports/figures/residuals_linear_1000.png

## Findings

* Residuals are centered around zero.
* No major nonlinear pattern observed.
* No significant heteroscedasticity detected.
* A few isolated larger residuals were present.

## Diagnostic Report

reports/linear_diagnostics_1000.md

## Recommendation

Linear Regression performs well as a baseline model. More advanced models such as Random Forest Regression can be explored to capture nonlinear relationships and potentially improve predictive performance.

# Models Directory

models/

├── linear_regression_1000.joblib

├── minmax_scaler_1000.joblib

└── minmax_scaler_train_1000.joblib

# Reports Directory

reports/

├── data_quality_1000.md

├── baseline_evaluation_1000.md

├── coefficient_interpretation_1000.md

├── linear_diagnostics_1000.md

├── metrics_linear_1000.json

└── figures/

    ├── corr_heatmap_1000.png

    ├── scatter_yield_1000.png

    └── residuals_linear_1000.png

# Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-Learn
* Joblib
* PyArrow
* Git
* GitHub

# Future Scope

* Random Forest Regression
* Hyperparameter tuning
* Streamlit dashboard development
* Real-time sensor monitoring
* Time-series forecasting

## Random Forest Regressor

A Random Forest Regressor was trained using the processed mushroom yield dataset to model nonlinear relationships between environmental variables and yield.

### Features Used

* temperature_c
* humidity_pct
* co2_ppm

### Model Configuration

* n_estimators = 100
* random_state = 42
* n_jobs = -1

### Test Results

| Metric | Value   |
| ------ | ------- |
| MAE    | 0.23 kg |
| RMSE   | 0.30 kg |
| R²     | 0.816   |

### Feature Importance

| Feature       | Importance |
| ------------- | ---------- |
| temperature_c | 0.499      |
| co2_ppm       | 0.311      |
| humidity_pct  | 0.190      |

### Generated Artifacts

* models/random_forest_1000.joblib
* reports/figures/rf_importance_1000.png

# Cross Validation Results

## TimeSeriesSplit Methodology

TimeSeriesSplit with 5 folds was used to evaluate model stability while preserving temporal order. Unlike random K-Fold validation, TimeSeriesSplit ensures that future observations are never used to predict past observations, making it suitable for time-series sensor data.

## Cross Validation Scores

| Model             | Mean CV MAE | Standard Deviation |
| ----------------- | ----------- | ------------------ |
| Linear Regression | 0.212       | 0.011              |
| Random Forest     | 0.252       | 0.009              |

## Test Set Comparison

| Model             | Test MAE |
| ----------------- | -------- |
| Linear Regression | 0.206    |
| Random Forest     | 0.235    |

## Overfitting Analysis

The Linear Regression model achieved a mean cross-validation MAE of 0.212 and a test MAE of 0.206. The small difference between these values indicates good generalization and minimal overfitting.

The Random Forest model achieved a mean cross-validation MAE of 0.252 and a test MAE of 0.235. Although the test MAE is slightly lower than the cross-validation MAE, the difference is small, suggesting stable performance across folds and no significant overfitting.

## Variance Interpretation

The standard deviation of the cross-validation scores was low for both models (0.011 for Linear Regression and 0.009 for Random Forest). This indicates that model performance remained consistent across different time-based validation folds and that the dataset is reasonably stable.

## Model Comparison

Linear Regression achieved a lower cross-validation MAE and a lower test MAE than Random Forest. Therefore, Linear Regression demonstrated better predictive accuracy and generalization performance on the current mushroom yield dataset.

## Conclusion

TimeSeriesSplit cross-validation confirmed that both models generalize reasonably well to unseen data. Linear Regression outperformed Random Forest in both cross-validation and test-set evaluation, making it the preferred model for this dataset.
