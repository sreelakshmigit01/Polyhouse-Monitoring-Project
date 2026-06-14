import pandas as pd
import numpy as np

from sklearn.model_selection import (
    TimeSeriesSplit,
    cross_val_score
)

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    r2_score
)

# ==========================================
# LOAD TRAIN AND TEST DATA
# ==========================================

X_train = pd.read_parquet(
    "data/processed/X_train_1000.parquet"
)

X_test = pd.read_parquet(
    "data/processed/X_test_1000.parquet"
)

y_train = pd.read_parquet(
    "data/processed/y_train_1000.parquet"
).squeeze()

y_test = pd.read_parquet(
    "data/processed/y_test_1000.parquet"
).squeeze()

print("Train Shape:", X_train.shape)
print("Test Shape :", X_test.shape)

# ==========================================
# TIMEsERIES SPLIT
# ==========================================

tscv = TimeSeriesSplit(
    n_splits=5
)

# ==========================================
# MODELS
# ==========================================

lin = LinearRegression()

rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

# ==========================================
# CROSS VALIDATION
# ==========================================

lin_scores = cross_val_score(
    lin,
    X_train,
    y_train,
    cv=tscv,
    scoring="neg_mean_absolute_error"
)

rf_scores = cross_val_score(
    rf,
    X_train,
    y_train,
    cv=tscv,
    scoring="neg_mean_absolute_error"
)

lin_mae = -lin_scores
rf_mae = -rf_scores

# ==========================================
# PRINT CV RESULTS
# ==========================================

print("\n===== CROSS VALIDATION RESULTS =====")

print(
    f"\nLinear Regression CV MAE: "
    f"{lin_mae.mean():.3f}"
    f" +/- {lin_mae.std():.3f}"
)

print(
    f"Random Forest CV MAE: "
    f"{rf_mae.mean():.3f}"
    f" +/- {rf_mae.std():.3f}"
)

# ==========================================
# TRAIN FULL MODELS
# ==========================================

lin.fit(X_train, y_train)
rf.fit(X_train, y_train)

# ==========================================
# TEST SET METRICS
# ==========================================

lin_pred = lin.predict(X_test)
rf_pred = rf.predict(X_test)

lin_test_mae = mean_absolute_error(
    y_test,
    lin_pred
)

rf_test_mae = mean_absolute_error(
    y_test,
    rf_pred
)

print("\n===== TEST SET RESULTS =====")

print(
    f"\nLinear Regression Test MAE: "
    f"{lin_test_mae:.3f}"
)

print(
    f"Random Forest Test MAE: "
    f"{rf_test_mae:.3f}"
)

# ==========================================
# OVERFITTING CHECK
# ==========================================

print("\n===== OVERFITTING ANALYSIS =====")

print(
    f"\nLinear Regression:"
)

print(
    f"CV MAE   : {lin_mae.mean():.3f}"
)

print(
    f"Test MAE : {lin_test_mae:.3f}"
)

print(
    f"\nRandom Forest:"
)

print(
    f"CV MAE   : {rf_mae.mean():.3f}"
)

print(
    f"Test MAE : {rf_test_mae:.3f}"
)

# ==========================================
# SAVE RESULTS
# ==========================================

with open(
    "reports/cv_results.md",
    "w",
    encoding="utf-8"
) as f:

    f.write("# Cross Validation Results\n\n")

    f.write("## Linear Regression\n")
    f.write(
        f"CV MAE: {lin_mae.mean():.3f} "
        f"+/- {lin_mae.std():.3f}\n\n"
    )

    f.write("## Random Forest\n")
    f.write(
        f"CV MAE: {rf_mae.mean():.3f} "
        f"+/- {rf_mae.std():.3f}\n\n"
    )

    f.write("## Test Set MAE\n")
    f.write(
        f"Linear Regression: "
        f"{lin_test_mae:.3f}\n"
    )

    f.write(
        f"Random Forest: "
        f"{rf_test_mae:.3f}\n"
    )

print(
    "\nResults saved to reports/cv_results_1000.md"
)