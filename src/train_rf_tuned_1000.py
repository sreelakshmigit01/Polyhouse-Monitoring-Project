import pandas as pd
import numpy as np
import json
import time
import joblib

from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==================================================
# START TIMER
# ==================================================

start_time = time.time()

# ==================================================
# LOAD PROCESSED DATA (YOUR PROJECT STRUCTURE)
# ==================================================

X_train = pd.read_parquet("data/processed/X_train_1000.parquet")
X_test  = pd.read_parquet("data/processed/X_test_1000.parquet")

y_train = pd.read_parquet("data/processed/y_train_1000.parquet").squeeze()
y_test  = pd.read_parquet("data/processed/y_test_1000.parquet").squeeze()

print("Train Shape:", X_train.shape)
print("Test Shape :", X_test.shape)

# ==================================================
# TIME SERIES CROSS VALIDATION
# ==================================================

tscv = TimeSeriesSplit(n_splits=3)

# ==================================================
# PARAMETER GRID (SMALL + CONTROLLED)
# ==================================================

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 8, 16],
    "min_samples_leaf": [1, 3, 5]
}

# Grid size: 3×3×3 = 27 models × CV folds

# ==================================================
# BASE MODEL
# ==================================================

rf = RandomForestRegressor(
    random_state=42,
    n_jobs=-1
)

# ==================================================
# GRID SEARCH CV
# ==================================================

search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=tscv,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
    refit=True
)

search.fit(X_train, y_train)

# ==================================================
# BEST MODEL INFO
# ==================================================

print("\n===== BEST PARAMETERS =====")
print(search.best_params_)

print("\nBest CV MAE:", round(-search.best_score_, 4))

best_model = search.best_estimator_

# ==================================================
# SAVE BEST PARAMETERS
# ==================================================

with open("models/rf_best_params_1000.json", "w") as f:
    json.dump(search.best_params_, f, indent=2)

# ==================================================
# SAVE CV RESULTS (MENTOR REQUIREMENT)
# ==================================================

cv_results = pd.DataFrame(search.cv_results_)
cv_results.to_csv("models/rf_gridsearch_results_1000.csv", index=False)

# ==================================================
# TEST SET EVALUATION (ONLY ONCE)
# ==================================================

y_pred = best_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n===== TEST RESULTS =====")
print("MAE :", round(mae, 4))
print("RMSE:", round(rmse, 4))
print("R²  :", round(r2, 4))

# ==================================================
# SAVE FINAL MODEL
# ==================================================

joblib.dump(best_model, "models/random_forest_tuned_1000.joblib")

# ==================================================
# RUNTIME
# ==================================================

runtime = time.time() - start_time
print("\nRuntime:", round(runtime, 2), "seconds")