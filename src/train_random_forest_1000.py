import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==========================================
# LOAD TRAIN / TEST DATA
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

print("Train shape:", X_train.shape)
print("Test shape :", X_test.shape)

# ==========================================
# TRAIN RANDOM FOREST
# ==========================================

rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

# ==========================================
# PREDICTIONS
# ==========================================

pred = rf.predict(X_test)

# ==========================================
# METRICS
# ==========================================

mae = mean_absolute_error(y_test, pred)

rmse = mean_squared_error(
    y_test,
    pred
) ** 0.5

r2 = r2_score(
    y_test,
    pred
)

print("\n===== RANDOM FOREST RESULTS =====")

print(f"MAE  : {mae:.2f} kg")
print(f"RMSE : {rmse:.2f} kg")
print(f"R²   : {r2:.3f}")

# ==========================================
# FEATURE IMPORTANCE
# ==========================================

importance = pd.Series(
    rf.feature_importances_,
    index=X_train.columns
)

print("\nFeature Importances")
print(
    importance.sort_values(
        ascending=False
    )
)

# ==========================================
# FEATURE IMPORTANCE PLOT
# ==========================================

plt.figure(figsize=(8, 5))

importance.sort_values().plot(
    kind="barh"
)

plt.xlabel("Importance")
plt.title(
    "Random Forest Feature Importance"
)

plt.tight_layout()

plt.savefig(
    "reports/figures/rf_importance_1000.png",
    dpi=150
)

# ==========================================
# SAVE MODEL
# ==========================================

joblib.dump(
    rf,
    "models/random_forest_1000.joblib"
)

print(
    "\nModel saved successfully."
)
plt.show()

