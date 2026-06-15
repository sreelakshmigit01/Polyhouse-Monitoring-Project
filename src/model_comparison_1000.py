import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import json

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==================================================
# LOAD TEST DATA (YOUR EXACT FILES)
# ==================================================

X_test = pd.read_parquet("data/processed/X_test_1000.parquet")
y_test = pd.read_parquet("data/processed/y_test_1000.parquet").squeeze()

# ==================================================
# LOAD MODELS (YOUR EXACT FILE NAMES)
# ==================================================

linear_model = joblib.load("models/linear_regression_1000.joblib")
rf_default = joblib.load("models/random_forest_1000.joblib")
rf_tuned = joblib.load("models/random_forest_tuned_1000.pkl")

# ==================================================
# EVALUATION FUNCTION
# ==================================================

def evaluate(model, X_test, y_test):
    pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)

    return mae, rmse, r2, pred

# ==================================================
# MODEL EVALUATION
# ==================================================

lin_mae, lin_rmse, lin_r2, lin_pred = evaluate(linear_model, X_test, y_test)
rf_mae, rf_rmse, rf_r2, rf_pred = evaluate(rf_default, X_test, y_test)
tuned_mae, tuned_rmse, tuned_r2, tuned_pred = evaluate(rf_tuned, X_test, y_test)

# ==================================================
# COMPARISON TABLE
# ==================================================

results = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Random Forest (Default)",
        "Random Forest (Tuned)"
    ],
    "MAE": [lin_mae, rf_mae, tuned_mae],
    "RMSE": [lin_rmse, rf_rmse, tuned_rmse],
    "R2": [lin_r2, rf_r2, tuned_r2]
})

print("\n===== MODEL COMPARISON =====")
print(results.to_markdown(index=False))

# Save table
results.to_csv("reports/model_comparison_1000.csv", index=False)

# ==================================================
# CHAMPION MODEL SELECTION (BASED ON MAE)
# ==================================================

best_idx = results["MAE"].idxmin()
champion = results.loc[best_idx, "Model"]

print("\n===== CHAMPION MODEL =====")
print(champion)

with open("reports/champion_model_1000.json", "w") as f:
    json.dump({"champion_model": champion}, f, indent=2)

# ==================================================
# SELECT BEST PREDICTIONS
# ==================================================

pred_map = {
    "Linear Regression": lin_pred,
    "Random Forest (Default)": rf_pred,
    "Random Forest (Tuned)": tuned_pred
}

best_pred = pred_map[champion]

# ==================================================
# PREDICTED VS ACTUAL PLOT
# ==================================================

plt.figure(figsize=(6, 6))
plt.scatter(y_test, best_pred, alpha=0.6)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    "r--"
)

plt.xlabel("Actual Yield (kg)")
plt.ylabel("Predicted Yield (kg)")
plt.title(f"Predicted vs Actual - {champion}")

plt.savefig("reports/pred_vs_actual_1000.png", dpi=150)
plt.close()