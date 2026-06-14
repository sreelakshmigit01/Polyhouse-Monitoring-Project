# Cross Validation Results

## Linear Regression
CV MAE: 0.212 +/- 0.011

## Random Forest
CV MAE: 0.252 +/- 0.009

## Test Set MAE
Linear Regression: 0.206
Random Forest: 0.235

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
