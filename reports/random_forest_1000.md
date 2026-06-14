\## Random Forest Results



| Metric | Value   |

| ------ | ------- |

| MAE    | 0.23 kg |

| RMSE   | 0.30 kg |

| R²     | 0.816   |



\## Feature Importance



| Feature       | Importance |

| ------------- | ---------- |

| temperature\_c | 0.499      |

| co2\_ppm       | 0.311      |

| humidity\_pct  | 0.190      |



\## Feature Importance Interpretation



Temperature was the most influential feature for predicting mushroom yield, with an importance score of 0.499. Carbon dioxide concentration was the second most important feature with a score of 0.311, while humidity had the lowest importance score of 0.190. These results indicate that temperature has the strongest effect on yield prediction in the current dataset.



\## Model Comparison



| Model             | MAE   | RMSE  | R²    |

| ----------------- | ----- | ----- | ----- |

| Linear Regression | 0.206 | 0.253 | 0.870 |

| Random Forest     | 0.230 | 0.300 | 0.816 |



Linear Regression achieved lower prediction error and a higher R² score than Random Forest on the current dataset.



\## Complexity Discussion



Random Forest is a more complex model because it combines multiple decision trees and can capture nonlinear relationships among features. However, in this project, Linear Regression achieved better performance with an MAE of 0.206 kg, RMSE of 0.253 kg, and R² score of 0.870, compared to the Random Forest model which achieved an MAE of 0.230 kg, RMSE of 0.300 kg, and R² score of 0.816. Therefore, the additional complexity of Random Forest is not justified for the current dataset.



\## Generated Files



\* models/random\_forest\_1000.joblib

\* reports/figures/rf\_importance\_1000.png

\* src/train\_random\_forest\_1000.py



