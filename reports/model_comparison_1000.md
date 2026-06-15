1\. Objective



The objective of this phase is to compare multiple regression models trained on polyhouse sensor data and identify the most suitable model for deployment based on predictive performance and generalization ability.



The evaluation focuses on:



Prediction accuracy

Error consistency

Model generalization on unseen test data

Practical interpretability for agricultural use



2\. Models Evaluated



Three regression models were trained and evaluated on the same untouched test dataset:



Linear Regression (baseline interpretable model)

Random Forest Regressor (default configuration)

Random Forest Regressor (hyperparameter tuned using GridSearchCV)



All models were trained on identical training data and evaluated on the same test split to ensure fair comparison.



3\. Evaluation Metrics



The following metrics were used:



MAE (Mean Absolute Error): Measures average absolute prediction error (primary metric)

RMSE (Root Mean Squared Error): Penalizes larger errors more heavily

R² Score: Measures variance explained by the model



MAE was prioritized as it directly reflects prediction error in real-world units (kg yield).





4\. Model Comparison Results



Model	                 MAE	RMSE	R²

Linear Regression	0.2056	0.2526	0.8703

Random Forest (Default)	0.2348	0.3011	0.8156

Random Forest (Tuned)	0.2365	0.3020	0.8146



5\. Observations



5.1 Linear Regression Performance

Achieved the lowest MAE (0.2056)

Highest R² score (0.8703)

Indicates strong linear relationship between sensor variables and yield

Demonstrates excellent generalization on unseen data



5.2 Random Forest (Default)



Higher error compared to linear model

Moderate R² (0.8156)

Indicates some non-linearity exists but is not strongly dominant



5.3 Random Forest (Tuned)



Slight improvement in hyperparameters did not improve performance

Similar or slightly worse than default RF

Suggests model may be overfitting or dataset is not highly nonlinear



6\. Champion Model Selection



Selected Model: Linear Regression

Selection Criteria:



The final model was selected based on lowest MAE on the test set, as MAE directly represents average prediction error in real-world units (kg yield).



Why Linear Regression was chosen:

Lowest prediction error across all models

Highest R² score indicating best fit

Stable and consistent performance on unseen data

Simple and highly interpretable for agritech stakeholders

Faster inference suitable for real-time applications

7\. Business Interpretation



The selected model achieves an average prediction error of approximately:



±0.21 kg yield per prediction



This level of accuracy is practically meaningful for:



Harvest planning and scheduling

Market demand estimation

Resource allocation in polyhouse farming

Yield forecasting for supply chain planning



8\. Model Behavior Insights



The linear model suggests that environmental factors (temperature, humidity, CO₂) have a largely linear relationship with crop yield in this dataset

Random Forest did not significantly outperform linear regression, indicating limited complex non-linearity in the dataset

The simplicity of the dataset favors interpretable models over complex ensemble methods



9\. Limitations



Sensor data is limited to controlled polyhouse conditions and may not generalize to open-field environments

Seasonal effects are not explicitly modeled

External agricultural factors (soil nutrients, pests, irrigation variability) are not included

Assumes consistent sensor calibration and stable data quality



10\. Conclusion



A comparative analysis of three regression models was conducted using consistent evaluation metrics. Linear Regression emerged as the best-performing model due to its superior accuracy, stability, and interpretability.



Despite the availability of more complex models like Random Forest, the simpler linear approach proved more effective for this dataset.



Final Outcome:

Champion Model: Linear Regression

Best MAE: 0.2056

Best R²: 0.8703



This model is recommended for deployment in a decision-support system for polyhouse yield prediction.

