MODELS OVERVIEW:

- The models explored were Linear Regresion, Random Forest and XGBoost.
- The combination between dataset and model that gave the best performance was the XGBoost with the new features, the HDBSCAN outlier isolation and the feature selection.
- The model presented the following performance:
  Train dataset
  RMSE: 138.90
  MAE: 92.72
  Validation Dataset
  RMSE: 308.21
  MAE: 198.64
- Compared to the base Linear regression model:
  Train dataset
  RMSE: 425.70
  MAE: 257.49
  Validation dataset
  RMSE: 459.94
  MAE: 269.99
- The XGBoost model had a 27% MAE reduction compared to the Liner Regression
- The model has problems with the houses with larger prices, higher that 2000millions pesos
- The XGBoost model was optimized using a GridSearch module and the model was succesfully optimized, showing better results and reducing the overfitting:
  Train dataset
  RMSE: 232.33
  MAE: 148.69
  Validation dataset
  RMSE: 301.02
  MAE: 192.76
- The MAE was and extra 30% compared to the last model version.
- The model had a similar performace with the test dataset
  Test dataset
  RMSE: 222.90
  MAE: 148.37
- The mean error percentage after optimizartion for the test dataset was 19%.
