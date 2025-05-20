import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import joblib


from features.f_01_features_exploration import ran
from features.f_01_features_exploration import features_target_split
from features.f_01_features_exploration import df_hdb_newf_train
from features.f_01_features_exploration import df_hdb_newf_valid
from features.f_01_features_exploration import df_hdb_newf_test
from m_01_model_exploration import model_evaluation

# Train and validation datasets will be evaluated at the same time in the cross validation
df_opt = pd.concat([df_hdb_newf_train, df_hdb_newf_valid], axis=0)
f_opt_nf, t_opt_nf = features_target_split(df_opt)


model_xgb_selected = xgb.XGBRegressor(eval_metric='rmse',
                            random_state=ran)
params = {
    'learning_rate'     : [0.01, 0.04, 0.08], 
    'max_depth'         : [3, 6, 12], 
    'subdf_hdb_newf_test'         : [0.5, 0.9],
    'coldf_hdb_newf_test_bytree'  : [0.5, 0.9],
    'n_estimators'      : [50, 70, 100],
    'alpha'             : [0.2, 1],
}
grid_search = GridSearchCV(estimator=model_xgb_selected,
                           param_grid=params,
                           verbose=3,
                           scoring= 'neg_mean_absolute_error', 
                           cv=3,)


grid_search.fit(f_opt_nf, t_opt_nf )


best_xgb = grid_search.best_estimator_


grid_search.best_params_


model_evaluation(best_xgb, df_hdb_newf_train, df_hdb_newf_valid, 'XGBoost - Cluster HDBSCAN + New features')


# Test dataset


model_evaluation(best_xgb, df_hdb_newf_train, df_hdb_newf_test, 'XGBoost - Cluster HDBSCAN + New features', True, 'Test')

joblib.dump(best_xgb, 'models/models/predictor_xgb_model.joblib')


# New data prediction with residuals

f_test, t_test = features_target_split(df_hdb_newf_test)
predictions_test = best_xgb.predict(f_test)

data_prediction_df_hdb_newf_test = {
    'real'          : np.round(df_hdb_newf_test['fixed_price'],1),
    'prediction'    : predictions_test,
    'residual'      : np.round(predictions_test - df_hdb_newf_test['fixed_price'],1),
    'percentage'    : np.round((abs((predictions_test-df_hdb_newf_test['fixed_price'])) / df_hdb_newf_test['fixed_price'])*100,1)
}
answer = pd.DataFrame(data_prediction_df_hdb_newf_test)
answer

answer['percentage'].mean()

sns.histplot(answer['percentage'], bins=40)
plt.show()





