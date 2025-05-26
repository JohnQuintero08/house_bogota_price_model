import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import xgboost as xgb

# Datasets

from features.f_01_features_exploration import ran
from features.f_01_features_exploration import features_target_split
from features.f_01_features_exploration import df_train
from features.f_01_features_exploration import df_valid
from features.f_01_features_exploration import df_train_isolation
from features.f_01_features_exploration import df_valid_isolation
from features.f_01_features_exploration import df_cluster_hdb_train_drop
from features.f_01_features_exploration import df_cluster_hdb_valid_drop
from features.f_01_features_exploration import df_train_scaled
from features.f_01_features_exploration import df_valid_scaled
from features.f_01_features_exploration import df_train_selected_features
from features.f_01_features_exploration import df_valid_selected_features
from features.f_01_features_exploration import df_train_new_features
from features.f_01_features_exploration import df_valid_new_features 
from features.f_01_features_exploration import df_train_new_features_removed
from features.f_01_features_exploration import df_valid_new_features_removed
from features.f_01_features_exploration import df_hdb_newf_train
from features.f_01_features_exploration import df_hdb_newf_valid
from features.f_02_log_transform import df_train_log_nf_r
from features.f_02_log_transform import df_valid_log_nf_r


# Functions


def metrics_eval(target, predictions, dataset_name):
    rmse = root_mean_squared_error(target, predictions)
    mae = mean_absolute_error(target, predictions)
    print(f'Dataset - {dataset_name}')
    print(f'RMSE: {rmse:.2f}')
    print(f'MAE: {mae:.2f}')


def graph_predictions_each_data(target, predictions):
    plt.figure(figsize=(15,6))
    arary_length = np.arange(len(target))
    plt.scatter(arary_length, predictions, marker='*', label='Predictions', s=20, alpha=0.8)
    plt.scatter(arary_length, target, label='Real values', s=20, alpha=0.8)
    plt.title('Price of the observation')
    plt.xlabel('Data number')
    plt.ylabel('Price')
    plt.legend()
    plt.show()


def graph_predictions(target, predictions, ax, title):
    ax.scatter(target, predictions, s=20, alpha=0.8)
    ax.plot([target.min(), target.max()], [target.min(), target.max()], 'r--')
    ax.set_title(title)
    ax.set_xlabel('Real value')
    ax.set_ylabel('Prediction')
    ax.legend(['Predictions', 'Ideal correlation'])


def model_evaluation(model, df_train, df_valid, model_name, has_plot=False, second_df ='Validation'):

    f_train, t_train = features_target_split(df_train)
    f_valid, t_valid = features_target_split(df_valid)

    model.fit(f_train, t_train)
    predictions_t = model.predict(f_train)
    predictions_v = model.predict(f_valid)

    print(f"Model evaluation: {model_name}")
    metrics_eval(t_train, predictions_t, 'Train')
    metrics_eval(t_valid, predictions_v, f'{second_df}')

    if has_plot:
        fig, axs = plt.subplots(1, 2, figsize=(15, 6))
        graph_predictions(t_train, predictions_t, axs[0], 'Train')
        graph_predictions(t_valid, predictions_v, axs[1], f'{second_df}')

        fig.suptitle(f'Comparison of the predictions - {model_name}', fontsize=16)
        plt.tight_layout()
        plt.show()


# ### Linear regression


model_linear = LinearRegression()
model_evaluation(model_linear, df_train, df_valid, 'Linear regression')


model_evaluation(model_linear, df_train_isolation, df_valid_isolation, 'Linear regression - Forest Isolation')


model_evaluation(model_linear, df_train_scaled, df_valid_scaled, 'Linear regression - DF escaled')


model_evaluation(model_linear, df_train_selected_features, df_valid_selected_features, 'Linear model - Selected features')


model_evaluation(model_linear, df_train_new_features, df_valid_new_features, 'Linear regression - New features')


model_evaluation(model_linear, df_train_new_features_removed, df_valid_new_features_removed, 'Linear regression - New features and feature selection')

model_evaluation(model_linear, df_train_log_nf_r, df_valid_log_nf_r, 'Linear regression  - Log transformation + New features', True)

# ### Random Forest


model_forest = RandomForestRegressor(n_estimators=50,
                                     max_depth=20,
                                     max_features=0.9,
                                     min_samples_split=10,
                                     random_state=ran)


model_evaluation(model_forest, df_train, df_valid, 'Random Forest')


model_evaluation(model_forest, df_train_isolation, df_valid_isolation, 'Random Forest - Forest Isolation')


model_evaluation(model_forest, df_train_scaled, df_valid_scaled, 'Random Forest - Df scaled')


model_evaluation(model_forest, df_train_selected_features, df_valid_selected_features, 'Random Forest - Selected features')


model_evaluation(model_forest, df_train_new_features, df_valid_new_features, 'Random Forest - New features')


model_evaluation(model_forest, df_train_new_features_removed, df_valid_new_features_removed, 'Random Forest - New features and feature selection')


# ### XGBoost


model_xgb = xgb.XGBRegressor(eval_metric='rmse',
                            learning_rate = 0.04, 
                            max_depth=6, 
                            subsample=0.9,
                            colsample_bytree=0.9,
                            n_estimators=70,
                            alpha=0.2,
                            random_state=ran)


model_evaluation(model_xgb, df_train, df_valid, 'XGBoost')


model_evaluation(model_xgb, df_train_isolation, df_valid_isolation, 'XGBoost - Forest Isolation')


model_evaluation(model_xgb, df_train_scaled, df_valid_scaled, 'XG-Boost - DF scaled')


model_evaluation(model_xgb, df_train_selected_features, df_valid_selected_features, 'XG-Boost - Selected features')


model_evaluation(model_xgb, df_train_new_features, df_valid_new_features, 'XGBoost - New features')


model_evaluation(model_xgb, df_train_new_features_removed, df_valid_new_features_removed, 'XGBoost - New features + features selection')


model_evaluation(model_xgb, df_cluster_hdb_train_drop, df_cluster_hdb_valid_drop, 'XGBoost - Cluster HDBSCAN')


model_evaluation(model_xgb, df_hdb_newf_train, df_hdb_newf_valid, 'XGBoost - Cluster HDBSCAN + New features')

model_evaluation(model_xgb, df_train_log_nf_r, df_valid_log_nf_r, 'XGBoost - Log transformation + New features', True)
# Best performance 

# df_hdb_newf_train.to_feather('data/intermediate/data_best_performance_model.feather')
# df_train_log_nf_r.to_feather('data/intermediate/data_log_performance_model.feather')