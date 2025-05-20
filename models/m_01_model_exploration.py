import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

ran =62

# Datasets

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
from features.f_01_features_exploration import df_valid_new_features #TODO ARREGLAR CARACTERISTICAS TIENE ERROR
from features.f_01_features_exploration import df_train_new_features_removed
from features.f_01_features_exploration import df_valid_new_features_removed
from features.f_01_features_exploration import df_hdb_newf_train
from features.f_01_features_exploration import df_hdb_newf_valid
from features.f_01_features_exploration import df_hdb_newf_test


# Functions


def metrics_eval(target, predictions, dataset_name):
    rmse = root_mean_squared_error(target, predictions)
    mae = mean_absolute_error(target, predictions)
    print(f'Dataset de {dataset_name}')
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
    ax.set_xlabel('Valor real')
    ax.set_ylabel('Predicción')
    ax.legend(['Predicciones', 'Línea ideal'])


def model_evaluation(model, df_train, df_valid, model_name, has_plot=False):

    f_train, t_train = features_target_split(df_train)
    f_valid, t_valid = features_target_split(df_valid)

    model.fit(f_train, t_train)
    predictions_t = model.predict(f_train)
    predictions_v = model.predict(f_valid)

    print(f"📊 Evaluación del modelo: {model_name}")
    metrics_eval(t_train, predictions_t, 'Entrenamiento')
    metrics_eval(t_valid, predictions_v, 'Validación')

    if has_plot:
        fig, axs = plt.subplots(1, 2, figsize=(15, 6))
        graph_predictions(t_train, predictions_t, axs[0], 'Entrenamiento')
        graph_predictions(t_valid, predictions_v, axs[1], 'Validación')

        fig.suptitle(f'Comparación de Predicciones - {model_name}', fontsize=16)
        plt.tight_layout()
        plt.show()


# ### Regresión lineal


model_linear = LinearRegression()
model_evaluation(model_linear, df_train, df_valid, 'Regresión Lineal')


model_evaluation(model_linear, df_train_isolation, df_valid_isolation, 'Regresión Lineal - Aislado')


model_evaluation(model_linear, df_train_scaled, df_valid_scaled, 'Regresión Lineal - DF escalado')


model_evaluation(model_linear, df_train_selected_features, df_valid_selected_features, 'Linear model - Selected features')


model_evaluation(model_linear, df_train_new_features, df_valid_new_features, 'Regresión Lineal - Nuevas caracteristicas')


model_evaluation(model_linear, df_train_new_features_removed, df_valid_new_features_removed, 'Regresión Lineal - Nuevas caracteristicas')


# ### Random Forest


model_forest = RandomForestRegressor(n_estimators=50,
                                     max_depth=20,
                                     max_features=0.9,
                                     min_samples_split=10,
                                     random_state=ran)


model_evaluation(model_forest, df_train, df_valid, 'Random Forest')


model_evaluation(model_forest, df_train_isolation, df_valid_isolation, 'Random Forest - Isolation')


model_evaluation(model_forest, df_train_scaled, df_valid_scaled, 'Random Forest - Escalado')


model_evaluation(model_forest, df_train_selected_features, df_valid_selected_features, 'Random Forest - Selected features')


model_evaluation(model_forest, df_train_new_features, df_valid_new_features, 'Random Forest - Nuevas caracteristicas')


model_evaluation(model_forest, df_train_new_features_removed, df_valid_new_features_removed, 'Random Forest - Nuevas caracteristicas')


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


model_evaluation(model_xgb, df_train_scaled, df_valid_scaled, 'XG-Boost - DF escalado')


model_evaluation(model_xgb, df_train_selected_features, df_valid_selected_features, 'XG-Boost - Selected features')


model_evaluation(model_xgb, df_train_new_features, df_valid_new_features, 'XGBoost - Nuevas caracteristicas')


model_evaluation(model_xgb, df_train_new_features_removed, df_valid_new_features_removed, 'XGBoost - Nuevas caracteristicas')


model_evaluation(model_xgb, df_cluster_hdb_train_drop, df_cluster_hdb_valid_drop, 'XGBoost - Cluster HDBSCAN')


model_evaluation(model_xgb, df_hdb_newf_train, df_hdb_newf_valid, 'XGBoost - Cluster HDBSCAN + New features')


# ## Optimizacion


df_opt = pd.concat([df_hdb_newf_train, df_hdb_newf_valid], axis=0)
f_opt_nf, t_opt_nf = features_target_split(df_opt)


model_xgb_selected = xgb.XGBRegressor(eval_metric='rmse',
                            random_state=ran)
params = {
    'learning_rate'     : [0.01, 0.04, 0.08], 
    'max_depth'         : [3, 6, 12], 
    'subsample'         : [0.5, 0.9],
    'colsample_bytree'  : [0.5, 0.9],
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


# ## Prueba con datos de test


model_evaluation(best_xgb, df_hdb_newf_train, df_hdb_newf_test, 'XGBoost - Cluster HDBSCAN + New features')


# sample = df_hdb_newf_test.sample(10, random_state=310)
sample = df_hdb_newf_test.copy()
f_test, t_test = features_target_split(sample)
predictions_test = best_xgb.predict(f_test)

data_prediction_sample = {
    'real': sample['fixed_price'],
    'prediction': predictions_test,
    'residual': predictions_test - sample['fixed_price'],
    'percentage': (abs((predictions_test-sample['fixed_price'])) / sample['fixed_price'])*100,
}
answer = pd.DataFrame(data_prediction_sample)
answer
 


sns.scatterplot(answer, x='real', y='residual')
plt.show()


answer['percentage'].mean()


sns.histplot(answer['percentage'], bins=40)
plt.show()





