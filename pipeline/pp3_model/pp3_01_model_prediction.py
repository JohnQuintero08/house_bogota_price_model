import pandas as pd
import numpy as np
import joblib
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


def features_target_split(df):
    features = df.drop('fixed_price', axis=1)
    target = df['fixed_price']
    return features, target

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


def graph_predictions(target, predictions):
    plt.scatter(target, predictions, s=20, alpha=0.8)
    plt.plot([target.min(), target.max()], [target.min(), target.max()], 'r--')
    plt.title('Correlation real value vs predictions')
    plt.xlabel('Real value')
    plt.ylabel('Prediction')
    plt.legend(['Predictions', 'Ideal correlation'])
    plt.show()


def model_prediction(model, df, has_plot=False):

    features, target = features_target_split(df)
    predictions_t = model.predict(features)
    print(predictions_t)

    metrics_eval(target, predictions_t, 'New data')

    if has_plot:
        graph_predictions(target, predictions_t)
        
    return predictions_t

xgb_model = joblib.load('pipeline/pp3_model/predictor_xgb_model.joblib')
df = pd.read_feather('pipeline/pp0_data/pp2_01_data.feather')
predictions = model_prediction(xgb_model, df, True)