import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
from matplotlib import pyplot as plt 
from features.f_01_features_exploration import create_new_features, remove_features_df, list_new_features
import hdbscan

ran =62

df = pd.read_feather('data/intermediate/data_to_model.feather')

df_model = df.copy()

df_train, df_pass = train_test_split(df_model, test_size=0.3, random_state=ran)
df_valid, df_test = train_test_split(df_pass, test_size=0.5, random_state=ran)

df_model.head()

def log_transformation(df, features):
    df_c = df.copy()
    for feature in features:
        df_c[feature] = np.log1p(df[feature])
    return df_c

new_df = log_transformation(df_model, ['fixed_price', 'total_area'])

df_train_log = log_transformation(df_train, ['fixed_price', 'total_area'])
df_valid_log = log_transformation(df_valid, ['fixed_price', 'total_area'])
df_test_log = log_transformation(df_test, ['fixed_price', 'total_area'])

plt.figure(figsize=(15,6))
sns.scatterplot(df_train_log,  
                x='total_area',
                y='fixed_price',
                # hue='is_anomaly', 
                palette="deep"
                )
plt.show()


sns.pairplot(df_train_log[['fixed_price',
                    'stratum',	
                    'bathrooms',
                    'age',
                    'rooms', 
                    'total_area', 
                    # 'is_anomaly'
                    ]], 
                # hue='is_anomaly', 
                palette='magma')
plt.show()

def scatterplot_one(df):
    plt.figure(figsize=(15,6))
    sns.scatterplot(df,  
                    x='total_area',
                    y='fixed_price',
                    hue='is_anomaly', 
                    palette="deep"
                    )
    plt.show()
    
def pairplot_one(df):
    sns.pairplot(df[['fixed_price',
                     'stratum',	
                     'bathrooms',
                     'age',
                     'rooms', 
                     'total_area', 
                     'is_anomaly']], 
                 hue='is_anomaly', 
                 palette='magma')
    plt.show()


def predict_hdb_anomalies(df, df_train, has_print=False):
    
    features_isolation_array = ['stratum','bathrooms','age','rooms', 'total_area']
    features_isolation_train = df_train.copy()[features_isolation_array]
    
    cluster_hdb = hdbscan.HDBSCAN(min_cluster_size=2, prediction_data=True).fit(features_isolation_train)
    
    new_df = df.copy()
    features_hdb = new_df[features_isolation_array]
    test_labels, strengths = hdbscan.approximate_predict(cluster_hdb, features_hdb)
    new_df['is_anomaly'] = [1 if i == -1 else 0 for i in test_labels ]
    
    
    print('Total anomalies:' , len(new_df.query('is_anomaly == 1')))
    
    if has_print:
        scatterplot_one(new_df)
        # pairplot_one(new_df)
        
    return new_df.query('is_anomaly == 0').drop('is_anomaly', axis=1)


df_train_log_hdb = predict_hdb_anomalies(df_train_log, df_train_log, True)

df_train_log_nf_r = remove_features_df(create_new_features(df_train_log), list_new_features)
df_valid_log_nf_r = remove_features_df(create_new_features(df_valid_log), list_new_features)
df_test_log_nf_r = remove_features_df(create_new_features(df_test_log), list_new_features)
