import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import hdbscan
import joblib


# Plotting

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


#  HDBSCAN clustering


def predict_hdb_anomalies(df, has_print=False):
    
    features_isolation_array = ['stratum','bathrooms','age','rooms', 'total_area']
    cluster_hdb = joblib.load('pipeline/pp2_features/model_hdbscan_anomalies_predictor.pkl')
    
    new_df = df.copy()
    features_hdb = new_df[features_isolation_array]
    test_labels, strengths = hdbscan.approximate_predict(cluster_hdb, features_hdb)
    new_df['is_anomaly'] = [1 if i == -1 else 0 for i in test_labels ]
    
    print('Total anomalies:' , len(new_df.query('is_anomaly == 1')))
    
    if has_print:
        scatterplot_one(new_df)
        pairplot_one(new_df)
        
    return new_df.query('is_anomaly == 0').drop('is_anomaly', axis=1)

# Feature creation

def remove_features_df(df):
    list_new_features = [
    'fixed_price',
    'total_area',
    'stratum',
    'rooms',
    'bathrooms',
    'age',
    'area_per_room',
    'area_per_bathroom',
    'area_per_stratum',
    'stratum per_rooms',
    'age_inverted',
    'total_amenities' 
    ]
    df_new = df.copy()
    df_new = df_new[list_new_features]
    return df_new


def create_new_features(df):
    f_df = df.copy()
    total_columns = list(f_df.columns)
    columns_to_delete = ['total_area','stratum', 'rooms', 'bathrooms','age', 'fixed_price']
    total_amenities = [i for i in total_columns if i not in columns_to_delete]    
    f_df['total_amenities'] = f_df[total_amenities].sum(axis=1)
    f_df['area_per_room'] = f_df['total_area'] / f_df['rooms']
    f_df['area_per_bathroom'] = f_df['total_area'] / f_df['bathrooms']
    f_df['area_per_stratum'] = f_df['total_area'] / f_df['stratum']
    f_df['stratum per_rooms'] = f_df['stratum'] / f_df['rooms']
    f_df['age_inverted'] = 1/ ( f_df['age'] +1)
    
    return f_df

def develop_feature_engineering(df):
    df_c = df.copy()
    df_c = df_c.drop(['id', 'neighbourhood'], axis=1)
    df_c = predict_hdb_anomalies(df_c)
    df_c = create_new_features(df_c)
    df_c = remove_features_df(df_c)
    return df_c


df = pd.read_feather('pipeline/pp0_data/pp1_02_data.feather')

df_model = df.copy()
df_model.info()
df_model = develop_feature_engineering(df_model)

df_model.to_feather('pipeline/pp0_data/pp2_01_data.feather')