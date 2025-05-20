import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import RobustScaler
import hdbscan

ran =62

df = pd.read_feather('data/intermediate/data_to_model.feather')

df_model = df.copy()


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



#  Split dataframes


def features_target_split(df):
    features = df.drop('fixed_price', axis=1)
    target = df['fixed_price']
    return features, target


df_train, df_pass = train_test_split(df_model, test_size=0.3, random_state=ran)
df_valid, df_test = train_test_split(df_pass, test_size=0.5, random_state=ran)


# Anomalies detection


#  Isolation Forest

# Takes a classifier and a dataframe and predicts the anomalies, return the df without anomalies
def predict_isolation_forest(df, df_train, has_print=False):
    
    features_isolation_array = ['stratum','bathrooms','age','rooms', 'total_area']
    features_isolation_train = df_train.copy()[features_isolation_array]
    
    clf_forest = IsolationForest(max_samples=100, 
                                max_features=0.9,
                                n_estimators=50,
                                contamination=0.1, # 0.05
                                random_state=ran)
    clf_forest.fit(features_isolation_train)
    
    df_new = df.copy()
    
    df_new['is_anomaly'] = clf_forest.predict(df_new[features_isolation_array])
    df_new['is_anomaly'] = df_new['is_anomaly'].apply(lambda x: 0 if x == 1 else 1)
    
    print('Total anomalies:' , len(df_new.query('is_anomaly == 1')))
    
    if has_print:
        scatterplot_one(df_new)
        pairplot_one(df_new)
    
    return df_new.query('is_anomaly == 0').drop('is_anomaly', axis=1)
    

#  HDBSCAN clustering


def predict_hdb_anomalies(df, df_train, has_print=False):
    
    features_isolation_array = ['stratum','bathrooms','age','rooms', 'total_area']
    features_isolation_train = df_train.copy()[features_isolation_array]
    
    cluster_hdb = hdbscan.HDBSCAN(min_cluster_size=15, prediction_data=True).fit(features_isolation_train)
    
    new_df = df.copy()
    features_hdb = new_df[features_isolation_array]
    test_labels, strengths = hdbscan.approximate_predict(cluster_hdb, features_hdb)
    new_df['is_anomaly'] = [1 if i == -1 else 0 for i in test_labels ]
    
    print('Total anomalies:' , len(new_df.query('is_anomaly == 1')))
    
    if has_print:
        scatterplot_one(new_df)
        pairplot_one(new_df)
        
    return new_df.query('is_anomaly == 0').drop('is_anomaly', axis=1)



#  Feature scaling


def scaler_transformation(df, df_train):
    scaler = RobustScaler()
    # Scale all features
    scaler.fit(df_train.drop('fixed_price', axis=1))   
    
    df_new = df.copy()
    features_df = df_new.drop('fixed_price', axis=1)
    df_new = pd.DataFrame(scaler.transform(features_df), 
                          columns=features_df.columns,
                          index=features_df.index)
    df_new['fixed_price'] = df['fixed_price']
    return df_new



#  Features selection


model_forest_selection = RandomForestRegressor(n_estimators=100,random_state=ran)
features_selection_train, target_selection_train = features_target_split(df_train)
model_forest_selection.fit(features_selection_train, target_selection_train)

data_impotances = {
    'importances' : model_forest_selection.feature_importances_,
    'names' : model_forest_selection.feature_names_in_
}
df_importances = pd.DataFrame(data_impotances, columns=['names', 'importances'], )\
                        .sort_values(by='importances', ascending=False)\
                        .reset_index(drop=True)


most_important_features_20 = list(df_importances.loc[0:20,'names'])
most_important_features_20.append('fixed_price')


# sns.barplot(df_importances.loc[0:20,:], x='importances', y='names')
# plt.savefig('files/plots/feature_importances.png')
# plt.show()



def remove_features_df(df, list_important=most_important_features_20 ):
    df_new = df.copy()
    df_new = df_new[list_important]
    return df_new



#  OHE


def ohe_transform(df):
    new_df = df.copy()
    ohe_features = new_df['stratum']
    ohe_features = pd.get_dummies(ohe_features, drop_first=True, dtype=int)
    ohe_features.columns = ['2','3', '4', '5', '6']
    new_df = new_df.drop('stratum', axis=1)
    new_df = pd.concat([new_df, ohe_features], axis=1)
    return new_df


def ohe_transform_valid(df):
    new_df = df.copy()
    ohe_features = new_df['stratum']
    ohe_features = pd.get_dummies(ohe_features, drop_first=True, dtype=int)
    ohe_features.columns = ['3', '4', '5', '6']
    new_df = new_df.drop('stratum', axis=1)
    new_df['2'] = np.full(len(new_df), 0)
    new_df = pd.concat([new_df, ohe_features], axis=1)
    return new_df


#  New features


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

list_new_features = ['total_area','stratum', 'rooms', 'bathrooms','age', 'fixed_price','area_per_room','area_per_bathroom','area_per_stratum','stratum per_rooms','age_inverted','total_amenities' ]




# ----- New Datasets



# Isolation forest

df_train_isolation = predict_isolation_forest(df_train, df_train)
df_valid_isolation = predict_isolation_forest(df_valid, df_train)
df_test_isolation = predict_isolation_forest(df_test, df_train)


# HDBSCAN


df_cluster_hdb_train_drop = predict_hdb_anomalies(df_train, df_train)
df_cluster_hdb_valid_drop = predict_hdb_anomalies(df_valid, df_train)
df_cluster_hdb_test_drop  = predict_hdb_anomalies(df_test, df_train)


# Feature scaling


df_train_scaled = scaler_transformation(df_train, df_train)
df_valid_scaled = scaler_transformation(df_valid, df_train)


# Features selection


df_train_selected_features = remove_features_df(df_train)
df_valid_selected_features = remove_features_df(df_valid)
df_test_selected_features = remove_features_df(df_test)


# New features 


df_train_new_features = create_new_features(df_train), list_new_features
df_valid_new_features = create_new_features(df_valid), list_new_features
df_test_new_features  = create_new_features(df_test), list_new_features


# New features amenities removed

df_train_new_features_removed = remove_features_df(create_new_features(df_train), list_new_features)
df_valid_new_features_removed = remove_features_df(create_new_features(df_valid), list_new_features)
df_test_new_features_removed = remove_features_df(create_new_features(df_test), list_new_features)



#  HDBSCAN + NEW FEATURES


df_hdb_newf_train = remove_features_df(create_new_features(df_cluster_hdb_train_drop), list_new_features)
df_hdb_newf_valid = remove_features_df(create_new_features(df_cluster_hdb_valid_drop), list_new_features)
df_hdb_newf_test = remove_features_df(create_new_features(df_cluster_hdb_test_drop), list_new_features)