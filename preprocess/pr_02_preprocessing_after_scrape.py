
import pandas as pd
import numpy as np

df = pd.read_feather('data/intermediate/clened_houses_data.feather')

df_copy = df.copy()

# Fill the null value of built_area in case private area is not null, or the contrary
def fill_built_area(data):
    if pd.isna(data['built_area']) and pd.notna(data['private_area']):
        data['built_area'] = data['private_area']
    return data
def fill_private_area(data):
    if pd.isna(data['private_area']) and pd.notna(data['built_area']):
        data['private_area'] = data['built_area']
    return data
df_copy = df_copy.apply(fill_built_area, axis=1)
df_copy = df_copy.apply(fill_private_area, axis=1)


# Format neighbourhoods names
def format_neighbourhoods(data):
    return data.strip().lower()
df_copy['neighbourhood'] = df_copy['neighbourhood'].apply(format_neighbourhoods)


# Assing stratum to those neighbourhoods that don't have it, based on the name of the neighbourhood
list_neighbourhood_stratum = df_copy.groupby(by=['neighbourhood'],as_index=False)['stratum'].max()
def filter_neighbourhoods(data):
    if pd.isna(data['stratum']):
        if data['neighbourhood'] in list_neighbourhood_stratum['neighbourhood'].values:
            data['stratum'] = float(list_neighbourhood_stratum[list_neighbourhood_stratum['neighbourhood'] == data['neighbourhood']]['stratum'].values[0])
    return data
df_copy = df_copy.apply(filter_neighbourhoods, axis=1)


# Keep the higher value between private_area and built_area to give a total_area of the property
df_copy['total_area'] = df_copy[['private_area', 'built_area']].max(axis=1)


df_copy['age'] = df_copy['age'].fillna('desconocido')


# Delete empty rows in any of the following columns, this data is important for the model
rows_to_dropna = ['bathrooms',
                  'rooms',
                  'built_area',
                  'stratum']
df_copy = df_copy.dropna(subset=rows_to_dropna)


# These variables are not important for the model:
# type has only one value
# status has 880 empty rows
# private_area and built_area were condensed in one new columns
columns_to_drop = ['type', 
                   'status', 
                   'private_area', 
                   'built_area',
                   'rs_agent',
                   'registered_date'
                   ]
df_copy = df_copy.drop(columns_to_drop, axis=1)


dict_age = {
    'menor a 1 año' : '< 1',
    '1 a 8 años' : '1 - 8' ,
    '9 a 15 años': '9 - 15', 
    '16 a 30 años': '16 - 30', 
    'más de 30 años': '> 30', 
    'desconocido': 'ND', 
}
df_copy['age'] = df_copy['age'].map(dict_age)


df_copy = df_copy.reset_index(drop=True)


# Delete the values of the houses that were registered wrong in the platform
df_copy = df_copy[df_copy['fixed_price'] <= 3000000000].copy()


df_copy.to_feather("data/intermediate/preprocessed_data.feather")

# Save the list of the neighbourhoods that was used to train the model
df_neighbourhood_stratum = pd.DataFrame(list_neighbourhood_stratum)
df_neighbourhood_stratum.to_feather("preprocess/neighbourhood_stratum.feather") 