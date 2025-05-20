import pandas as pd


# Fill the null value of built_area in case private area is not null, or the contrary
def fill_built_area(data):
    if pd.isna(data['built_area']) and pd.notna(data['private_area']):
        data['built_area'] = data['private_area']
    return data

def fill_private_area(data):
    if pd.isna(data['private_area']) and pd.notna(data['built_area']):
        data['private_area'] = data['built_area']
    return data

# Format neighbourhoods names
def format_neighbourhoods(data):
    return data.strip().lower()

# Assign stratum to those neighbourhoods that don't have it, based on the name of the neighbourhood

def filter_neighbourhoods(data):
    list_neighbourhood_stratum = pd.read_feather('pipeline/pp1_preprocess/neighbourhood_stratum.feather')
    if pd.isna(data['stratum']):
        if data['neighbourhood'] in list_neighbourhood_stratum['neighbourhood'].values:
            data['stratum'] = float(list_neighbourhood_stratum[list_neighbourhood_stratum['neighbourhood'] == data['neighbourhood']]['stratum'].values[0])
    return data

def dropna_data(df):
    df_new = df.copy()
    rows_to_dropna = ['bathrooms',
                  'rooms',
                  'built_area',
                  'stratum']
    return df_new.dropna(subset=rows_to_dropna)

def drop_columns(df):
    df_new = df.copy()
    columns_to_drop = ['type', 
                   'status', 
                   'private_area', 
                   'built_area',
                   'rs_agent',
                   'registered_date'
                   ]
    return df_new.drop(columns_to_drop, axis=1)

def format_age(df):
    df_new = df.copy()
    dict_age = {
        'menor a 1 año'   : 1,
        '1 a 8 años'      : 2,
        '9 a 15 años'     : 3,
        '16 a 30 años'    : 4,
        'más de 30 años'  : 5,
        'desconocido'     : 0,
    }
    df_new['age'] = df_new['age'].map(dict_age)
    return df_new

def preprocess_data_scrap(df):
    df_copy = df.copy()
    df_copy = df_copy.apply(fill_built_area, axis=1)
    df_copy = df_copy.apply(fill_private_area, axis=1)
    df_copy['neighbourhood'] = df_copy['neighbourhood'].apply(format_neighbourhoods)
    df_copy = df_copy.apply(filter_neighbourhoods, axis=1)
    df_copy['total_area'] = df_copy[['private_area', 'built_area']].max(axis=1)
    df_copy['age'] = df_copy['age'].fillna('desconocido')
    df_copy = dropna_data(df_copy)
    df_copy = drop_columns(df_copy)
    df_copy = format_age(df_copy)
    df_copy = df_copy.reset_index(drop=True)
    df_copy['fixed_price'] = df_copy['fixed_price']/1000000
    df_copy = df_copy[df_copy['fixed_price'] <= 6000] 
    return df_copy

df = pd.read_feather('pipeline/pp0_data/pp1_01_data.feather')
df_copy = df.copy()
df_copy = preprocess_data_scrap(df_copy)
df_copy.to_feather('pipeline/pp0_data/pp1_02_data.feather')


