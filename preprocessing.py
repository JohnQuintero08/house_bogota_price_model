import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_feather('../data/clened_houses_data.feather')

df_copy = df.copy()

# Rellenar los valores nulos de la variable built_area en caso de que la variable private_area si los tenga
def fill_built_area(data):
    if pd.isna(data['built_area']) and pd.notna(data['private_area']):
        data['built_area'] = data['private_area']
    return data

def fill_private_area(data):
    if pd.isna(data['private_area']) and pd.notna(data['built_area']):
        data['private_area'] = data['built_area']
    return data

# Le da formato a todos los barrios
def format_neighbourhoods(data):
    return data.strip().lower()

# Asgina un estrato a los barrios que no tienes estrato, basado en los barrios ya preexistentes
list_neighbourhood_stratum = df_copy.groupby(by=['neighbourhood'],as_index=False)['stratum'].max()
def filter_neighbourhoods(data):
    if pd.isna(data['stratum']):
        if data['neighbourhood'] in list_neighbourhood_stratum['neighbourhood'].values:
            data['stratum'] = list_neighbourhood_stratum[list_neighbourhood_stratum['neighbourhood'] == data['neighbourhood']]['stratum'].values
    return data

# Toma el valor máximo entre el area construida y el area privada para considerar el area total de la propiedad
def define_area(data):
    private_area = data['private_area']
    built_area = data['built_area']
    return np.max([private_area, built_area])


df_copy = df_copy.apply(fill_built_area, axis=1)

df_copy = df_copy.apply(fill_private_area, axis=1)

df_copy['neighbourhood'] = df_copy['neighbourhood'].apply(format_neighbourhoods)

df_copy = df_copy.apply(filter_neighbourhoods, axis=1)

df_copy['total_area'] = df_copy.apply(define_area, axis=1)

df_copy['age'] = df_copy['age'].fillna('desconocido')

# Elimina los valores vacios de estas columnas
rows_to_dropna = ['bathrooms',
                  'rooms',
                  'built_area']
df_copy = df_copy.dropna(subset=rows_to_dropna)


# Estas variables no son utiles 
columns_to_drop = ['type',          # type es solo casa 
                   'status',        # status hay 880 variables vacias
                   'private_area',  # private_area es en el 60% de los casos es igual a built area
                   'built_area',    # se totaliza entre private_area y total_area la mayor
                   'rs_agent',      # no se usara para entrenar el modelo
                   'registered_date'# no presenta información como tal de la propiedad
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






