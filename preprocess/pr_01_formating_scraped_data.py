import json
import pandas as pd
import numpy as np
import re


with open('data/input/raw_houses_data.json', 'r', encoding='utf-8') as f:
    datos = json.load(f)
df = pd.DataFrame(datos)

def transform_location(raw_location):
    try:
        neighbourhood = raw_location.split(',')[0]
        return neighbourhood
    except:
        return np.nan


def transform_price(raw_price):
    try:
        price = raw_price.replace('$','').strip()
        price = int(price.replace('.',''))
        return price
    except:
        return np.nan


def transform_detail_to(raw_detail, loc):
    try:
        data = raw_detail[loc]
        if data == '¡Pregúntale!' or data == 'Sin Definir':
            return np.nan
        else:
            return data.strip()
    except:
        return np.nan


def transform_detail_to_area(raw_detail, loc):
    try:
        return float(raw_detail[loc].split(' ')[0])
    except:
        return np.nan


def transform_code(raw_code):
    try:
        return raw_code[1]
    except:
        return np.nan


def transform_agent(raw_agent):
    try:
        pattern = r"ingresada por(.*?) el"
        result = re.search(pattern, raw_agent[0])
        return result.group(1).strip().lower()
    except:
        return np.nan


def transform_date(raw_agent):
    try:
        pattern = r"el (.*?). El"
        result = re.search(pattern, raw_agent[0])
        string = result.group(1).strip().lower()
        format_string = string.replace('de','/').replace(' ','')
        if 'enero' in format_string:
            date =  format_string.replace('enero', '01')
        elif 'febrero' in format_string:
            date =  format_string.replace('febrero', '02')
        elif 'marzo' in format_string:
            date =  format_string.replace('marzo', '03')
        elif 'abril' in format_string:
            date =  format_string.replace('abril', '04')
        elif 'mayo' in format_string:
            date =  format_string.replace('mayo', '05')
        elif 'junio' in format_string:
            date =  format_string.replace('junio', '06')
        elif 'julio' in format_string:
            date =  format_string.replace('julio', '07')
        elif 'agosto' in format_string:
            date =  format_string.replace('agosto', '08')
        elif 'septiembre' in format_string:
            date =  format_string.replace('septiembre', '09')
        elif 'octubre' in format_string:
            date =  format_string.replace('octubre', '10')
        elif 'noviembre' in format_string:
            date =  format_string.replace('noviembre', '11')
        elif 'diciembre' in format_string:
            date =  format_string.replace('diciembre', '12')
        else:
            date = format_string
        return date[-10:]
    except:
        return np.nan


def dummies_comodities(df):
    # Separa los comodities y el ID en un nuevo dataframe y expande la lista de comodities en una unica fila de compinación id-comodity
    comodities = df[['comodities', 'id']].explode('comodities')
    # Genera el listado de variables dummies por todos los comodities 
    dummies = pd.get_dummies(comodities['comodities'])
    # Genera un df con los ids y las respectivas variables dummies, se une por el indice asi que no hay problema de que se corran los valores. Finalmente se agrupan por id.
    other_df = pd.concat([comodities['id'],dummies], axis=1).groupby('id').sum().reset_index().sort_values('id')
    # Se concatenan el df anterior y el original por ID
    final_df = df.merge(other_df, on='id', how='inner', sort='id')
    return final_df


def feature_columns(df):
    new_df = df.copy()
    new_df['id'] = new_df['code'].apply(transform_code)
    new_df['neighbourhood'] = new_df['location'].apply(transform_location)
    new_df['fixed_price'] = new_df['price'].apply(transform_price)
    new_df['stratum'] = new_df['details'].apply(lambda x: transform_detail_to(x, 0)).astype('float')
    new_df['type'] = new_df['details'].apply(lambda x: transform_detail_to(x, 1))
    new_df['status'] = new_df['details'].apply(lambda x: transform_detail_to(x, 2))
    new_df['bathrooms'] = new_df['details'].apply(lambda x: transform_detail_to(x, 3)).astype('float')
    new_df['built_area'] = new_df['details'].apply(lambda x: transform_detail_to_area(x,4))
    new_df['private_area'] = new_df['details'].apply(lambda x: transform_detail_to_area(x,5))
    new_df['age'] = new_df['details'].apply(lambda x: transform_detail_to(x, 6))
    new_df['rooms'] = new_df['details'].apply(lambda x: transform_detail_to(x, 7)).astype('float')
    new_df['rs_agent'] = new_df['real_state_agent'].apply(transform_agent)
    new_df['registered_date'] = pd.to_datetime(new_df['real_state_agent'].apply(transform_date), format='%d/%m/%Y', errors='coerce')
    new_df = dummies_comodities(new_df)
    new_df = new_df.drop(['location', 'price', 'details', 'real_state_agent', 'code', 'comodities'], axis=1)
    return new_df


df_cleaned = feature_columns(df)

df_cleaned.to_feather("data/intermediate/clened_houses_data.feather")
cleaned_data = pd.read_feather("data/intermediate/clened_houses_data.feather")

