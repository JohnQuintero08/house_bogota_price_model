import json
import pandas as pd

from pipeline.pp1_preprocess.pp1_01_format import feature_columns
from pipeline.pp1_preprocess.pp1_02_preprocess_s import preprocess_data_scrap
from pipeline.pp2_features.pp2_01_feature_engineering import develop_feature_engineering
from pipeline.pp3_model.pp3_01_model_prediction import model_prediction

with open('pipeline/pp0_data/raw_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    
df = pd.DataFrame(data)
df_c = df.copy()
df_c = feature_columns(df_c)
df_c = preprocess_data_scrap(df_c)
df_c = develop_feature_engineering(df_c)
df_c = model_prediction(df_c)