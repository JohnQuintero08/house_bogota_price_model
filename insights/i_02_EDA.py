import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


df = pd.read_feather("data/intermediate/data_to_model.feather")

df_copy = df.copy()

df.isna().sum()

df_copy.duplicated().sum()


# - No data duplicated
# - But there are some missing values in several variables.

# Delete the values of the houses that were registered wrong in the platform
df_final = df_copy[df_copy['fixed_price'] <= 3000000000].copy()

columns_to_pair = ['fixed_price', 'rooms','age', 'total_area', 'bathrooms', 'stratum' ]
sns.pairplot(df_final[columns_to_pair])
plt.savefig('files/plots/pairplot_data_eda.png')
plt.show()

plt.figure(figsize=(15,6))
sns.scatterplot(df_final,  
                x='total_area',
                y='fixed_price',
                hue='stratum',
                palette="deep"
                )
plt.xlabel('Total area, m^2')
plt.ylabel('Total price, thousand millions pesos')
plt.title('Area vs price')
plt.savefig('files/plots/area_vs_price.png')
plt.show()

plt.figure(figsize=(15,6))
sns.scatterplot(df_final,  
                x='total_area',
                y='stratum',
                hue='fixed_price', 
                )
plt.xlabel('Total area, m^2')
plt.ylabel('Stratum')
plt.title('Area vs stratum')
plt.show()



