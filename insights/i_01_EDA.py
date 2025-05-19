import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_feather('data/intermediate/clened_houses_data.feather')

df.head()
df.info()


df['neighbourhood'].value_counts()


plt.figure(figsize=(15,3))
plt.boxplot(df['fixed_price'], vert=False)
plt.title('Boxplot - Price distribution')
plt.xlabel('Price')
plt.show()


# - There are some houses that are poblably have wrong price


def hist_graph(data, name, xlabl, ylabl, discrt = True, xticks=None, size= (7,5)):
    height = size[1]
    aspect = size[0] / size[1]
    graph = sns.displot(data, discrete=discrt, height=height, aspect=aspect)

    ax = graph.ax
    ax.set_title(f'Histogram - {name}')
    ax.set_xlabel(xlabl)
    ax.set_ylabel(ylabl)

    if xticks is not None:
        ax.set_xticks(xticks)

    ax.grid(True)
    plt.tight_layout()
    plt.show()



hist_graph(df['stratum'], 'Stratum distribution', 'Stratum', 'Number of houses')


hist_graph(df['bathrooms'], 'Bathrooms distribution', 'Bathrooms', 'Number of houses', True, np.arange(1,11,1))


hist_graph(df['built_area'], 'Built area distribution', 'Built area', 'Number of houses', False)


hist_graph(df['private_area'], 'Private area distribution', 'Private area', 'Number of houses', False)


hist_graph(df['age'], 'Age distribution', 'Age', 'Number of houses')


hist_graph(df['rooms'], 'Rooms distribution', 'Rooms', 'Number of houses', True, np.arange(1,21,1) )


plt.figure(figsize=(7,5))
df['registered_date'].hist(bins=50, edgecolor='lightblue')
plt.title(f'Histogram - Date registered')
plt.xlabel('Year')
plt.ylabel('Number of houses')
plt.grid(True)
plt.tight_layout()
plt.show()


df.query('built_area == private_area')['id'].count()


df['private_area'].isna().sum()


df['fixed_price'].sort_values(ascending=False)


df_t_copy = df.copy()
pd.set_option('display.max_rows', None)
# Scaling the data
df_t_copy['fixed_price'] = df_t_copy['fixed_price']/1000000
# Delete 2 outliers 
df_t_copy = df_t_copy.query('fixed_price < 100000')
# Spliting data into bins
df_t_copy['grupo'] = pd.cut(df_t_copy['fixed_price'], bins=100)
# Counting the number of examples per bin
frequency = df_t_copy['grupo'].value_counts(normalize=True).sort_index()
# Adding to create a cumulative
cumulative = frequency.cumsum() * 100
cumulative


fig, ax1 = plt.subplots(figsize=(15, 6))

# Primer gráfico: Histograma con eje Y izquierdo
sns.histplot(df_t_copy['fixed_price'], stat='count', bins=100, edgecolor='black', ax=ax1, color='skyblue')
ax1.set_ylabel('Porcentage (Histogram)')
ax1.set_xlabel('Price of the house (millions of pesos)')
ax1.tick_params(axis='y')

# Segundo eje Y (derecho) para la línea acumulada
ax2 = ax1.twinx()
sns.histplot(df_t_copy['fixed_price'], bins=100, stat='percent', cumulative=True, fill=False, ax=ax2, color='darkred', element='step')
ax2.set_ylabel('Cumulative price percentage', color='darkred')
ax2.tick_params(axis='y', labelcolor='darkred')

# Título y diseño
plt.title('Histogram of price distribution and Price percentage acumulated')
ax1.grid(True)
plt.tight_layout()
plt.savefig('files/plots/histogram_price_distribution.png')
plt.show()


# - Around 99% of the houses have a price lower than 6000 millions pesos.
# - Around 96% of the houses have a price lower than 3000 millions pesos.
# - Around 90% of the houses have a price lower than 2000 millions pesos.
# - 76% of the private area data are the same as built area or they are missing.
# - Most of the properties were registered after 2024.
# - The majority of the houses have 3 to 5 rooms and 2 to 4 bathrooms.
# - The stratum 3 and 4 is predomiinant over the others.
# - Most of the houses have more that 16 years since they were built.

