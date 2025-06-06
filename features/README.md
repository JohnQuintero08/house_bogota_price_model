FEATURES OVERVIEW

Multiple versions of the dataset are generated to explore which transformations yield better model performance. This includes:

- Feature engineering, creation of new variables: total amenities, area per room, area per bathroom, stratum per room, and the reverse of the age.
- Outlier filtering using HDBSCAN and Isolation Forest.
- Feature selection using the importances with RandomForestRegressor. Most of the ammenies were drop.
- Data scaling using RobustScaler from scikitlearn
- One hot encoding of the variables
