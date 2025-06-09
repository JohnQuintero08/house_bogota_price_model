````markdown
# Real Estate Price Prediction Project

## Project Overview

- This project aims to predict house prices in Colombia using data scraped from a real estate website.
- Key libraries used include Scikit-learn for machine learning, Pandas for data manipulation, and Matplotlib/Seaborn for data visualization.
- Exploratory Data Analysis (EDA) revealed that around 99% of houses are priced below 6000 million pesos, with a significant concentration below 3000 million pesos.
- The final model selected for price prediction was XGBoost, which demonstrated a 27% reduction in Mean Absolute Error (MAE) compared to a baseline Linear Regression model.

## Motivation

The real estate market is a critical sector in any economy, and understanding price dynamics can significantly benefit buyers, sellers, and investors. This project leverages machine learning techniques to provide insights into housing prices based on various features, enabling stakeholders to make informed decisions. By utilizing advanced tools like XGBoost, we can enhance predictive accuracy, which is essential for navigating the complexities of the real estate market.

The use of web scraping to gather data from a Colombian real estate website not only showcases the power of data collection but also highlights the importance of data preprocessing and feature engineering in building robust predictive models. The insights derived from this project can help identify market trends, optimize pricing strategies, and ultimately contribute to a more efficient real estate market.

## Code and Resources

- Key libraries used in this project:
  - Scikit-learn
  - Pandas
  - Matplotlib
  - Seaborn
  - XGBoost
- To install the required libraries, run:
  ```bash
  pip install -r requirements.txt
  ```
````

## Data Collection

The data was collected through web scraping from a Colombian real estate website. The scraping process was implemented using Scrapy and Selenium. For more details, refer to another of my projects project [here](https://github.com/JohnQuintero08/house_scraping_web).

## EDA

The exploratory data analysis yielded several important insights:

- **Price Distribution**: The histogram and cumulative distribution function (CDF) of house prices indicate a right-skewed distribution, with most houses priced below 2000 million pesos. This is illustrated in the `histogram_price_distribution.png`.
- **Area vs. Price**: A scatter plot (`area_vs_price.png`) shows a positive correlation between the total area of houses and their prices, with higher-stratum houses clustering at higher price points.
- **Feature Importance**: The analysis of feature importance (`feature_importances.png`) revealed that total area is the most significant factor influencing house prices, followed by stratum, number of rooms, and bathrooms.

## Processing and Feature Engineering

Several transformations were explored to enhance model performance:

- New features were created, including total amenities, area per room, and age categories.
- Outlier filtering was performed using HDBSCAN and Isolation Forest techniques.
- Feature selection was conducted using RandomForestRegressor, leading to the removal of less important amenities.
- Data scaling was applied using RobustScaler, and one-hot encoding was utilized for categorical variables.

## Model Building

The following models were tested:

- Linear Regression
- Random Forest
- XGBoost

The combination of XGBoost with the newly engineered features and outlier isolation yielded the best performance.

## Model Performance

The optimized XGBoost model achieved the following performance metrics:

- RMSE: 222.90
- MAE: 148.37
- After optimization the model showed a 30% improvement in MAE compared to the previous version and maintained similar performance on the test dataset.

## Pipeline

To execute the project, follow these steps:

1. Create a `pp0_data` folder and place the raw data inside using the scraping spider from the link provided before.
2. Run the `pp4_01_exe.py` file located in the `pp4_execution` folder.
3. The output will be displayed in the console.

```

```
