# Bogotá Housing Price Prediction

This project builds a machine learning model to predict housing prices in Bogotá, Colombia, using data collected through web scraping from a real estate website. A total of 1,000 listings were scraped and processed.

## Project Structure

### 1. `preprocess/`

This folder contains Python scripts to clean and preprocess the raw scraped data. Tasks include:

- Standardizing column names and fixing data types.
- Handling missing or inconsistent values.
- Preparing the data for further analysis and modeling.

### 2. `insights/`

Exploratory Data Analysis (EDA) is conducted here to:

- Understand feature distributions and relationships.
- Identify patterns, trends, and potential outliers.

### 3. `features/`

Multiple versions of the dataset are generated to explore which transformations yield better model performance. This includes:

- Feature engineering (creation of new variables).
- Outlier filtering using HDBSCAN and Isolation Forest.
- Feature selection.
- Data scaling.

### 4. `models/`

This section explores several regression algorithms:

- Linear Regression
- Random Forest Regressor
- XGBoost Regressor

Each model is evaluated using regression metrics such as **MAE** (Mean Absolute Error) and **RMSE** (Root Mean Square Error). Grid Search is applied to optimize the best-performing model.

### 5. `pipeline/`

A full pipeline is defined to allow processing and prediction on new data obtained from the same real estate site.

### 6. `files/`

Contains selected outputs and variable importance analysis. According to the results, **housing area** is the most influential feature in predicting the price.

## Final Model and Results

The best model is an **XGBoost Regressor**, trained on:

- A dataset filtered using **HDBSCAN** to remove noise.
- Additional engineered features.
- Irrelevant features removed via feature selection.

**Performance:**

- Average error between **15% and 20%**.
- Most accurate for properties priced **below 2 billion Colombian pesos**.

## Future Improvements

- Train with a larger dataset from the website or complementary external sources.
- Integrate more socioeconomic or geographic features.
- Improve model performance for high-priced properties.

## Data Collection

The data used in this project was obtained via web scraping. The scraping logic is available in a separate repository:

👉 [house_scraping_web](https://github.com/JohnQuintero08/house_scraping_web)

---

**Note:** This project is intended for educational and experimental purposes. The data and predictions should not be used for real financial decisions without further validation.
