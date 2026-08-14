# House_price_prediction
House Price Prediction using Machine Learning and Streamlit An end-to-end machine learning project that predicts house prices based on property features such as square footage, number of bedrooms, bathrooms, and location. It includes a complete data science pipeline in Jupyter Notebook—covering exploratory data analysis (EDA), feature engineering.


# 🏡 House Price Prediction App

[streamlit][https://housepriceprediction03.streamlit.app/]

An end-to-end Machine Learning project that predicts the selling price of a house based on various property features. The project consists of a Jupyter Notebook for model training and a Streamlit web application for interactive predictions.

## 🌟 Overview
This project applies machine learning techniques to a housing dataset. We handle missing values, engineer new features from dates, and apply log transformations to skewed data. We then evaluate three different models:
*   Linear Regression
*   Random Forest Regressor
*   **Gradient Boosting Regressor** (Selected as the best performing model)

The best model is saved using `joblib` and served through a user-friendly Streamlit web interface.

## 🛠️ Tech Stack
*   **Language:** Python
*   **Data Manipulation:** Pandas, NumPy
*   **Machine Learning:** Scikit-Learn
*   **Web Framework:** Streamlit
*   **Model Serialization:** Joblib

## 📂 Project Structure
```text
├── data.csv                   # The raw housing dataset
├── house_price_prediction.ipynb # Jupyter Notebook with EDA, training, and evaluation
├── house_price_model.pkl      # The trained model pipeline (generated from the notebook)
├── app.py                     # Streamlit web application script
└── README.md                  # Project documentation
