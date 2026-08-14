import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime

# --- 1. Load the trained model ---
@st.cache_resource
def load_model():
    # Load the pipeline saved from the Jupyter notebook
   # Line 11
    # Load the pipeline saved from the Jupyter notebook
 return joblib.load('/Users/yashtripathi/Desktop/internship_project/project-2/house_price_model.pkl')  # Use relative path
model = load_model()

# --- 2. Build the Streamlit UI ---
st.title("🏡 House Price Prediction App")
st.write("Enter the property details below to get an estimated price.")

# Organize inputs into columns for a cleaner layout
col1, col2, col3 = st.columns(3)

with col1:
    date = st.date_input("Date of Sale", datetime.date(2014, 8, 1))
    bedrooms = st.number_input("Bedrooms", min_value=0.0, value=3.0, step=1.0)
    bathrooms = st.number_input("Bathrooms", min_value=0.0, value=2.0, step=0.25)
    floors = st.selectbox("Floors", options=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
    waterfront = st.selectbox("Waterfront", options=[0, 1], help="0 = No, 1 = Yes")

with col2:
    sqft_living = st.number_input("Sqft Living", min_value=100, value=1500, step=50)
    sqft_lot = st.number_input("Sqft Lot", min_value=100, value=5000, step=100)
    sqft_above = st.number_input("Sqft Above", min_value=0, value=1500, step=50)
    sqft_basement = st.number_input("Sqft Basement", min_value=0, value=0, step=50)
    view = st.selectbox("View Rating", options=[0, 1, 2, 3, 4])

with col3:
    yr_built = st.number_input("Year Built", min_value=1800, max_value=datetime.date.today().year, value=1990)
    yr_renovated = st.number_input("Year Renovated (0 if none)", min_value=0, value=0)
    condition = st.slider("Condition", min_value=1, max_value=5, value=3)
    city = st.text_input("City", value="Seattle")
    statezip = st.text_input("State & Zip", value="WA 98133")

# --- 3. Prediction Logic ---
if st.button("Predict Price", type="primary"):
    
    # Pack the inputs into a dictionary matching your training data structure
    input_data = {
        'date': pd.to_datetime(date),
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'sqft_living': sqft_living,
        'sqft_lot': sqft_lot,
        'floors': floors,
        'waterfront': waterfront,
        'view': view,
        'condition': condition,
        'sqft_above': sqft_above,
        'sqft_basement': sqft_basement,
        'yr_built': yr_built,
        'yr_renovated': yr_renovated,
        'city': city,
        'statezip': statezip
    }
    
    # Convert to DataFrame
    df = pd.DataFrame([input_data])
    
    # Apply the same manual feature engineering used in the notebook
    df['year_sold'] = df['date'].dt.year
    df['month_sold'] = df['date'].dt.month
    df['log_sqft_living'] = np.log1p(df['sqft_living'])
    df['log_sqft_lot'] = np.log1p(df['sqft_lot'])
    
    # Predict and inverse-transform the log price
    try:
        log_pred = model.predict(df)
        predicted_price = np.expm1(log_pred)[0]
        
        st.success(f"### Estimated Price: ${predicted_price:,.2f}")
        st.balloons()
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}\n\n*Make sure you entered a valid city/statezip that the model recognizes.*")