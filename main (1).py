import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Set page configuration
st.set_page_config(page_title="House Price Predictor", layout="wide")

# --- Load Data & Model ---
@st.cache_resource
def load_artifacts():
    model = pickle.load(open('xgboost_model.pkl', 'rb'))
    encoders = pickle.load(open('label_encoder.pkl', 'rb'))
    scaler_fitur = pickle.load(open('scaler_fitur.pkl', 'rb'))
    target_scaler = pickle.load(open('target_scaler.pkl', 'rb'))
    rentang = pickle.load(open('rentang_fitur.pkl', 'rb'))
    return model, encoders, scaler_fitur, target_scaler, rentang

try:
    model, encoders, scaler_fitur, target_scaler, rentang = load_artifacts()
except Exception as e:
    st.error(f"Error loading model artifacts: {e}")
    st.stop()

# --- UI/UX ---
st.title("🏠 Smart House Price Prediction App")
st.markdown("Enter the house details below to estimate the market price.")
st.divider()

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Physical Features")
        bedrooms = st.slider("Bedrooms",
                             min_value=float(rentang['bedrooms']['min']),
                             max_value=float(rentang['bedrooms']['max']),
                             value=float(rentang['bedrooms']['min']))

        bathrooms = st.slider("Bathrooms",
                              min_value=float(rentang['bathrooms']['min']),
                              max_value=float(rentang['bathrooms']['max']),
                              value=float(rentang['bathrooms']['min']))

        floors = st.slider("Floors",
                           min_value=float(rentang['floors']['min']),
                           max_value=float(rentang['floors']['max']),
                           value=float(rentang['floors']['min']))

    with col2:
        st.subheader("Space & Location")
        sqft_living = st.slider("Living Area (Sqft)",
                                min_value=float(rentang['sqft_living']['min']),
                                max_value=float(rentang['sqft_living']['max']),
                                value=float(rentang['sqft_living']['min']))

        sqft_above = st.slider("Above Ground Area (Sqft)",
                               min_value=float(rentang['sqft_above']['min']),
                               max_value=float(rentang['sqft_above']['max']),
                               value=float(rentang['sqft_above']['min']))

        city = st.selectbox("City", options=list(encoders['city'].classes_))
        statezip = st.selectbox("State Zip Code", options=list(encoders['statezip'].classes_))

    st.divider()
    submit_button = st.form_submit_button("Predict House Price")

if submit_button:
    # 1. Pre-processing: Categorical Encoding
    enc_city = encoders['city'].transform([city])[0]
    enc_zip = encoders['statezip'].transform([statezip])[0]

    # 2. Construct DataFrame with exact training order
    input_data = pd.DataFrame({
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms],
        'sqft_living': [sqft_living],
        'floors': [floors],
        'sqft_above': [sqft_above],
        'city': [enc_city],
        'statezip': [enc_zip]
    })

    # 3. Scaling features
    input_scaled = scaler_fitur.transform(input_data)

    # 4. Prediction
    pred_scaled = model.predict(input_scaled)

    # 5. Inverse Transform target
    # Reshape is required because inverse_transform expects 2D
    price_final = target_scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]

    # Output Results
    st.balloons()
    st.success(f"### Estimated House Price: ${price_final:,.2f}")
