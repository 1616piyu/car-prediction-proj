import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model and encoders
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

st.title("🚗 Car Price Predictor")

# User input
name = st.text_input("Car Name")
year = st.number_input("Year of Manufacture", min_value=1990, max_value=2025, step=1)
km_driven = st.number_input("Kilometers Driven", min_value=0)

fuel = st.selectbox("Fuel Type", label_encoders['fuel'].classes_)
seller_type = st.selectbox("Seller Type", label_encoders['seller_type'].classes_)
transmission = st.selectbox("Transmission", label_encoders['transmission'].classes_)
owner = st.selectbox("Owner Type", label_encoders['owner'].classes_)

# Preprocess user input
input_data = pd.DataFrame({
    'name': [name],
    'year': [year],
    'km_driven': [km_driven],
    'fuel': [fuel],
    'seller_type': [seller_type],
    'transmission': [transmission],
    'owner': [owner]
})

# Encode categorical features
for col in ['name', 'fuel', 'seller_type', 'transmission', 'owner']:
    if col in label_encoders:
        input_data[col] = label_encoders[col].transform(input_data[col])

# Predict
if st.button("Predict Price"):
    prediction = model.predict(input_data)[0]
    st.success(f"Estimated Selling Price: ₹{int(prediction):,}")
