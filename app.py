import streamlit as st
import numpy as np
import pickle

# Load trained model
with open("model_RF.sav", "rb") as f:
    model = pickle.load(f)

# Load scaler
with open("scaler_model.sav", "rb") as f:
    scaler = pickle.load(f)

st.title("🍷 Wine Quality Prediction App")
st.write("Enter wine chemical properties to predict quality")

# User inputs
fixed_acidity = st.number_input("Fixed Acidity", value=7.4)
volatile_acidity = st.number_input("Volatile Acidity", value=0.7)
citric_acid = st.number_input("Citric Acid", value=0.0)
residual_sugar = st.number_input("Residual Sugar (log)", value=0.64)
chlorides = st.number_input("Chlorides (log)", value=-2.57)
free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide (log)", value=0.56)
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide (log)", value=0.44)
density = st.number_input("Density", value=0.56)
pH = st.number_input("pH", value=0.49)
sulphates = st.number_input("Sulphates (log)", value=0.50)
alcohol = st.number_input("Alcohol", value=0.58)

# Combine inputs
input_data = np.array([[  
    fixed_acidity,
    volatile_acidity,
    citric_acid,
    residual_sugar,
    chlorides,
    free_sulfur_dioxide,
    total_sulfur_dioxide,
    density,
    pH,
    sulphates,
    alcohol
]])

# Scaling
input_data = scaler.transform(input_data)

# Prediction
if st.button("Predict Wine Quality"):
    prediction = model.predict(input_data)
    st.success(f"🍷 Predicted Wine Quality: {prediction[0]}")
