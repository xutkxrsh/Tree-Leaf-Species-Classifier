import streamlit as st
import numpy as np
import pickle

# Load model + scaler
model = pickle.load(open("knn_model.pkl","rb"))
scaler = pickle.load(open("scaler.pkl","rb"))

st.title("Tree Leaf Species Classifier")

length = st.number_input("Leaf Length (cm)", min_value=0.0)
width = st.number_input("Leaf Width (cm)", min_value=0.0)
intensity = st.number_input("Color Intensity", min_value=0.0)

if st.button("Predict Species"):
    sample = np.array([[length, width, intensity]])
    sample_scaled = scaler.transform(sample)
    pred = model.predict(sample_scaled)[0]   # THIS WILL BE THE NAME
    st.success(f"Predicted Species: {pred}")
