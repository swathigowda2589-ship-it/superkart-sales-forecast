import os
import streamlit as st
import pandas as pd
import requests

# Base URL of the Flask backend. "backend" is the container name on the Docker network;
# it can be overridden with the BACKEND_URL environment variable (e.g. for local testing).
BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:7860")

st.set_page_config(page_title="SuperKart Sales Forecast", page_icon="🛒", layout="centered")

# Page title
st.title("SuperKart Sales Forecast")
st.write("Enter the product and store details below to predict the total sales.")

# Input fields for product and store data
col1, col2 = st.columns(2)
with col1:
    Product_Weight = st.number_input("Product Weight", min_value=0.0, value=12.66)
    Product_Sugar_Content = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
    Product_Allocated_Area = st.number_input("Product Allocated Area", min_value=0.0, max_value=1.0, value=0.027, format="%.3f")
    Product_MRP = st.number_input("Product MRP", min_value=0.0, value=117.08)
    Product_Id_char = st.selectbox("Product ID Prefix", ["FD", "DR", "NC"])
with col2:
    Store_Size = st.selectbox("Store Size", ["Small", "Medium", "High"], index=1)
    Store_Location_City_Type = st.selectbox("Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"], index=1)
    Store_Type = st.selectbox("Store Type", ["Supermarket Type1", "Supermarket Type2", "Departmental Store", "Food Mart"], index=1)
    Store_Age_Years = st.number_input("Store Age (Years)", min_value=0, value=16)
    Product_Type_Category = st.selectbox("Product Type Category", ["Perishables", "Non Perishables"], index=1)

# Create JSON payload
product_data = {
    "Product_Weight": Product_Weight,
    "Product_Sugar_Content": Product_Sugar_Content,
    "Product_Allocated_Area": Product_Allocated_Area,
    "Product_MRP": Product_MRP,
    "Store_Size": Store_Size,
    "Store_Location_City_Type": Store_Location_City_Type,
    "Store_Type": Store_Type,
    "Product_Id_char": Product_Id_char,
    "Store_Age_Years": Store_Age_Years,
    "Product_Type_Category": Product_Type_Category
}

# Single Prediction
if st.button("Predict", type='primary'):
    try:
        response = requests.post(f"{BACKEND_URL}/v1/predict", json=product_data, timeout=30)
        if response.status_code == 200:
            predicted_sales = response.json()["Sales"]
            st.success(f"Predicted Product Store Sales Total: ₹{predicted_sales:,.2f}")
            with st.expander("Request payload / API response"):
                st.json({"request": product_data, "response": response.json()})
        else:
            st.error(f"API error ({response.status_code}): {response.json().get('error', response.text)}")
    except requests.exceptions.RequestException as e:
        st.error(f"Unable to connect to the prediction API at {BACKEND_URL}: {e}")

# Batch Prediction
st.subheader("Batch Prediction")
st.caption("CSV columns: " + ", ".join(product_data.keys()))

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    if st.button("Predict for Batch", type='primary'):
        try:
            response = requests.post(
                f"{BACKEND_URL}/v1/predictbatch",
                files={"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")},
                timeout=60,
            )
            if response.status_code == 200:
                results = response.json()
                st.success(f"Predictions completed successfully for {len(results)} records!")

                # Attach predictions to the uploaded rows so each forecast is traceable
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file)
                df.insert(0, "Predicted_Sales", [results[str(i)] for i in range(len(df))])
                st.dataframe(df, use_container_width=True)
                st.metric("Total predicted sales (batch)", f"₹{df['Predicted_Sales'].sum():,.2f}")

                st.download_button(
                    "Download predictions as CSV",
                    df.to_csv(index=False).encode("utf-8"),
                    file_name="superkart_batch_predictions.csv",
                    mime="text/csv",
                )
            else:
                st.error(f"API error ({response.status_code}): {response.json().get('error', response.text)}")
        except requests.exceptions.RequestException as e:
            st.error(f"Unable to connect to the prediction API at {BACKEND_URL}: {e}")
