import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf

# -------------------------------------------------
# Load trained ANN model
# -------------------------------------------------
model = tf.keras.models.load_model("model.h5")

# -------------------------------------------------
# MANUAL PREPROCESSOR (replaces sklearn pipeline)
# -------------------------------------------------

# === IMPORTANT ===
# These values MUST come from TRAINING DATA
# Replace with your actual scaler values if different

NUMERICAL_MEAN = {
    "CreditScore": 650,
    "Age": 39,
    "Tenure": 5,
    "Balance": 75000,
    "NumOfProducts": 2,
    "EstimatedSalary": 100000
}

NUMERICAL_STD = {
    "CreditScore": 96,
    "Age": 10,
    "Tenure": 3,
    "Balance": 63000,
    "NumOfProducts": 1,
    "EstimatedSalary": 50000
}

def standard_scale(value, mean, std):
    return (value - mean) / std if std != 0 else 0

def preprocess_input(df):
    """
    Final feature order (must match training):
    [
      CreditScore,
      Age,
      Tenure,
      Balance,
      NumOfProducts,
      HasCrCard,
      IsActiveMember,
      EstimatedSalary,
      Gender,
      Geography_France,
      Geography_Germany,
      Geography_Spain
    ]
    """

    # -------- Handle Missing Values (Imputer) --------
    df = df.fillna({
        "CreditScore": NUMERICAL_MEAN["CreditScore"],
        "Age": NUMERICAL_MEAN["Age"],
        "Tenure": NUMERICAL_MEAN["Tenure"],
        "Balance": NUMERICAL_MEAN["Balance"],
        "NumOfProducts": NUMERICAL_MEAN["NumOfProducts"],
        "EstimatedSalary": NUMERICAL_MEAN["EstimatedSalary"],
        "Gender": "Male",
        "Geography": "France"
    })

    # -------- Scale Numerical Columns --------
    credit_score = standard_scale(df["CreditScore"].iloc[0],
                                  NUMERICAL_MEAN["CreditScore"],
                                  NUMERICAL_STD["CreditScore"])

    age = standard_scale(df["Age"].iloc[0],
                         NUMERICAL_MEAN["Age"],
                         NUMERICAL_STD["Age"])

    tenure = standard_scale(df["Tenure"].iloc[0],
                            NUMERICAL_MEAN["Tenure"],
                            NUMERICAL_STD["Tenure"])

    balance = standard_scale(df["Balance"].iloc[0],
                             NUMERICAL_MEAN["Balance"],
                             NUMERICAL_STD["Balance"])

    num_products = standard_scale(df["NumOfProducts"].iloc[0],
                                  NUMERICAL_MEAN["NumOfProducts"],
                                  NUMERICAL_STD["NumOfProducts"])

    salary = standard_scale(df["EstimatedSalary"].iloc[0],
                            NUMERICAL_MEAN["EstimatedSalary"],
                            NUMERICAL_STD["EstimatedSalary"])

    # -------- Binary Encoding --------
    gender = 1 if df["Gender"].iloc[0] == "Male" else 0
    has_cr_card = int(df["HasCrCard"].iloc[0])
    is_active = int(df["IsActiveMember"].iloc[0])

    # -------- One-Hot Encoding --------
    geo = df["Geography"].iloc[0]
    geo_france = 1 if geo == "France" else 0
    geo_germany = 1 if geo == "Germany" else 0
    geo_spain = 1 if geo == "Spain" else 0

    # -------- Final Feature Vector --------
    final_array = np.array([[
        credit_score,
        age,
        tenure,
        balance,
        num_products,
        has_cr_card,
        is_active,
        salary,
        gender,
        geo_france,
        geo_germany,
        geo_spain
    ]], dtype=np.float32)

    return final_array

# -------------------------------------------------
# Streamlit UI
# -------------------------------------------------
st.set_page_config(page_title="Customer Churn Prediction")
st.title("Customer Churn Prediction")

credit_score = st.number_input("Credit Score", 300, 900, 600)
geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 18, 92, 40)
tenure = st.slider("Tenure", 0, 10, 3)
balance = st.number_input("Balance", value=60000.0)
num_of_products = st.slider("Number of Products", 1, 4, 2)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])
estimated_salary = st.number_input("Estimated Salary", value=50000.0)

# -------------------------------------------------
# Input DataFrame
# -------------------------------------------------
input_data = pd.DataFrame({
    "CreditScore": [credit_score],
    "Geography": [geography],
    "Gender": [gender],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_of_products],
    "HasCrCard": [has_cr_card],
    "IsActiveMember": [is_active_member],
    "EstimatedSalary": [estimated_salary]
})

# -------------------------------------------------
# Prediction
# -------------------------------------------------
if st.button("Predict Churn"):
    processed_input = preprocess_input(input_data)
    prediction = model.predict(processed_input)
    probability = float(prediction[0][0])

    st.subheader(f"Churn Probability: {probability:.2f}")

    if probability > 0.5:
        st.error("The customer is likely to churn ❌")
    else:
        st.success("The customer is not likely to churn ✅")

