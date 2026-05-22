import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Load dataset
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
    <style>

    .main {
        background-color: #0B1120;
    }

    h1 {
        color: white;
        text-align: center;
    }

    .stButton>button {
        background-color: #2563EB;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 18px;
    }

    .stTextInput>div>div>input {
        background-color: #1E293B;
        color: white;
    }

    </style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("📊 Customer Churn Prediction System")

st.write("Enter Customer ID to predict churn probability.")

# ---------------- CUSTOMER INPUT ----------------
customer_id = st.text_input("Customer ID")

# ---------------- PREDICTION ----------------
if st.button("Predict Churn"):

    customer = df[df["customerID"] == customer_id]

    if customer.empty:

        st.error("Customer ID not found")

    else:

        st.subheader("Customer Information")

        st.write(customer)

        # Copy customer data
        input_data = customer.copy()

        # Drop customerID
        input_data.drop("customerID", axis=1, inplace=True)

        # Encode categorical columns
        from sklearn.preprocessing import LabelEncoder

        le = LabelEncoder()

        categorical_columns = [
            'gender', 'Partner', 'Dependents',
            'PhoneService', 'MultipleLines',
            'InternetService', 'OnlineSecurity',
            'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV',
            'StreamingMovies', 'Contract',
            'PaperlessBilling', 'PaymentMethod'
        ]

        for col in categorical_columns:
            input_data[col] = le.fit_transform(input_data[col])

        # Convert TotalCharges
        input_data['TotalCharges'] = input_data['TotalCharges'].replace(' ', np.nan)
        input_data['TotalCharges'] = pd.to_numeric(input_data['TotalCharges'])
        input_data['TotalCharges'].fillna(input_data['TotalCharges'].median(), inplace=True)

        # Drop target column if exists
        if 'Churn' in input_data.columns:
            input_data.drop("Churn", axis=1, inplace=True)

        # Scale columns
        input_data[['MonthlyCharges', 'TotalCharges', 'tenure']] = scaler.transform(
            input_data[['MonthlyCharges', 'TotalCharges', 'tenure']]
        )

        # Prediction
        prediction = model.predict(input_data)

        # Prediction probability
        probability = model.predict_proba(input_data)

        churn_probability = probability[0][1] * 100

        st.subheader("Prediction Result")

        if prediction[0] == 1:
            st.error(f"⚠ Customer likely to churn ({churn_probability:.2f}% probability)")
        else:
            st.success(f"✅ Customer likely to stay ({100 - churn_probability:.2f}% confidence)")