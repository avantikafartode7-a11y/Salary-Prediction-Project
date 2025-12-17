import streamlit as st
import pandas as pd
import pickle
import os

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Salary Prediction Project", layout="centered")
st.title("📊 Salary Prediction App")

# -----------------------------
# File Checks
# -----------------------------
MODEL_FILE = "Linear Regression.pkl"
DATA_FILE = "linear_regression_dataset.csv"

if not os.path.exists(MODEL_FILE):
    st.error("❌ Linear Regression.pkl not found. Keep it in the same folder as app.py")
    st.stop()

if not os.path.exists(DATA_FILE):
    st.error("❌ linear_regression_dataset.csv not found. Keep it in the same folder as app.py")
    st.stop()

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    with open(MODEL_FILE, "rb") as f:
        return pickle.load(f)

model = load_model()

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv(DATA_FILE)

df = load_data()

st.subheader("📂 Dataset Preview")
st.dataframe(df.head())

# -----------------------------
# Feature & Target Detection
# -----------------------------
# Assumption: last column is target
feature_columns = df.columns[:-1]
target_column = df.columns[-1]

st.subheader("🧮 Enter Input Values")

input_data = {}

for col in feature_columns:
    if df[col].dtype == "object":
        input_data[col] = st.selectbox(
            f"{col}",
            df[col].unique()
        )
    else:
        input_data[col] = st.number_input(
            f"{col}",
            value=float(df[col].mean())
        )

input_df = pd.DataFrame([input_data])

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):
    try:
        prediction = model.predict(input_df)[0]
        st.success(f"🎯 Predicted {target_column}: **{prediction}**")
    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
