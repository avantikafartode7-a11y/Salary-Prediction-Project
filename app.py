import streamlit as st
import pandas as pd
import pickle
import os

st.set_page_config(page_title="Salary Prediction App", layout="centered")

st.title("📊 Salary Prediction App")

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    with open("Student_model.pkl", "rb") as f:
        return pickle.load(f)

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("Employee_clean_Data.csv")

# -----------------------------
# Check files
# -----------------------------
if not os.path.exists("Student_model.pkl"):
    st.error("❌ Student_model.pkl not found in project folder")
    st.stop()

if not os.path.exists("Employee_clean_Data.csv"):
    st.error("❌ Employee_clean_Data.csv not found in project folder")
    st.stop()

model = load_model()
df = load_data()

st.subheader("📂 Dataset Preview")
st.dataframe(df.head())

# -----------------------------
# Detect feature columns
# -----------------------------
# Assume last column is TARGET
feature_columns = df.columns[:-1]
target_column = df.columns[-1]

st.subheader("🧮 Enter Input Values")

input_data = {}

for col in feature_columns:
    if df[col].dtype == "object":
        input_data[col] = st.selectbox(
            col,
            options=df[col].unique()
        )
    else:
        input_data[col] = st.number_input(
            col,
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
        st.error(f"❌ Prediction failed: {e}")
