import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os

# --- Constants ---
MODEL_PATH = 'Linear_Regression pkl.pkl'
# Note: For deployment in an environment like Streamlit Cloud, the file
# must be in the same directory as the app.py or in a path accessible to it.

# --- Helper Functions ---
@st.cache_resource
def load_model(path):
    """Loads the pickled model object from the specified path."""
    if not os.path.exists(path):
        st.error(f"Model file not found at: {path}")
        return None
    try:
        with open(path, 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

# Load the model
model = load_model(MODEL_PATH)

# --- Streamlit UI and Logic ---
def main():
    st.set_page_config(
        page_title="Salary Prediction App",
        layout="centered",
        initial_sidebar_state="auto"
    )

    st.title("💰 Employee Salary Predictor")
    st.markdown("Use the controls below to input an employee's characteristics and predict their potential salary.")

    if model is None:
        st.warning("Cannot run prediction because the model failed to load. Please ensure 'Linear_Regression.pkl' is correctly uploaded.")
        return

    # --- Input Fields ---
    st.sidebar.header("Input Employee Data")

    # Experience (e.g., in years)
    experience = st.sidebar.slider(
        "Years of Experience",
        min_value=0,
        max_value=30,
        value=10,
        step=1
    )

    # Education Level (assuming 1=Low, 2=Medium, 3=High based on typical dataset encoding)
    education_map = {
        "1 - High School": 1,
        "2 - Bachelor's Degree": 2,
        "3 - Master's/PhD": 3
    }
    education_level_display = st.sidebar.selectbox(
        "Education Level",
        options=list(education_map.keys())
    )
    education_level = education_map[education_level_display]


    # Age
    age = st.sidebar.number_input(
        "Age",
        min_value=18,
        max_value=70,
        value=35,
        step=1
    )

    # --- Prediction Logic ---
    if st.button("Predict Salary"):
        # Create a DataFrame for the prediction
        # The column names MUST match the features the model was trained on
        input_data = pd.DataFrame({
            'Experience': [experience],
            'Education_Level': [education_level],
            'Age': [age]
        })

        try:
            # Make the prediction
            predicted_salary = model.predict(input_data)[0]

            st.success("Prediction Complete!")
            st.metric(
                label="Predicted Salary (Ruppes)",
                value=f"₹{predicted_salary:,.2f}"
            )

            st.balloons()
            st.markdown(f"""
                <p style='font-size:14px; color: marron;'>
                    *Prediction based on: {experience} years of experience, Education Level {education_level} ({education_level_display.split(' - ')[1]}), and Age {age}.
                </p>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.warning("Please check the model file and ensure the input types/structure match the training data.")

    st.sidebar.markdown("---")
    st.sidebar.info("Adjust the parameters to see how they impact the predicted salary.")


if _name_ == '_main_':
    main()
