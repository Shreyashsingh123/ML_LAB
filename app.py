import pickle
import streamlit as st
import numpy as np
import os

# Load the logistic regression model
model_path = os.path.join(os.getcwd(), 'model_diabetes.sav')
model_logistic = pickle.load(open(model_path, 'rb'))

# Set page configuration
st.set_page_config(page_title="Diabetes Prediction", layout="wide")

# Main Header
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Diabetes Prediction Application</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Predict the likelihood of diabetes using health parameters.</p>", unsafe_allow_html=True)

# Tabs for navigation
tab1, tab2, tab3 = st.tabs(["🏥 Patient Information", "📊 Prediction", "ℹ️ About"])

# Tab 1: Patient Information
with tab1:
    st.markdown("<h2 style='color: #2196F3;'>Enter Patient Details</h2>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        Pregnancies = st.slider('Number of Pregnancies', 0, 20, 0)
        Glucose = st.slider('Glucose Level (mg/dL)', 0, 300, 100)
        BloodPressure = st.slider('Blood Pressure (mm Hg)', 0, 200, 80)
        SkinThickness = st.slider('Skin Thickness (mm)', 0, 100, 20)

    with col2:
        Insulin = st.slider('Insulin Level (mu U/mL)', 0.0, 500.0, 100.0, step=1.0)
        BMI = st.slider('BMI (Body Mass Index)', 0.0, 70.0, 25.0, step=0.1)
        DiabetesPedigreeFunction = st.slider('Diabetes Pedigree Function', 0.0, 2.5, 0.5, step=0.01)
        Age = st.slider('Age', 0, 120, 25)

# Tab 2: Prediction
with tab2:
    st.markdown("<h2 style='color: #FF5722;'>Prediction Results</h2>", unsafe_allow_html=True)
    
    st.markdown("<p>Click the button below to predict whether the patient is likely to have diabetes.</p>", unsafe_allow_html=True)
    
    if st.button("Predict Diabetes"):
        # Prepare input data
        input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        
        try:
            diabetes_prediction = model_logistic.predict(input_data)
            if diabetes_prediction[0] == 1:
                st.error("The patient is **likely to have diabetes**.", icon="🚨")
            else:
                st.success("The patient is **unlikely to have diabetes**.", icon="✅")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}", icon="⚠️")

# Tab 3: About
with tab3:
    st.markdown("<h2 style='color: #9C27B0;'>About the Application</h2>", unsafe_allow_html=True)
    st.markdown("""
        This application uses health parameters to predict the likelihood of diabetes based on a logistic regression model.
        ### Parameters:
        - **Pregnancies**: Number of pregnancies.
        - **Glucose Level**: Blood sugar level.
        - **Blood Pressure**: Blood pressure level.
        - **Skin Thickness**: Thickness of the skinfold.
        - **Insulin Level**: Insulin concentration.
        - **BMI**: Body mass index.
        - **Diabetes Pedigree Function**: Family history of diabetes.
        - **Age**: Patient's age.

        ### Models:
        Currently, the application uses a Logistic Regression model. Additional models may be added in future updates.

        **Developed by**: Your Name  
        **GitHub**: [your-github-link](https://github.com)
    """, unsafe_allow_html=True)
