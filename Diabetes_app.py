import streamlit as st
import hashlib 
from streamlit_lottie import st_lottie
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="Diabetes Prediction App",
                page_icon= ":bar_chart:")
st.title('Diabetes Prediction app')
st.write("""
## This app predicts the likelihood of having Pre-Diabetes and Diabetes.

Data obtained from Kaggle and is used to predict Diabetes. Patients were classified as no diabetes, pr-diabetes or having diabetes.
https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset 
         
It is most suitable for medical experts with the necessary domain knowledge on Diabetes
""")
st.write('---')
def load_lottieurl(url):
     r = requests.get(url)
     if r.status_code != 200:
          return None
     return r.json()

import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Function to preprocess input data
def preprocess_data(df):
    # Perform any necessary preprocessing (e.g., encoding categorical variables)
    # For simplicity, we'll assume all data is numeric and ready to use
    return df

# Load the pre-trained model
@st.cache
def load_model():
    model = RandomForestClassifier()  # You can replace this with your trained model
    return model

def predict_diabetes(input_data, model):
    # Preprocess input data
    input_df = pd.DataFrame([input_data])
    input_df = preprocess_data(input_df)

    # Load the trained model
    model = load_model()

    # Make predictions
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    return prediction[0], prediction_proba

# Define the Streamlit app
def main():
    st.title("Diabetes Prediction App")

    # Collect user input
    st.sidebar.header("User Input")
    pregnancies = st.sidebar.number_input("Number of Pregnancies", min_value=0, max_value=17, value=0)
    cholesterol = st.sidebar.selectbox(
    "Select cholesterol level:",
    ('High', 'Low')
    )
    blood_pressure = st.sidebar.slider("Blood Pressure", min_value=0, max_value=1, value=1)
    skin_thickness = st.sidebar.slider("Skin Thickness (mm)", min_value=0, max_value=99, value=20)
    insulin = st.sidebar.slider("Insulin (mu U/ml)", min_value=0, max_value=846, value=79)
    bmi = st.sidebar.slider("BMI", min_value=0.0, max_value=67.1, value=25.0)
    age = st.sidebar.slider("Age (years)", min_value=0, max_value=120, value=25)


    # Make prediction
prediction, prediction_proba = predict_diabetes(input_data, model)

    # Interpret prediction
classes = ['No Diabetes', 'Pre-Diabetes', 'Diabetes']
result_class = classes[prediction]
st.write(f"Prediction: {result_class}")

    # Display prediction probabilities
st.write("Prediction Probabilities:")
for i, prob in enumerate(prediction_proba[0]):
        st.write(f"{classes[i]}: {prob:.2f}")

    # Provide recommendations based on prediction
if result_class == 'No Diabetes':
        st.write("You are healthy!")
elif result_class == 'Pre-Diabetes':
        st.write("You are likely to have pre-diabetes. We recommend some lifestyle changes.")
        # Provide lifestyle change recommendations for pre-diabetes
else:
        st.write("You are likely to have diabetes. Please seek medical attention.")

if __name__ == "__main__":
    main()
