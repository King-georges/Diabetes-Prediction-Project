import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import streamlit as st

# Set page configuration with the logo
st.set_page_config(
    page_title="Model Mavericks Analytics",
    page_icon='images/model mavericks.jpg',  # Correct path to your logo
    layout='wide'
)

# Define the primary colors from the logo
primary_color = "#1a73e8"  # Example primary blue shade
background_color = "##1a73e8"  # White background
secondary_background_color = "#1a73e8"  # Lighter blue shade
text_color = "#000000"  # Black text for better readability

# Use custom styles to apply color scheme via a CSS block in Markdown
css = f"""
<style>
    /* Main content area */
    .main .block-container{{
        background-color: {secondary_background_color};
        color: {text_color};
    }}
    /* Sidebar styling */
    .sidebar .sidebar-content {{
        background-color: {background_color};
        color: {text_color};
    }}
    /* Button styling */
    button {{
        background-color: {background_color};
        color: {background_color};
    }}
</style>
"""

st.markdown(css, unsafe_allow_html=True)

# Display the logo at the top of your app
st.image('images/model mavericks.webp', width=100)  # Adjust size as needed

# Your app content
st.title("Welcome to Model Mavericks Analytics")
st.header("Data-driven insights with a stylish interface.")

# Function to preprocess input data
def preprocess_data(df):
    # Here, you may want to encode categorical variables if they are not already numeric
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

def main():
    st.title("Diabetes Prediction App")

    # Collect user input on the sidebar
    st.sidebar.header("User Input Parameters")
    
    # Categorical Inputs
    blood_pressure = st.sidebar.radio("Blood Pressure", (1, 0), format_func=lambda x: "High BP" if x == 1 else "Normal BP")
    cholesterol = st.sidebar.radio("Cholesterol", (1, 0), format_func=lambda x: "High Cholesterol" if x == 1 else "Normal Cholesterol")
    stroke = st.sidebar.radio("Stroke", (1, 0), format_func=lambda x: "Had Stroke" if x == 1 else "Never had Stroke")
    heart_disease = st.sidebar.radio("Heart Disease or Attack", (1, 0), format_func=lambda x: "Heart Disease" if x == 1 else "No Heart Disease/Attack")
    difficulty_walking = st.sidebar.radio("Difficulty Walking", (1, 0), format_func=lambda x: "Difficulty Walking" if x == 1 else "No Difficulty Walking")

    # Numerical and Categorical Input Using Dropdown
    age_options = ["18-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", 
                   "60-64", "65-69", "70-74", "75-79", "80 or older"]
    age = st.sidebar.selectbox("Age Range", options=age_options)

    bmi = st.sidebar.slider("BMI", min_value=0, max_value=100, value=25)
    general_health = st.sidebar.slider("General Health", min_value=1, max_value=5, value=3)
    mental_health = st.sidebar.slider("Mental Health Days", min_value=0, max_value=30, value=2)
    physical_health = st.sidebar.slider("Physical Health Days", min_value=0, max_value=30, value=1)

    input_data = {
        'BloodPressure': blood_pressure,
        'Cholesterol': cholesterol,
        'Stroke': stroke,
        'HeartDisease': heart_disease,
        'DifficultyWalking': difficulty_walking,
        'BMI': bmi,
        'GeneralHealth': general_health,
        'MentalHealth': mental_health,
        'Age': age,
        'PhysicalHealth': physical_health
    }

    # Make prediction
    model = load_model()
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
    else:
        st.write("You are likely to have diabetes. Please seek medical attention.")

if _name_ == "_main_":
    main()