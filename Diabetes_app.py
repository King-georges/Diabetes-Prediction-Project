import streamlit as st
import pandas as pd
import pickle

# Set page configuration with the logo
st.set_page_config(
    page_title="Model Mavericks Analytics",
    page_icon='images/model mavericks.jpg',  # Correct path to your logo
    layout='wide'
)

# Display the logo at the top of your app
st.image('images/model mavericks.jpg', width=100)  # Adjust size as needed

# Load the pre-trained model
@st.cache_data
def load_model():
    # Assuming model.pkl is in the same directory as your script
    with open('MVP.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

def predict_diabetes(input_data, model):
    # Ensure input data features match those the model was trained with
    expected_features = [
        'BloodPressure', 'Cholesterol', 'Stroke', 'HeartDisease', 'DifficultyWalking',
        'BMI', 'GeneralHealth', 'MentalHealth', 'Age', 'PhysicalHealth'
    ]

    # Create DataFrame from input data ensuring the order of features
    input_df = pd.DataFrame([input_data], columns=expected_features)
    
    # Load the trained model (already cached)
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    return prediction[0], prediction_proba

def main():
    st.title("Diabetes Prediction App")
    st.sidebar.header("User Input Parameters")

    # Collecting inputs using sidebar
    blood_pressure = st.sidebar.radio("Blood Pressure", (1, 0), format_func=lambda x: "High BP" if x == 1 else "Normal BP")
    cholesterol = st.sidebar.radio("Cholesterol", (1, 0), format_func=lambda x: "High Cholesterol" if x == 1 else "Normal Cholesterol")
    stroke = st.sidebar.radio("Stroke", (1, 0), format_func=lambda x: "Had Stroke" if x == 1 else "Never had Stroke")
    heart_disease = st.sidebar.radio("Heart Disease or Attack", (1, 0), format_func=lambda x: "Heart Disease" if x == 1 else "No Heart Disease/Attack")
    difficulty_walking = st.sidebar.radio("Difficulty Walking", (1, 0), format_func=lambda x: "Difficulty Walking" if x == 1 else "No Difficulty Walking")

    age_options = [
        "18-24", "25-29", "30-34", "35-39", "40-44", "45-49",
        "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80 or older"
    ]
    age = st.sidebar.selectbox("Age Range", options=age_options)
    # Convert age range to class number
    age_class = age_options.index(age) + 1

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
        'Age': age_class,  # Using age class derived from the selected age range
        'PhysicalHealth': physical_health
    }

    # Make prediction
    model = load_model()
    prediction, prediction_proba = predict_diabetes(input_data, model)

    # Interpret prediction
    classes = ['Diabetes', 'Pre-Diabetes', 'No Diabetes']
    result_class = classes[int(prediction)]  # Convert prediction to integer for indexing
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

if __name__ == "__main__":
    main()
