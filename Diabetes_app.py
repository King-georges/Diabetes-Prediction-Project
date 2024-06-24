import streamlit as st
import pandas as pd
import pickle
import os
import lzma
from streamlit_lottie import st_lottie
import requests
import gc

# Set page configuration with the logo
st.set_page_config(
    page_title="Model Mavericks Analytics",
    page_icon='images/model_mavericks.jpg',  # Ensure the correct path to your logo
    layout='wide'
)

st.title("Diabetes Prediction App")

def load_lottieurl(url):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception as e:
        st.error(f"Error loading lottie animation: {e}")
        return None

lottie_coding_url = "https://lottie.host/57fc3155-e3dc-4611-b50a-ccd5a58291e0/84oE4KwywI.json"
lottie_coding = load_lottieurl(lottie_coding_url)
if lottie_coding:
    st_lottie(lottie_coding, height=300, width=400, key="health")

st.write('---')

left_column, right_column = st.columns(2)
with left_column:
    def load_model():
        file_path = 'MVP.pkl.xz'

        if not os.path.isfile(file_path):
            st.error(f"Model file not found at {file_path}. Please ensure 'MVP.pkl.xz' is in the correct location.")
            return None

        try:
            with lzma.open(file_path, 'rb') as file:
                model = pickle.load(file)
            return model
        except Exception as e:
            st.error(f"An error occurred while loading the model: {e}")
            return None

def predict_diabetes(input_data, model):
    expected_features = [
        'BloodPressure', 'Cholesterol', 'Stroke', 'HeartDisease', 'DifficultyWalking',
        'BMI', 'GeneralHealth', 'MentalHealth', 'Age', 'PhysicalHealth'
    ]
    input_df = pd.DataFrame([input_data], columns=expected_features)
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)
    return prediction[0], prediction_proba

def main():
    st.sidebar.header("User Input Parameters")

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
        'Age': age_class,
        'PhysicalHealth': physical_health
    }

    # Load the model only when needed
    model = load_model()
    if model:
        prediction, prediction_proba = predict_diabetes(input_data, model)
        classes = ['Diabetes', 'Pre-Diabetes', 'No Diabetes']
        result_class = classes[int(prediction)]
        st.write(f"Prediction: {result_class}")

        st.write("Prediction Probabilities:")
        for i, prob in enumerate(prediction_proba[0]):
            st.write(f"{classes[i]}: {prob * 100:.2f}%")

        if result_class == 'No Diabetes':
            st.write("You are healthy!")
        elif result_class == 'Pre-Diabetes':
            st.write("You are likely to have pre-diabetes. We recommend some [lifestyle changes.](https://www.hopkinsmedicine.org/health/wellness-and-prevention/prediabetes-diet#:~:text=Stay%20active,aim%20for%2010%2C000%20daily%20steps)")
        else:
            st.write("You are likely to have diabetes. Please seek medical attention.")

        # Clean up memory
        del model
        gc.collect()
    else:
        st.write("Model could not be loaded. Please check the error messages above.")

if __name__ == "__main__":
    main()

with right_column:
    lottie_coding_url_2 = "https://lottie.host/9a509219-e153-4a7b-a42e-06652ea04e9e/eewuCHHb79.json"
    lottie_coding_2 = load_lottieurl(lottie_coding_url_2)
    if lottie_coding_2:
        st_lottie(lottie_coding_2, height=300, width=400, key="health2")

st.write('---')
st.markdown('Data obtained from [Kaggle](https://www.kaggle.com/code/nanda107/diabetes) and is used to predict Diabetes.')

st.write("""
## This app predicts the likelihood of having Diabetes.

Disclaimer:
This web application provides predictions for diabetes risk based on statistical analysis and machine learning algorithms; however, it is not a substitute for professional medical advice, and users should consult with a healthcare provider for any medical concerns.

Patients were classified as having Pre-diabetes, Diabetes or not having diabetes.

The primary stakeholders for this diabetes prediction project include healthcare providers, public health organizations, and individual patients concerned about their diabetes risk. 
""")

st.write('---')

def local_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error loading CSS file: {e}")

local_css("Style/style.css")

with st.container():
    st.write("---")
    st.header("Get in touch with us!")
    st.write("##")
    contact_form = '''
    <input type="hidden" name="_captcha" value="false">
    <form action="https://formsubmit.co/opondigeorge@gmail.com" method="POST">
        <input type="text" name="name" placeholder="Your name" required>
        <input type="email" name="email" placeholder="Your email" required>
        <textarea name="Message" placeholder="Your message here" required></textarea>
        <button type="submit">Send</button>
    </form>
    '''
    lottie_animation_url = "https://lottie.host/87b1eda6-f65b-47f0-b06a-fa93ee6f73ba/bNjIHSLbzV.json"
    lottie_animation = load_lottieurl(lottie_animation_url)
    left_column, right_column = st.columns(2)
    with left_column:
        st.markdown(contact_form, unsafe_allow_html=True)
    with right_column:
        if lottie_animation:
            st_lottie(lottie_animation, height=300, width=400, key="message")

# Final memory cleanup
gc.collect()
