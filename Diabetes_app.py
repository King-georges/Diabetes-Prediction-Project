import streamlit as st
import requests
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