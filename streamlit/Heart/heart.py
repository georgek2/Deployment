import streamlit as st
import pandas as pd


data = pd.read_csv('heart.csv')

st.write('''
    # Heart Disease Analysis App..
    Predicting the risk of heart disease using some vital health signals

    Dataset used is from kaggle
''')

st.write(data.head(8), '''Data Labels

    Age : Age of the patient
    Sex : Sex of the patient
    exang: exercise induced angina (1 = yes; 0 = no)
    ca: number of major vessels (0-3)
    cp : Chest Pain type chest pain type
        Value 0: typical angina
        Value 1: atypical angina
        Value 2: non-anginal pain
        Value 3: asymptomatic
    trtbps : resting blood pressure (in mm Hg)
    chol : cholestoral in mg/dl fetched via BMI sensor
    fbs : (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
    rest_ecg : resting electrocardiographic results
        Value 0: normal
        Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
        Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria
    thalach : maximum heart rate achieved
    target : risk or chance of heart attack
        0 = less chance of heart attack
        1 = more chance of heart attack

''')














