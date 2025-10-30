# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 19:48:23 2025

@author: 91817
"""

import numpy as np
import pickle
import streamlit as st
from streamlit_option_menu import option_menu

diabetes_model = pickle.load(open("diabetes_trained_model.sav", 'rb'))

heart_model = pickle.load(open("heart_model.sav", 'rb'))

parkinsons_model = pickle.load(open("Parkinson_classifier.sav", 'rb'))


with st.sidebar:
    selected = option_menu('Multiple Disease Prediction',[
        'Diabetes Prediction', 
        'Heart Disease Prediction',
        'Parkinson Prediction'],
        
        icons = ['activity', 'heart', 'person'],
        
        default_index = 0)
    

if (selected == 'Diabetes Prediction'):
    
    st.title('Diabetes Prediction System')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Pregnancies = st.text_input('No. of Pregnancies')
    with col2:
        Glucose = st.text_input('Glucose Level')
    with col3:
        BloodPressure = st.text_input('Blood Pressure Level')
    with col1:
        SkinThickness = st.text_input('Skin Thickness')
    with col2:
        Insulin = st.text_input('Insulin Level')
    with col3:
        BMI = st.text_input('BMI level')
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function Value')
    with col2:        
        Age = st.text_input('Age')
    
    dia_diagnosis = ''
    
    if st.button("Diabetes Test Result"):
        diabetes_prediction = diabetes_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        
        if(diabetes_prediction == 0):
            dia_diagnosis = 'The Patient is Healthy'
        else:
            dia_diagnosis = 'Patient has Diabetes'
        
    st.success(dia_diagnosis)
    
    

if (selected == 'Heart Disease Prediction'):
    
    st.title('Heart Disease Prediction System')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.text_input('Age')
    with col2:
        sex = st.text_input('Gender M-->1, F-->0')
    with col3:
        cp = st.text_input('Chest Pain Type')
    with col1:
        trestbps = st.text_input('Resting BP value')
    with col2:
        chol = st.text_input('Cholestrol Level')
    with col3:
        fbs = st.text_input('Fasting Blood Sugar')
    with col1:
        restecg = st.text_input('Resting ECG Value')
    with col2:
        thalach = st.text_input('Max heart rate achieved')
    with col3:
        exang = st.text_input('Exercise induced angina')
    with col1:
        oldpeak = st.text_input('ST Depression')
    with col2:
        slope = st.text_input('Slope of Peak Exercise')
    with col3:
        ca = st.text_input('No. of major vessels')
    with col1:
        thal = st.text_input('Thallium Stress Test')
    
    # Convert all inputs to float (or int)
    features = [float(age), float(sex), float(cp), float(trestbps), float(chol),
            float(fbs), float(restecg), float(thalach), float(exang),
            float(oldpeak), float(slope), float(ca), float(thal)]

# Make prediction

    heart_diag = ''
    
    if st.button("Heart Test Result"):
        heart_result = heart_model.predict([features])
        
        if heart_result == 1:
            heart_diag = 'Patient has Heart Disease'
        else:
            heart_diag = 'Patient is Healthy'
    st.success(heart_diag)
    
if (selected == 'Parkinson Prediction'):
    
    st.title('Parkinson Prediciton System')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        MDVPFo = st.text_input('MDVP:Fo Average Vocal Freq')
    with col2:
        MDVPFhi = st.text_input('MDVP:Fhi Maximum Vocal Freq')
    with col3:
        MDVPFlo = st.text_input('MDVP:Flo Minimum Vocal Freq')
    with col1:
      MDVPJitter = st.text_input('MDVP: Jitter Percentage Variation')
    with col2:
      MDVPJitterABS = st.text_input('MDVP:JitterABS Absolute Variation')
    with col3:
      MDVPRAP = st.text_input('MDVP:RAP Relative Amplitude Perturbation')
    with col1:
      MDVPPPQ = st.text_input('MDVP:PPQ 5 Point Period Perturbation')
    with col2:
      JitterDDP = st.text_input('JitterDDP Avg Absolute Difference')
    with col3:
      MDVPShimmer = st.text_input('MDVP:Shimmer % of Variation')
    with col1:
      MDVPShimmerDB = st.text_input('MDVP:ShimmerDB Variation in DB')
    with col2:
      ShimmerAPQ3 = st.text_input('Shimmer:APQ3 3 Point Amp Perturbation')
    with col3:
      ShimmerAPQ5 = st.text_input('Shimmer:APQ5 5 Point Amp Perturbation')
    with col1:
      MDVPAPQ = st.text_input('MDVP:APQ Avf Amplitude Perturbation')
    with col2:
      ShimmerDDA = st.text_input('Shimmer:DDA Avg Absolute Difference')
    with col3:
      NHR = st.text_input('NHR Noise to Harmonics Ratio')
    with col1:
      HNR = st.text_input('HNR Harmonics to Noise Ratio')
    with col2:
      RPDE = st.text_input('RPDE Recurrence Period Density')
    with col3:
      DFA = st.text_input('DFA Detrended Fluctuation Analysis')
    with col1:
      spread1 = st.text_input('Spread1 Non linear measure of Fund Freq')
    with col2:
      spread2 = st.text_input('Spread2 Non linear measure of Fund Freq')
    with col3:
      D2 = st.text_input('D2 Correlation Dimension')
    with col1:
      PPE = st.text_input('PPE Pitch Period Entropy')
      
    park_features = [MDVPFo, MDVPFhi, MDVPFlo, MDVPJitter, MDVPJitterABS, MDVPRAP, MDVPPPQ, JitterDDP, MDVPShimmer, MDVPShimmerDB, ShimmerAPQ3, ShimmerAPQ5, MDVPAPQ, ShimmerDDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]
    
    park_diag = ''
    
    if st.button('Parkinsons Result'):
        park_result = parkinsons_model.predict([park_features])
        
        if(park_result == 1):
            park_diag = 'Parkinson Detected'
        else:
            park_diag = 'Patient is Healthy'

    st.success(park_diag)        
    
    