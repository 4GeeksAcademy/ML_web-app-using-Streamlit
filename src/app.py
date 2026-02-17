import streamlit as st
import pandas as pd
import joblib
import os

# ----------- Cargar modelo dinámicamente -----------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "adult_income_model.pkl")

model = joblib.load(MODEL_PATH)

# ----------- Interfaz -----------

st.set_page_config(page_title="Income Prediction App", page_icon="💰")

st.title("💰 Income Prediction System")
st.write("Predicción de probabilidad de ganar más de 50K USD anuales.")

# ----------- Inputs del usuario -----------

age = st.slider("Edad", 18, 70, 25)

education = st.selectbox("Nivel educativo", [
    "HS-grad", "Some-college", "Bachelors", "Masters", "Doctorate"
])

marital_status = st.selectbox("Estado civil", [
    "Never-married", "Married-civ-spouse", "Divorced", "Separated", "Widowed"
])

occupation = st.selectbox("Ocupación", [
    "Sales", "Tech-support", "Exec-managerial",
    "Prof-specialty", "Adm-clerical", "Machine-op-inspct"
])

hours = st.slider("Horas trabajadas por semana", 1, 80, 40)

sex = st.selectbox("Sexo", ["Male", "Female"])

country = st.selectbox("País de origen", ["United-States"])

# ----------- Crear DataFrame compatible -----------

input_data = pd.DataFrame([{
    "age": age,
    "workclass": "Private",
    "fnlwgt": 200000,
    "education": education,
    "education.num": 9,
    "marital.status": marital_status,
    "occupation": occupation,
    "relationship": "Not-in-family",
    "race": "White",
    "sex": sex,
    "capital.gain": 0,
    "capital.loss": 0,
    "hours.per.week": hours,
    "native.country": country
}])

# ----------- Predicción -----------

if st.button("Predecir ingreso"):

    prob = model.predict_proba(input_data)[0][1]

    st.subheader("Resultado")

    if prob > 0.5:
        st.success(f"Alta probabilidad de superar 50K: {prob:.2%}")
    else:
        st.warning(f"Baja probabilidad de superar 50K: {prob:.2%}")
        st.info("Recomendación: aumentar nivel educativo o trabajar más horas.")
