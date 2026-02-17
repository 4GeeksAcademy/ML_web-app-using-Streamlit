# 💰 Income Prediction System

Aplicación web desarrollada con **Streamlit** que utiliza Machine Learning para predecir la probabilidad de que una persona gane más de 50K USD al año según su perfil socioeconómico.

---

##  Demo en vivo

https://ml-web-app-using-streamlit-1-jb63.onrender.com

---

##  Dataset

Se utilizó el **Adult Census Income Dataset**, que incluye variables como:

- Edad
- Nivel educativo
- Estado civil
- Ocupación
- Horas trabajadas por semana
- Sexo
- País de origen
- Ingreso anual (>50K / <=50K)

---

## Modelo de Machine Learning

Se entrenó un modelo de:

- **Logistic Regression**
- Pipeline con:
  - StandardScaler
  - OneHotEncoder
  - ColumnTransformer

El modelo fue serializado con `joblib` y posteriormente desplegado.

---

## 🖥️ Tecnologías utilizadas

- Python
- Pandas
- NumPy
- Scikit-Learn
- Joblib
- Streamlit
- Render (Deployment)

---

## Ejecución local

```bash
pip install -r requirements.txt
streamlit run src/app.py
