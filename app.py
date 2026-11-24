import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer

# Cargar el modelo y el DictVectorizer
with open('churn-model.pck', 'rb') as f:
    dv, model = pickle.load(f)

# Título de la aplicación
st.title("Predicción de Churn de Clientes")

# Formulario para introducir datos del cliente
st.header("Introduce los datos del cliente:")

tenure = st.number_input("Antigüedad (meses)", min_value=0, max_value=100, step=1)
monthly_charges = st.number_input("Cargos mensuales", min_value=0.0, max_value=500.0, step=0.1)
total_charges = st.number_input("Cargos totales", min_value=0.0, max_value=10000.0, step=0.1)
contract = st.selectbox("Tipo de contrato", ["month-to-month", "one_year", "two_year"])
online_security = st.selectbox("Seguridad online", ["no", "yes", "no_internet_service"])
tech_support = st.selectbox("Soporte técnico", ["no", "yes", "no_internet_service"])
internet_service = st.selectbox("Tipo de Internet", ["dsl", "fiber_optic", "no"])
online_backup = st.selectbox("Copia de seguridad online", ["no", "yes", "no_internet_service"])
device_protection = st.selectbox("Protección de dispositivos", ["no", "yes", "no_internet_service"])
payment_method = st.selectbox("Método de pago", ["bank_transfer_(automatic)", "credit_card_(automatic)", "electronic_check", "mailed_check"])
streaming_tv = st.selectbox("Televisión en streaming", ["no", "yes", "no_internet_service"])

# Botón de predicción
if st.button("Predecir"):
    # Crear un diccionario con los datos del cliente
    client_data = {
        "tenure": tenure,
        "monthlycharges": monthly_charges,
        "totalcharges": total_charges,
        "contract": contract,
        "onlinesecurity": online_security,
        "techsupport": tech_support,
        "internetservice": internet_service,
        "onlinebackup": online_backup,
        "deviceprotection": device_protection,
        "paymentmethod": payment_method,
        "streamingtv": streaming_tv
    }

    # Transformar los datos del cliente
    X_client = dv.transform([client_data])

    # Realizar la predicción
    y_pred_proba = model.predict_proba(X_client)[0][1]  # Probabilidad de churn

    # Mostrar resultado
    st.subheader("Resultado:")
    if y_pred_proba > 0.5:
        st.error(f"El cliente tiene una alta probabilidad de churn: {y_pred_proba:.2f}")
    else:
        st.success(f"El cliente tiene una baja probabilidad de churn: {y_pred_proba:.2f}")
