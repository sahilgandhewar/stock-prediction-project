import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# Fix import paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from preprocess import load_and_prepare_data
from tensorflow.keras.models import load_model

# =========================
# Safe model loader
# =========================
def load_safe_model(path, model_type="sklearn"):
    full_path = os.path.join(BASE_DIR, "..", path)

    if not os.path.exists(full_path):
        st.warning(f"{path} not found. Using dummy model.")
        return None
    try:
        if model_type == "lstm":
            return load_model(full_path, compile=False)
        else:
            return joblib.load(full_path)
    except:
        st.warning(f"Error loading {path}")
        return None

# =========================
# Page title
# =========================
st.title("Stock Price Prediction Dashboard")
st.subheader("Prediction for AAPL")

# =========================
# Load data (FIXED PATH)
# =========================
data_path = os.path.join(BASE_DIR, "..", "data", "AAPL.csv")
data = pd.read_csv(data_path)
data = data.sort_values("Date")

# =========================
# Model selection
# =========================
st.subheader("Select Prediction Model")

model_choice = st.selectbox(
    "Choose a model",
    ["Linear Regression", "Random Forest", "XGBoost", "LSTM"]
)

# Load model safely
if model_choice == "Linear Regression":
    model = load_safe_model("models/linear.pkl")
elif model_choice == "Random Forest":
    model = load_safe_model("models/random_forest.pkl")
elif model_choice == "XGBoost":
    model = load_safe_model("models/xgboost.pkl")
else:
    model = load_safe_model("models/lstm_model.h5", "lstm")

# =========================
# Historical price graph
# =========================
st.subheader("Historical Closing Price")

plt.figure()
plt.plot(data["Close"])
plt.xlabel("Days")
plt.ylabel("Price")
plt.title("Stock Price History")
st.pyplot(plt)

# =========================
# Actual vs predicted graph
# =========================
st.subheader("Actual vs Predicted Prices")

X_train, X_test, y_train, y_test = load_and_prepare_data()

if model is not None:
    if model_choice == "LSTM":
        X_test_model = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        preds = model.predict(X_test_model)
    else:
        preds = model.predict(X_test)
else:
    preds = np.zeros(len(y_test))

plt.figure()
plt.plot(y_test, label="Actual")
plt.plot(preds, label="Predicted")
plt.legend()
plt.title(f"Actual vs Predicted ({model_choice})")
st.pyplot(plt)

# =========================
# Model comparison chart
# =========================
st.subheader("Model RMSE Comparison")

try:
    rmse_values = {}

    for name in ["linear", "random_forest", "xgboost"]:
        path = os.path.join(BASE_DIR, "..", "models", f"{name}.pkl")
        if os.path.exists(path):
            m = joblib.load(path)
            p = m.predict(X_test)
            rmse = np.sqrt(np.mean((y_test - p) ** 2))
            rmse_values[name] = rmse

    lstm_path = os.path.join(BASE_DIR, "..", "models", "lstm_model.h5")
    if os.path.exists(lstm_path):
        lstm_model = load_model(lstm_path, compile=False)
        X_test_lstm = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        p_lstm = lstm_model.predict(X_test_lstm)
        rmse_lstm = np.sqrt(np.mean((y_test - p_lstm.flatten()) ** 2))
        rmse_values["lstm"] = rmse_lstm

    if rmse_values:
        plt.figure()
        plt.bar(rmse_values.keys(), rmse_values.values())
        plt.ylabel("RMSE")
        plt.title("Model Comparison")
        st.pyplot(plt)
    else:
        st.warning("No models available for comparison.")

except:
    st.warning("Train all models to see comparison.")

# =========================
# Manual prediction
# =========================
st.subheader("Predict Next Day Price")

st.write("Enter last 5 days closing prices:")

inputs = []
for i in range(5):
    val = st.number_input(f"Day {i+1}", value=100.0)
    inputs.append(val)

if st.button("Predict"):
    arr = np.array(inputs).reshape(1, -1)

    if model is not None:
        if model_choice == "LSTM":
            arr = arr.reshape((1, 5, 1))
        prediction = model.predict(arr)
        pred_value = float(np.array(prediction).flatten()[0])
    else:
        pred_value = 0.0

    st.success(f"Predicted next day price: {pred_value:.2f}")