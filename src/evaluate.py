import joblib
import numpy as np
from sklearn.metrics import mean_squared_error
from preprocess import load_and_prepare_data
from tensorflow.keras.models import load_model

# =========================
# Load data
# =========================
print("Loading data...")
X_train, X_test, y_train, y_test = load_and_prepare_data()

# =========================
# Evaluate ML models
# =========================
print("\nEvaluating ML models...")

models = ["linear", "random_forest", "xgboost"]

for name in models:
    model = joblib.load(f"models/{name}.pkl")
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print(f"{name} RMSE: {rmse:.2f}")

# =========================
# Evaluate LSTM model
# =========================
print("\nEvaluating LSTM...")

try:
    lstm_model = load_model("models/lstm_model.h5", compile=False)

    # reshape for LSTM
    X_test_lstm = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    preds_lstm = lstm_model.predict(X_test_lstm)
    rmse_lstm = np.sqrt(mean_squared_error(y_test, preds_lstm))

    print(f"lstm RMSE: {rmse_lstm:.2f}")

except Exception as e:
    print("LSTM evaluation failed:", e)
