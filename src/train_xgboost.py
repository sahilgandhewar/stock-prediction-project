import joblib
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from preprocess import load_and_prepare_data

print("Loading data...")
X_train, X_test, y_train, y_test = load_and_prepare_data()

print("Training XGBoost model...")
model = XGBRegressor(n_estimators=100)
model.fit(X_train, y_train)

preds = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))

joblib.dump(model, "models/xgboost.pkl")

print("XGBoost RMSE:", rmse)
print("Model saved to models/xgboost.pkl")
