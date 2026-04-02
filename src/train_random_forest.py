import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from preprocess import load_and_prepare_data

X_train, X_test, y_train, y_test = load_and_prepare_data()

model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

preds = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))

joblib.dump(model, "models/random_forest.pkl")
print("Random Forest RMSE:", rmse)
