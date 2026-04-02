import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from preprocess import load_and_prepare_data

X_train, X_test, y_train, y_test = load_and_prepare_data()

model = LinearRegression()
model.fit(X_train, y_train)

preds = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))

joblib.dump(model, "models/linear.pkl")
print("Linear Regression RMSE:", rmse)
