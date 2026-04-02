import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
from preprocess import load_and_prepare_data

X_train, X_test, y_train, y_test = load_and_prepare_data()

# reshape for LSTM
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

model = Sequential()
model.add(LSTM(50, activation="relu", input_shape=(X_train.shape[1], 1)))
model.add(Dense(1))

model.compile(optimizer="adam", loss="mse")
model.fit(X_train, y_train, epochs=10, batch_size=32)

preds = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))

model.save("models/lstm_model.h5")
print("LSTM RMSE:", rmse)
