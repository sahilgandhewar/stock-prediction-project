import pandas as pd
import numpy as np

def create_features(data, window_size=5):
    X = []
    y = []

    close = data["Close"].values

    for i in range(window_size, len(close)):
        X.append(close[i-window_size:i])
        y.append(close[i])

    return np.array(X), np.array(y)

def load_and_prepare_data():
    data = pd.read_csv("data/AAPL.csv")
    data = data.sort_values("Date")

    X, y = create_features(data)

    split = int(0.8 * len(X))

    X_train = X[:split]
    X_test = X[split:]
    y_train = y[:split]
    y_test = y[split:]

    # print training details
    total = len(X)
    train_percent = (len(X_train) / total) * 100
    test_percent = (len(X_test) / total) * 100

    print(f"Total samples: {total}")
    print(f"Training samples: {len(X_train)} ({train_percent:.2f}%)")
    print(f"Testing samples: {len(X_test)} ({test_percent:.2f}%)")

    return X_train, X_test, y_train, y_test

