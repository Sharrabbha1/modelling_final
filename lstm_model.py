import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import load_model

def train_lstm(csv_path: str, model_name="toronto_lstm"):
    df = pd.read_csv(csv_path)
    if "temperature" not in df.columns:
        raise ValueError("CSV must contain 'temperature' column.")

    data = df["temperature"].values.reshape(-1, 1)

    # Scale data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    # Prepare sequences
    X, y = [], []
    for i in range(10, len(scaled_data)):
        X.append(scaled_data[i-10:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Build model
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X.shape[1], 1)),
        LSTM(32),
        Dense(1)
    ])
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X, y, epochs=30, batch_size=16, verbose=1)

    # Save model
    os.makedirs("models", exist_ok=True)
    model.save(f"models/{model_name}.h5")
    print(f"[✓] LSTM model saved as models/{model_name}.h5")

    return model, scaler, data

def predict_next_days(model, scaler, last_data, days=7):
    forecast = []
    input_seq = last_data[-10:].reshape(1, 10, 1)

    for _ in range(days):
        pred = model.predict(input_seq)[0][0]
        forecast.append(pred)
        input_seq = np.append(input_seq[:, 1:, :], [[[pred]]], axis=1)

    return scaler.inverse_transform(np.array(forecast).reshape(-1, 1))

def plot_lstm_forecast(original, forecast):
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(original)), original, label="Historical")
    plt.plot(range(len(original), len(original) + len(forecast)), forecast, label="LSTM Forecast", color="green")
    plt.xlabel("Days")
    plt.ylabel("Temperature (°C)")
    plt.title("LSTM Temperature Forecast")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    model, scaler, data = train_lstm("data/toronto_weather_sample.csv")
    forecast = predict_next_days(model, scaler, scaler.transform(data))
    print(f"\n[📈] 7-Day LSTM Forecast:\n{forecast.flatten().tolist()}")
    plot_lstm_forecast(data, forecast)
