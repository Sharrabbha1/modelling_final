import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from datetime import timedelta

# ✅ Train ARIMA model and save it
def train_arima(csv_path: str, model_name: str):
    df = pd.read_csv(csv_path)

    if "temperature" not in df.columns or "date" not in df.columns:
        raise ValueError("CSV must contain 'date' and 'temperature' columns.")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    series = df["temperature"]
    dates = df["date"]

    model = ARIMA(series, order=(5, 1, 0))
    model_fit = model.fit()

    # ✅ Save model
    models_dir = os.path.join("models")
    os.makedirs(models_dir, exist_ok=True)

    model_path = os.path.join(models_dir, f"arima_model_{model_name.lower()}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model_fit, f)

    print(f"[✓] ARIMA model saved as {model_path}")
    return model_fit, series, dates

# ✅ Plot forecast
def plot_forecast(series, forecast, original_dates, city_name):
    forecast_dates = pd.date_range(start=original_dates.iloc[-1] + timedelta(days=1), periods=len(forecast))

    plt.figure(figsize=(10, 5))
    plt.plot(original_dates, series, label="Historical", color="blue")
    plt.plot(forecast_dates, forecast, label="Forecast", color="orange")
    plt.xlabel("Date")
    plt.ylabel("Temperature (°C)")
    plt.title(f"{city_name.title()} - 7-Day ARIMA Temperature Forecast")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ✅ Load model for API
def load_arima(city):
    city = city.lower()
    model_path = os.path.join("models", f"arima_model_{city}.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    with open(model_path, "rb") as f:
        return pickle.load(f)

# ✅ Manual training
if __name__ == "__main__":
    cities = ["toronto", "vancouver", "montreal", "newyork", "calgary", "ottawa"]
    for city in cities:
        csv_path = os.path.join("data", f"{city}_weather_sample.csv")
        try:
            model, series, dates = train_arima(csv_path, model_name=city)
            forecast = model.forecast(steps=7)
            print(f"\n[📈] 7-Day ARIMA Forecast for {city}:\n{forecast.tolist()}")
            plot_forecast(series, forecast, dates, city)
        except Exception as e:
            print(f"[❌] Failed to train ARIMA model for {city}: {e}")
