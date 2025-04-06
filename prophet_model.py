import os
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

# 📌 Auto-detect current directory where this script lives
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def train_prophet(csv_path: str, city: str):
    df = pd.read_csv(csv_path)
    df = df[["date", "temperature"]].rename(columns={"date": "ds", "temperature": "y"})

    model = Prophet(daily_seasonality=True)
    model.fit(df)

    future = model.make_future_dataframe(periods=7)
    forecast = model.predict(future)

    # Save forecast CSV
    models_dir = os.path.join(BASE_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)

    output_path = os.path.join(models_dir, f"prophet_forecast_{city.lower()}.csv")
    forecast[["ds", "yhat"]].tail(7).to_csv(output_path, index=False)

    # Plot
    model.plot(forecast)
    plt.title(f"Prophet Forecast - {city.title()} (Next 7 Days)")
    plt.xlabel("Date")
    plt.ylabel("Temperature (°C)")
    plt.tight_layout()
    plt.show()

    print(f"[✓] Forecast saved to {output_path}")
    return forecast

if __name__ == "__main__":
    cities = ["toronto", "vancouver", "montreal", "newyork", "calgary", "ottawa"]
    for city in cities:
        csv_path = os.path.join(BASE_DIR, "data", f"{city}_weather_sample.csv")
        try:
            train_prophet(csv_path, city=city)
        except Exception as e:
            print(f"[❌] Failed to train Prophet model for {city}: {e}")
