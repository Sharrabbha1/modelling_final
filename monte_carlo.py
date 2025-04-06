import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ✅ Generates a 7-day forecast for a given city
def run_monte_carlo_simulation(city: str, num_simulations=1000, forecast_days=7):
    file_path = os.path.join("data", f"{city}_weather_sample.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"[❌] Data file not found: {file_path}")

    df = pd.read_csv(file_path)
    if "temperature" not in df.columns:
        raise ValueError("CSV must contain 'temperature' column.")

    historical = df["temperature"].values
    mean_temp = np.mean(historical)
    std_temp = np.std(historical)

    simulations = []
    for _ in range(num_simulations):
        simulated = np.random.normal(loc=mean_temp, scale=std_temp, size=forecast_days)
        simulations.append(simulated)

    simulations = np.array(simulations)
    forecast = np.mean(simulations, axis=0)
    return forecast.tolist()

# ✅ Optional: Visualization for one city
def plot_monte_carlo(series, lower, upper, city_name):
    plt.figure(figsize=(10, 5))
    plt.plot(series, label="Historical")
    plt.fill_between(range(len(series)), lower, upper, color="red", alpha=0.3, label="Confidence Interval (5–95%)")
    plt.title(f"{city_name.title()} - Monte Carlo Simulation")
    plt.xlabel("Days")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    plt.tight_layout()
    plt.show()

# ✅ Run simulation + plot for all cities
if __name__ == "__main__":
    cities = ["toronto", "vancouver", "montreal", "newyork", "calgary", "ottawa"]

    for city in cities:
        try:
            file_path = os.path.join("data", f"{city}_weather_sample.csv")
            df = pd.read_csv(file_path)
            series = df["temperature"].values

            simulations = []
            for _ in range(1000):
                simulated = series + np.random.normal(0, np.std(series), size=len(series))
                simulations.append(simulated)

            simulations = np.array(simulations)
            lower = np.percentile(simulations, 5, axis=0)
            upper = np.percentile(simulations, 95, axis=0)

            print(f"[📈] Simulation completed for {city.title()}")
            plot_monte_carlo(series, lower, upper, city)

        except Exception as e:
            print(f"[❌] Failed to simulate for {city}: {e}")
