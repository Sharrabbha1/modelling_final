import os
import requests
import pandas as pd

API_KEY = "b3b42ea7d81a418099c6bc908719ebd0"  

def fetch_weather_data(city):
    url = "https://api.openweathermap.org/data/2.5/weather?q=Toronto&appid=b3b42ea7d81a418099c6bc908719ebd0&units=metric"
    response = requests.get(url)

    if response.status_code != 200:
        print(f"[✗] Error fetching data for {city}: {response.status_code}")
        print(response.json())
        return None

    return response.json()

def save_weather_data(city):
    os.makedirs("data", exist_ok=True)

    data = fetch_weather_data(city)
    if not data or "main" not in data or "weather" not in data:
        print(f"[✗] Incomplete data received for {city}. Skipping save.")
        return

    processed = {
        "city": city,
        "temperature": data["main"]["temp"],
        "feels_like": data["main"]["feels_like"],
        "humidity": data["main"]["humidity"],
        "pressure": data["main"]["pressure"],
        "wind_speed": data["wind"]["speed"],
        "weather": data["weather"][0]["description"]
    }

    df = pd.DataFrame([processed])
    df.to_csv(f"data/{city}_weather.csv", index=False)
    print(f"[✓] Saved weather data to data/{city}_weather.csv")

if __name__ == "__main__":
    save_weather_data("Toronto")
