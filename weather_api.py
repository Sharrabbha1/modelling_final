from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from arima_model import load_arima
from monte_carlo import run_monte_carlo_simulation  

app = FastAPI()

# ✅ Enable CORS for frontend communication (development only)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Root check route
@app.get("/")
def root():
    return {"message": "🌤️ Weather Forecast API is running!"}

# ✅ ARIMA Forecast route
@app.get("/forecast/arima/{city}")
def arima_forecast(city: str):
    try:
        city = city.lower()
        model = load_arima(city)
        forecast = model.forecast(steps=7)
        rounded_forecast = [round(temp, 2) for temp in forecast]
        return {
            "model": "ARIMA",
            "city": city,
            "forecast": rounded_forecast
        }
    except FileNotFoundError as fnf:
        return JSONResponse(status_code=404, content={"error": str(fnf)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# ✅ Monte Carlo Forecast route
@app.get("/forecast/monte-carlo/{city}")
def monte_carlo_forecast(city: str):
    try:
        forecast = run_monte_carlo_simulation(city.lower())
        return {
            "model": "Monte Carlo",
            "city": city,
            "forecast": [round(temp, 2) for temp in forecast]
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# ✅ Run manually if this file is executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("weather_api:app", host="127.0.0.1", port=8000, reload=True)
