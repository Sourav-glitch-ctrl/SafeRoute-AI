import httpx


OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


WEATHER_CODE_MAP = {
    0: "Clear",
    1: "Clear",
    2: "Cloudy",
    3: "Cloudy",
    45: "Fog",
    48: "Fog",
    51: "Rain",
    53: "Rain",
    55: "Rain",
    56: "Ice",
    57: "Ice",
    61: "Rain",
    63: "Rain",
    65: "Rain",
    66: "Ice",
    67: "Ice",
    71: "Snow",
    73: "Snow",
    75: "Snow",
    77: "Snow",
    80: "Rain",
    81: "Rain",
    82: "Rain",
    85: "Snow",
    86: "Snow",
    95: "Storm",
    96: "Storm",
    99: "Storm",
}


def get_weather(latitude: float, longitude: float) -> dict:
    """
    Fetch current weather conditions for a latitude/longitude.
    """

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "surface_pressure,"
            "visibility,"
            "wind_speed_10m,"
            "precipitation,"
            "weather_code"
        ),
        "wind_speed_unit": "mph",
        "temperature_unit": "fahrenheit",
        "precipitation_unit": "inch",
    }

    try:
        response = httpx.get(
            OPEN_METEO_URL,
            params=params,
            timeout=10.0,
        )

        response.raise_for_status()

        data = response.json()

        current = data["current"]

        weather_code = current["weather_code"]

        weather_category = WEATHER_CODE_MAP.get(
            weather_code,
            "Unknown",
        )

        temperature = current["temperature_2m"]
        humidity = current["relative_humidity_2m"]
        pressure = current["surface_pressure"]
        visibility_m = current["visibility"]
        wind_speed = current["wind_speed_10m"]
        precipitation = current["precipitation"]

        # Open-Meteo visibility is in metres.
        visibility_mi = visibility_m / 1609.344

        return {
            "Temperature(F)": temperature,
            "Humidity(%)": humidity,
            "Pressure(in)": pressure * 0.0295299833,
            "Visibility(mi)": visibility_mi,
            "Wind_Speed(mph)": wind_speed,
            "Precipitation(in)": precipitation,
            "Weather_Category": weather_category,
            "HasPrecipitation": int(precipitation > 0),
            "LowVisibility": int(visibility_mi < 2),
        }

    except httpx.HTTPError as exc:
        raise RuntimeError(
            f"Weather service request failed: {exc}"
        ) from exc

    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Invalid weather service response: {exc}"
        ) from exc