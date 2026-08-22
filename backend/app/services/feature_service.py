from datetime import datetime

from app.services.osm_service import get_road_features
from app.services.weather_service import get_weather


def build_features(
    latitude: float,
    longitude: float,
    distance_mi: float,
    timestamp: datetime | None = None,
) -> dict:
    """
    Build the complete real-time feature set required
    by the Random Forest model.
    """

    if timestamp is None:
        timestamp = datetime.now()

    # ---------------------------------------------------------
    # Weather features
    # ---------------------------------------------------------

    weather = get_weather(
        latitude=latitude,
        longitude=longitude,
    )

    # ---------------------------------------------------------
    # Road infrastructure features
    # ---------------------------------------------------------

    road_features = get_road_features(
        latitude=latitude,
        longitude=longitude,
    )

    # ---------------------------------------------------------
    # Time-based features
    # ---------------------------------------------------------

    hour = timestamp.hour
    day_of_week = timestamp.isoweekday()

    is_weekend = int(day_of_week >= 6)

    is_night = int(
        hour < 6 or hour >= 19
    )

    is_rush_hour = int(
        7 <= hour <= 9
        or 16 <= hour <= 19
    )

    morning_rush_hour = int(
        7 <= hour <= 9
    )

    evening_rush_hour = int(
        16 <= hour <= 19
    )

    quarter = (
        (timestamp.month - 1) // 3
    ) + 1

    # ---------------------------------------------------------
    # Complete feature dictionary
    # ---------------------------------------------------------

    features = {

        # Location / distance
        "Distance(mi)": distance_mi,
        "Year": timestamp.year,
        "Start_Lng": longitude,
        "Start_Lat": latitude,

        # Weather
        "Temperature(F)": weather["Temperature(F)"],
        "Humidity(%)": weather["Humidity(%)"],
        "Pressure(in)": weather["Pressure(in)"],
        "Visibility(mi)": weather["Visibility(mi)"],
        "Wind_Speed(mph)": weather["Wind_Speed(mph)"],
        "Precipitation(in)": weather["Precipitation(in)"],
        "Weather_Category": weather["Weather_Category"],

        # Time
        "Month": timestamp.month,
        "Hour": hour,
        "Quarter": quarter,
        "DayOfWeek": day_of_week,

        # Road infrastructure
        "Traffic_Signal": road_features["Traffic_Signal"],
        "Crossing": road_features["Crossing"],
        "Junction": road_features["Junction"],
        "Railway": road_features["Railway"],
        "Stop": road_features["Stop"],
        "NearRoadInfrastructure": (
            road_features["NearRoadInfrastructure"]
        ),

        # Derived features
        "IsWeekend": is_weekend,
        "IsNight": is_night,
        "IsRushHour": is_rush_hour,
        "MorningRushHour": morning_rush_hour,
        "EveningRushHour": evening_rush_hour,
        "HasPrecipitation": weather["HasPrecipitation"],
        "LowVisibility": weather["LowVisibility"],
    }

    return features