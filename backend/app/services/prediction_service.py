from sqlalchemy.orm import Session

from app.models.prediction import Prediction
from app.schemas.prediction import PredictionRequest, PredictionResponse


def save_prediction(
    db: Session,
    request: PredictionRequest,
    response: PredictionResponse,
) -> Prediction:

    probabilities = response.probabilities

    prediction = Prediction(
        distance_mi=request.Distance_mi,
        year=request.Year,
        start_lng=request.Start_Lng,
        start_lat=request.Start_Lat,
        pressure_in=request.Pressure_in,
        temperature_f=request.Temperature_F,
        month=request.Month,
        humidity_percent=request.Humidity_percent,
        hour=request.Hour,
        wind_speed_mph=request.Wind_Speed_mph,
        quarter=request.Quarter,
        day_of_week=request.DayOfWeek,
        traffic_signal=request.Traffic_Signal,
        weather_category=request.Weather_Category,
        visibility_mi=request.Visibility_mi,
        near_road_infrastructure=request.NearRoadInfrastructure,
        crossing=request.Crossing,
        junction=request.Junction,
        is_weekend=request.IsWeekend,
        is_night=request.IsNight,
        is_rush_hour=request.IsRushHour,
        precipitation_in=request.Precipitation_in,
        morning_rush_hour=request.MorningRushHour,
        evening_rush_hour=request.EveningRushHour,
        stop=request.Stop,
        has_precipitation=request.HasPrecipitation,
        low_visibility=request.LowVisibility,
        railway=request.Railway,
        severity=response.severity,
        probability_1=probabilities.get(1),
        probability_2=probabilities.get(2),
        probability_3=probabilities.get(3),
        probability_4=probabilities.get(4),
    )

    db.add(prediction)
    db.commit()
    db.refresh(prediction)

    return prediction