from fastapi import APIRouter, HTTPException

from app.schemas.prediction import (
    PredictionRequest,
    PredictionResponse,
)

from ml.inference.inference import predict_severity


router = APIRouter()


@router.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(request: PredictionRequest):

    try:

        input_data = {
            "Distance(mi)": request.Distance_mi,
            "Year": request.Year,
            "Start_Lng": request.Start_Lng,
            "Start_Lat": request.Start_Lat,
            "Pressure(in)": request.Pressure_in,
            "Temperature(F)": request.Temperature_F,
            "Month": request.Month,
            "Humidity(%)": request.Humidity_percent,
            "Hour": request.Hour,
            "Wind_Speed(mph)": request.Wind_Speed_mph,
            "Quarter": request.Quarter,
            "DayOfWeek": request.DayOfWeek,
            "Traffic_Signal": request.Traffic_Signal,
            "Weather_Category": request.Weather_Category,
            "Visibility(mi)": request.Visibility_mi,
            "NearRoadInfrastructure": request.NearRoadInfrastructure,
            "Crossing": request.Crossing,
            "Junction": request.Junction,
            "IsWeekend": request.IsWeekend,
            "IsNight": request.IsNight,
            "IsRushHour": request.IsRushHour,
            "Precipitation(in)": request.Precipitation_in,
            "MorningRushHour": request.MorningRushHour,
            "EveningRushHour": request.EveningRushHour,
            "Stop": request.Stop,
            "HasPrecipitation": request.HasPrecipitation,
            "LowVisibility": request.LowVisibility,
            "Railway": request.Railway,
        }

        result = predict_severity(input_data)

        return PredictionResponse(
            severity=result["severity"],
            probabilities=result["probabilities"],
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}",
        )