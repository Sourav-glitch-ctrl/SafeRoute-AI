from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database.connection import get_db
from app.schemas.prediction import (
    PredictionRequest,
    PredictionResponse,
    RoutePredictionRequest,
    RoutePredictionResponse,
    RoutePredictionPoint,
)
from app.services.prediction_service import save_prediction
from app.services.feature_service import build_features

from ml.inference.inference import predict_severity
from ml.inference.inference import run_inference


router = APIRouter()


@router.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(
    request: PredictionRequest,
    db: Session = Depends(get_db),
):

    try:

        # ---------------------------------------------------------
        # Convert API fields to ML feature names
        # ---------------------------------------------------------

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

        # ---------------------------------------------------------
        # Run Random Forest inference
        # ---------------------------------------------------------

        result = predict_severity(input_data)

        response = PredictionResponse(
            severity=result["severity"],
            probabilities=result["probabilities"],
        )

        # ---------------------------------------------------------
        # Save prediction to PostgreSQL
        # ---------------------------------------------------------

        save_prediction(
            db=db,
            request=request,
            response=response,
        )

        # ---------------------------------------------------------
        # Return prediction to frontend
        # ---------------------------------------------------------

        return response

    except Exception as e:

        db.rollback()

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}",
        )


@router.post(
    "/route",
    response_model=RoutePredictionResponse,
)
def predict_route(
    request: RoutePredictionRequest,
):
    try:

        results = []

        total_points = len(request.points)

        if total_points == 0:
            raise HTTPException(
                status_code=400,
                detail="Route must contain at least one point.",
            )

        # ---------------------------------------------------------
        # Analyze representative route points
        # ---------------------------------------------------------

        for index, point in enumerate(request.points):

            # Use the route distance as the distance feature.
            # For this simplified version, the same route distance
            # is provided to each sampled point.
            features = build_features(
                latitude=point.lat,
                longitude=point.lng,
                distance_mi=request.distance_mi,
            )

            result = run_inference(features)

            probabilities = result["probabilities"]

            severity = result["severity"]

            results.append(
                RoutePredictionPoint(
                    lat=point.lat,
                    lng=point.lng,
                    severity=severity,
                    probability=float(
                        probabilities.get(
                            severity,
                            0.0,
                        )
                    ),
                )
            )

        # ---------------------------------------------------------
        # Overall route severity
        # ---------------------------------------------------------

        overall_severity = max(
            point.severity
            for point in results
        )

        return RoutePredictionResponse(
            overall_severity=overall_severity,
            points=results,
        )

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Route prediction failed: {str(e)}",
        )