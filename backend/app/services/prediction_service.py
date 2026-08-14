"""
SafeRoute AI - Prediction Service

Connects the FastAPI backend with the ML inference layer.
"""

import sys
from pathlib import Path


# ---------------------------------------------------------------------
# Project Path
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

INFERENCE_DIR = PROJECT_ROOT / "ml" / "inference"

if str(INFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(INFERENCE_DIR))


# ---------------------------------------------------------------------
# Import ML Inference
# ---------------------------------------------------------------------

from inference import run_inference


# ---------------------------------------------------------------------
# Prediction Service
# ---------------------------------------------------------------------

def predict_severity(data: dict):
    """
    Convert API input into the format expected by the ML model
    and run inference.
    """

    model_input = {
        "Distance(mi)": data["Distance_mi"],
        "Year": data["Year"],
        "Start_Lng": data["Start_Lng"],
        "Start_Lat": data["Start_Lat"],
        "Pressure(in)": data["Pressure_in"],
        "Temperature(F)": data["Temperature_F"],
        "Month": data["Month"],
        "Humidity(%)": data["Humidity_percent"],
        "Hour": data["Hour"],
        "Wind_Speed(mph)": data["Wind_Speed_mph"],
        "Quarter": data["Quarter"],
        "DayOfWeek": data["DayOfWeek"],
        "Traffic_Signal": data["Traffic_Signal"],
        "Weather_Category": data["Weather_Category"],
        "Visibility(mi)": data["Visibility_mi"],
        "NearRoadInfrastructure": data[
            "NearRoadInfrastructure"
        ],
        "Crossing": data["Crossing"],
        "Junction": data["Junction"],
        "IsWeekend": data["IsWeekend"],
        "IsNight": data["IsNight"],
        "IsRushHour": data["IsRushHour"],
        "Precipitation(in)": data["Precipitation_in"],
        "MorningRushHour": data["MorningRushHour"],
        "EveningRushHour": data["EveningRushHour"],
        "Stop": data["Stop"],
        "HasPrecipitation": data["HasPrecipitation"],
        "LowVisibility": data["LowVisibility"],
        "Railway": data["Railway"],
    }

    return run_inference(model_input)