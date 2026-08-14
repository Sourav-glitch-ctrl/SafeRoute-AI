"""
SafeRoute AI - Inference Module

Application-level inference layer.

This module receives prediction input, calls the prediction
module, and returns a clean result for the backend/API.
"""

import sys
from pathlib import Path


# ---------------------------------------------------------------------
# Project Path
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PREDICTION_DIR = PROJECT_ROOT / "ml" / "prediction"

if str(PREDICTION_DIR) not in sys.path:
    sys.path.insert(0, str(PREDICTION_DIR))


# ---------------------------------------------------------------------
# Import Prediction Module
# ---------------------------------------------------------------------

from predict import predict_severity


# ---------------------------------------------------------------------
# Run Inference
# ---------------------------------------------------------------------

def run_inference(input_data: dict):
    """
    Run SafeRoute AI inference for a single input.

    Parameters
    ----------
    input_data : dict
        Road, weather and accident-related features.

    Returns
    -------
    dict
        Prediction result containing severity and probabilities.
    """

    if not isinstance(input_data, dict):
        raise TypeError(
            "input_data must be a dictionary."
        )

    if not input_data:
        raise ValueError(
            "input_data cannot be empty."
        )

    # -------------------------------------------------------------
    # Call trained ML prediction pipeline
    # -------------------------------------------------------------

    result = predict_severity(
        input_data
    )

    # -------------------------------------------------------------
    # Return clean inference result
    # -------------------------------------------------------------

    return {
        "severity": result["severity"],
        "probabilities": result["probabilities"],
    }


# ---------------------------------------------------------------------
# Test Inference
# ---------------------------------------------------------------------

if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("SafeRoute AI - Inference Test")
    print("=" * 70)

    sample_input = {
        "Distance(mi)": 1.2,
        "Year": 2024,
        "Start_Lng": 72.8777,
        "Start_Lat": 19.0760,
        "Pressure(in)": 29.85,
        "Temperature(F)": 82.0,
        "Month": 8,
        "Humidity(%)": 75.0,
        "Hour": 18,
        "Wind_Speed(mph)": 8.0,
        "Quarter": 3,
        "DayOfWeek": 5,
        "Traffic_Signal": 1,
        "Weather_Category": "Rain",
        "Visibility(mi)": 5.0,
        "NearRoadInfrastructure": 1,
        "Crossing": 0,
        "Junction": 0,
        "IsWeekend": 0,
        "IsNight": 0,
        "IsRushHour": 1,
        "Precipitation(in)": 0.2,
        "MorningRushHour": 0,
        "EveningRushHour": 1,
        "Stop": 0,
        "HasPrecipitation": 1,
        "LowVisibility": 0,
        "Railway": 0,
    }

    result = run_inference(
        sample_input
    )

    print("\nInference Result")
    print("-" * 70)

    print(
        f"Predicted Severity : "
        f"{result['severity']}"
    )

    if result["probabilities"]:

        print("\nClass Probabilities:")

        for severity, probability in (
            result["probabilities"].items()
        ):

            print(
                f"Severity {severity} : "
                f"{probability:.4f}"
            )

    print("\nInference completed successfully.")