"""
SafeRoute AI - Prediction Module

Loads the trained Random Forest model and preprocessing artifacts
and generates Severity predictions for new accident/road conditions.
"""

from pathlib import Path

import joblib
import pandas as pd


# ---------------------------------------------------------------------
# Project Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = (
    PROJECT_ROOT
    / "ml"
    / "saved_models"
)

MODEL_FILE = (
    MODEL_DIR
    / "random_forest.pkl"
)

PREPROCESSING_FILE = (
    MODEL_DIR
    / "preprocessing.pkl"
)


# ---------------------------------------------------------------------
# Load Model
# ---------------------------------------------------------------------

def load_model():
    """
    Load the trained Random Forest model.
    """

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Trained model not found:\n"
            f"{MODEL_FILE}\n\n"
            "Run ml/training/train.py first."
        )

    print("Loading Random Forest model...")

    model = joblib.load(
        MODEL_FILE
    )

    print(
        "Model loaded successfully."
    )

    return model


# ---------------------------------------------------------------------
# Load Preprocessing Artifacts
# ---------------------------------------------------------------------

def load_preprocessing():
    """
    Load the preprocessing artifacts used during training.

    Returns
    -------
    encoder :
        Fitted OrdinalEncoder.

    categorical_columns : list
        Categorical feature names used during training.
    """

    if not PREPROCESSING_FILE.exists():
        raise FileNotFoundError(
            f"Preprocessing artifacts not found:\n"
            f"{PREPROCESSING_FILE}\n\n"
            "Run ml/training/train.py first."
        )

    print(
        "Loading preprocessing artifacts..."
    )

    artifacts = joblib.load(
        PREPROCESSING_FILE
    )

    if not isinstance(
        artifacts,
        dict
    ):
        raise ValueError(
            "Invalid preprocessing artifact format."
        )

    if "encoder" not in artifacts:
        raise KeyError(
            "Encoder not found in preprocessing artifacts."
        )

    if "categorical_columns" not in artifacts:
        raise KeyError(
            "Categorical columns not found "
            "in preprocessing artifacts."
        )

    encoder = artifacts["encoder"]

    categorical_columns = (
        artifacts["categorical_columns"]
    )

    print(
        "Preprocessing artifacts "
        "loaded successfully."
    )

    print(
        f"Categorical Features : "
        f"{categorical_columns}"
    )

    return (
        encoder,
        categorical_columns,
    )


# ---------------------------------------------------------------------
# Preprocess Input
# ---------------------------------------------------------------------

def preprocess_input(
    input_data: dict,
    model,
    encoder,
    categorical_columns,
):
    """
    Convert a single prediction input into the exact
    feature structure expected by the trained model.

    Parameters
    ----------
    input_data : dict
        Feature names and their values.

    model :
        Trained Random Forest model.

    encoder :
        OrdinalEncoder fitted during training.

    categorical_columns : list
        Names of categorical features.

    Returns
    -------
    pd.DataFrame
        Preprocessed feature DataFrame.
    """

    # -------------------------------------------------------------
    # Convert input dictionary to DataFrame
    # -------------------------------------------------------------

    X = pd.DataFrame(
        [input_data]
    )

    # -------------------------------------------------------------
    # Get exact feature order used during training
    # -------------------------------------------------------------

    if not hasattr(
        model,
        "feature_names_in_"
    ):
        raise AttributeError(
            "The trained model does not contain "
            "feature_names_in_. "
            "Feature order cannot be verified safely."
        )

    expected_features = list(
        model.feature_names_in_
    )

    # -------------------------------------------------------------
    # Check missing features
    # -------------------------------------------------------------

    missing_features = [
        feature
        for feature in expected_features
        if feature not in X.columns
    ]

    if missing_features:

        raise ValueError(
            "Missing required features:\n"
            + "\n".join(
                f"- {feature}"
                for feature in missing_features
            )
        )

    # -------------------------------------------------------------
    # Keep only training features
    # and enforce exact feature order
    # -------------------------------------------------------------

    X = X[
        expected_features
    ].copy()

    # -------------------------------------------------------------
    # Check categorical columns
    # -------------------------------------------------------------

    missing_categorical_columns = [
        column
        for column in categorical_columns
        if column not in X.columns
    ]

    if missing_categorical_columns:

        raise ValueError(
            "Categorical features missing "
            "from input:\n"
            + "\n".join(
                f"- {column}"
                for column in missing_categorical_columns
            )
        )

    # -------------------------------------------------------------
    # Encode categorical features
    # -------------------------------------------------------------

    if categorical_columns:

        categorical_data = (
            X[categorical_columns]
            .astype(object)
        )

        encoded_values = (
            encoder.transform(
                categorical_data
            )
        )

        for index, column in enumerate(
            categorical_columns
        ):

            X[column] = (
                encoded_values[:, index]
                .astype("float32")
            )

    # -------------------------------------------------------------
    # Convert remaining features to numeric
    # -------------------------------------------------------------

    for column in X.columns:

        if column not in categorical_columns:

            X[column] = pd.to_numeric(
                X[column],
                errors="raise",
            )

    return X


# ---------------------------------------------------------------------
# Predict Severity
# ---------------------------------------------------------------------

def predict_severity(
    input_data: dict,
):
    """
    Predict accident Severity for a single input.

    Parameters
    ----------
    input_data : dict
        Input feature values.

    Returns
    -------
    dict
        Prediction result containing:

        severity
        probabilities
    """

    # -------------------------------------------------------------
    # Load model
    # -------------------------------------------------------------

    model = load_model()

    # -------------------------------------------------------------
    # Load preprocessing artifacts
    # -------------------------------------------------------------

    (
        encoder,
        categorical_columns,
    ) = load_preprocessing()

    # -------------------------------------------------------------
    # Preprocess input
    # -------------------------------------------------------------

    X = preprocess_input(
        input_data=input_data,
        model=model,
        encoder=encoder,
        categorical_columns=categorical_columns,
    )

    # -------------------------------------------------------------
    # Generate prediction
    # -------------------------------------------------------------

    prediction = model.predict(
        X
    )

    severity = int(
        prediction[0]
    )

    # -------------------------------------------------------------
    # Generate probabilities
    # -------------------------------------------------------------

    probabilities = None

    if hasattr(
        model,
        "predict_proba"
    ):

        probability_values = (
            model.predict_proba(X)[0]
        )

        probabilities = {
            int(class_label): float(
                probability
            )
            for class_label, probability in zip(
                model.classes_,
                probability_values,
            )
        }

    # -------------------------------------------------------------
    # Return result
    # -------------------------------------------------------------

    return {
        "severity": severity,
        "probabilities": probabilities,
    }


# ---------------------------------------------------------------------
# Example Prediction
# ---------------------------------------------------------------------

if __name__ == "__main__":

    print(
        "\n"
        + "=" * 70
    )

    print(
        "SafeRoute AI - Prediction Test"
    )

    print(
        "=" * 70
    )

    # -------------------------------------------------------------
    # Example input
    #
    # IMPORTANT:
    # These values are only for testing the prediction pipeline.
    # They are NOT recommended real-world values.
    # -------------------------------------------------------------

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

    # -------------------------------------------------------------
    # Prediction
    # -------------------------------------------------------------

    result = predict_severity(
        sample_input
    )

    # -------------------------------------------------------------
    # Display result
    # -------------------------------------------------------------

    print(
        "\nPrediction Result"
    )

    print(
        "-" * 70
    )

    print(
        f"Predicted Severity : "
        f"{result['severity']}"
    )

    # -------------------------------------------------------------
    # Display probabilities
    # -------------------------------------------------------------

    if result["probabilities"]:

        print(
            "\nClass Probabilities:"
        )

        for (
            severity,
            probability
        ) in result["probabilities"].items():

            print(
                f"Severity {severity} : "
                f"{probability:.4f}"
            )

    print(
        "\nPrediction completed successfully."
    )