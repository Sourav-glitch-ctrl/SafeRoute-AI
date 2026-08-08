"""
Using Train.py we are going to train the model using the dataset that we have prepared in the previous step.
The models that we are going to use are:
    1. Logistic Regression
    2. Random Forest
    3. XGBoost
    4. LightGBM
"""
from pathlib import Path
import sys
import time
import joblib

# Import necessary libraries for data manipulation and machine learning
TRAINING_DIR = Path(__file__).resolve().parent
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

# Project Imports

from config import MODEL_DIR
from dataset import prepare_dataset
from models import get_models

# Model Selection
# We start with Logistic Regression as our baseline.
# Later we will change this to:
#     "Random Forest"
#     "XGBoost"
#     "LightGBM"

SELECTED_MODEL = "Logistic Regression"

# Train Model
def train_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
):

    print("\n" + "=" * 70)
    print("Model Training")
    print("=" * 70)

    print(f"Model : {SELECTED_MODEL}")

    print("\nTraining model...")

    start_time = time.perf_counter()

    model.fit(
        X_train,
        y_train,
    )

    end_time = time.perf_counter()

    training_time = end_time - start_time

    print("\nTraining completed successfully.")

    print(
        f"Training Time : "
        f"{training_time:.2f} seconds"
    )

    return model, training_time

# Save Model
def save_model(
    model,
    model_name: str,
):
    """
    Save the trained model as a pickle file.

    Models are stored inside:
        ml/saved_models/

    Example:
        logistic_regression.pkl
    """

    safe_name = (
        model_name
        .lower()
        .replace(" ", "_")
    )

    model_path = (
        MODEL_DIR
        / f"{safe_name}.pkl"
    )

    joblib.dump(
        model,
        model_path,
    )

    print(
        f"\nModel saved successfully at:\n"
        f"{model_path}"
    )

    return model_path

# Main Training Pipeline

def main():
    """
    Execute the complete model training pipeline.
    """

    print("=" * 70)
    print("SafeRoute AI - Machine Learning Training")
    print("=" * 70)

    # -------------------------------------------------------------
    # Load and prepare dataset
    # -------------------------------------------------------------

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        encoder,
        categorical_columns,
    ) = prepare_dataset()

    # -------------------------------------------------------------
    # Load model registry
    # -------------------------------------------------------------

    models = get_models()

    if SELECTED_MODEL not in models:
        raise ValueError(
            f"Model '{SELECTED_MODEL}' is not available.\n"
            f"Available models: {list(models.keys())}"
        )

    model = models[SELECTED_MODEL]

    # -------------------------------------------------------------
    # Train model
    # -------------------------------------------------------------

    trained_model, training_time = train_model(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
    )

    # -------------------------------------------------------------
    # Save model
    # -------------------------------------------------------------

    model_path = save_model(
        trained_model,
        SELECTED_MODEL,
    )

    # -------------------------------------------------------------
    # Training Summary
    # -------------------------------------------------------------

    print("\n" + "=" * 70)
    print("Training Summary")
    print("=" * 70)

    print(f"Model          : {SELECTED_MODEL}")
    print(f"Training Rows  : {len(X_train):,}")
    print(f"Validation Rows: {len(X_val):,}")
    print(f"Testing Rows   : {len(X_test):,}")
    print(f"Training Time  : {training_time:.2f} seconds")
    print(f"Saved Model    : {model_path}")

    print("\n" + "=" * 70)
    print("Model Training Completed Successfully!")
    print("=" * 70)


# ---------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------

if __name__ == "__main__":
    main()