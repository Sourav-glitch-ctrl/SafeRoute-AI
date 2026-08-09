"""
Using Train.py we are going to train the model using the dataset that we have
prepared in the previous step.

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


# ---------------------------------------------------------------------
# Project Path
# ---------------------------------------------------------------------

TRAINING_DIR = Path(__file__).resolve().parent

if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))


# ---------------------------------------------------------------------
# Project Imports
# ---------------------------------------------------------------------

from config import (
    MODEL_DIR,
    RANDOM_STATE,
    RANDOM_FOREST_TRAIN_SIZE,
)

from dataset import (
    prepare_dataset,
    get_random_forest_training_data,
)

from models import get_models


# ---------------------------------------------------------------------
# Model Selection
# ---------------------------------------------------------------------

# We start with Random Forest after completing Logistic Regression
# as our baseline.
#
# Later we will change this to:
#
# "Logistic Regression"
# "XGBoost"
# "LightGBM"

SELECTED_MODEL = "Random Forest"


# ---------------------------------------------------------------------
# Train Model
# ---------------------------------------------------------------------

def train_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
):
    """
    Train the selected machine learning model.

    XGBoost requires class labels to start from 0.
    Therefore, Severity values 1-4 are converted to
    0-3 only during XGBoost training.
    """

    print("\n" + "=" * 70)
    print("Model Training")
    print("=" * 70)

    print(f"Model : {SELECTED_MODEL}")

    print("\nTraining model...")

    # -------------------------------------------------------------
    # XGBoost requires classes 0, 1, 2, 3
    # Original Severity values are 1, 2, 3, 4
    # -------------------------------------------------------------

    if SELECTED_MODEL == "XGBoost":

        print(
            "\nConverting Severity labels "
            "from [1, 2, 3, 4] to [0, 1, 2, 3] for XGBoost..."
        )

        y_train = y_train - 1
        y_val = y_val - 1

    # -------------------------------------------------------------
    # Train model
    # -------------------------------------------------------------

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


# ---------------------------------------------------------------------
# Save Model
# ---------------------------------------------------------------------

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
        random_forest.pkl
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


# ---------------------------------------------------------------------
# Main Training Pipeline
# ---------------------------------------------------------------------

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
    # Model-specific training data
    # -------------------------------------------------------------

    X_model_train = X_train
    y_model_train = y_train

    # Random Forest is memory-intensive when trained on the complete
    # 5.4M-row training dataset.
    #
    # Therefore, only Random Forest uses a stratified subset.
    #
    # Validation and testing datasets remain unchanged.

    if SELECTED_MODEL == "Random Forest":

        print("\n" + "=" * 70)
        print("Preparing Random Forest Training Data")
        print("=" * 70)

        X_model_train, y_model_train = (
            get_random_forest_training_data(
                X_train,
                y_train,
                sample_size=RANDOM_FOREST_TRAIN_SIZE,
                random_state=RANDOM_STATE,
            )
        )

    else:
        # XGBoost, LightGBM and Logistic Regression
        # use the complete training dataset.

        X_model_train = X_train
        y_model_train = y_train

    # -------------------------------------------------------------
    # Train model
    # -------------------------------------------------------------

    trained_model, training_time = train_model(
        model,
        X_model_train,
        y_model_train,
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

    print(f"Model                    : {SELECTED_MODEL}")

    print(
        f"Original Training Rows   : "
        f"{len(X_train):,}"
    )

    print(
        f"Model Training Rows      : "
        f"{len(X_model_train):,}"
    )

    print(
        f"Validation Rows          : "
        f"{len(X_val):,}"
    )

    print(
        f"Testing Rows             : "
        f"{len(X_test):,}"
    )

    print(
        f"Training Time            : "
        f"{training_time:.2f} seconds"
    )

    print(
        f"Saved Model              : "
        f"{model_path}"
    )

    print("\n" + "=" * 70)
    print("Model Training Completed Successfully!")
    print("=" * 70)


# ---------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------

if __name__ == "__main__":
    main()