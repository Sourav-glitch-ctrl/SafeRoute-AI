"""
SafeRoute AI - Model Evaluation

This module evaluates the selected trained machine learning model
using the untouched testing dataset.

Evaluation metrics:
- Accuracy
- Weighted Precision
- Weighted Recall
- Weighted F1 Score
- Macro F1 Score
- Classification Report
- Confusion Matrix

The evaluation results are saved inside:
    ml/metrics/
"""

from pathlib import Path
import json
import time
import joblib
import pandas as pd
import sys

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

TRAINING_DIR = (
    Path(__file__).resolve().parent.parent / "training"
)

sys.path.insert(
    0,
    str(TRAINING_DIR)
)


from config import (
    DATA_FILE,
    TARGET_COLUMN,
    MODEL_DIR,
    METRICS_DIR,
    RANDOM_STATE,
    TRAIN_SIZE,
    VALIDATION_SIZE,
    TEST_SIZE,
)


# ---------------------------------------------------------------------
# Model Configuration
# ---------------------------------------------------------------------

MODEL_NAME = "random_forest"

MODEL_FILE = (
    MODEL_DIR / f"{MODEL_NAME}.pkl"
)

METRICS_FILE = (
    METRICS_DIR
    / f"{MODEL_NAME}_metrics.json"
)

CONFUSION_MATRIX_FILE = (
    METRICS_DIR
    / f"{MODEL_NAME}_confusion_matrix.csv"
)


# ---------------------------------------------------------------------
# Load Dataset
# ---------------------------------------------------------------------

def load_dataset():
    """
    Load the feature engineered dataset.

    Returns:
        pandas.DataFrame
    """

    print("=" * 70)
    print("Loading Feature Engineered Dataset")
    print("=" * 70)

    df = pd.read_parquet(DATA_FILE)

    print(
        f"Dataset Shape : {df.shape}"
    )

    return df


# ---------------------------------------------------------------------
# Prepare Features and Target
# ---------------------------------------------------------------------

def prepare_data(df):
    """
    Separate features and target and encode categorical features.

    Returns:
        X : Feature dataframe
        y : Target series
    """

    print("\nSeparating features and target...")

    X = df.drop(
        columns=[TARGET_COLUMN]
    )

    y = df[TARGET_COLUMN]

    print(
        f"Features Shape : {X.shape}"
    )

    print(
        f"Target Shape   : {y.shape}"
    )

    # -------------------------------------------------------------
    # Encode categorical features
    # -------------------------------------------------------------

    print(
        "\nEncoding categorical features..."
    )

    categorical_columns = (
        X.select_dtypes(
            include=[
                "object",
                "string",
                "category",
            ]
        )
        .columns
        .tolist()
    )

    print(
        f"Categorical Features : "
        f"{categorical_columns}"
    )

    for column in categorical_columns:

        X[column] = (
            X[column]
            .astype("category")
            .cat.codes
            .astype("float32")
        )

    return X, y


# ---------------------------------------------------------------------
# Create Train / Validation / Test Split
# ---------------------------------------------------------------------

def create_split(X, y):
    """
    Recreate the same stratified train/validation/test split
    used during model training.

    Split:
        Training   -> 70%
        Validation -> 15%
        Testing    -> 15%

    Returns:
        X_train
        X_validation
        X_test
        y_train
        y_validation
        y_test
    """

    print(
        "\nCreating dataset split..."
    )

    # -------------------------------------------------------------
    # First split
    # -------------------------------------------------------------
    # 70% training
    # 30% temporary

    X_train, X_temp, y_train, y_temp = (
        train_test_split(
            X,
            y,
            test_size=(
                VALIDATION_SIZE
                + TEST_SIZE
            ),
            stratify=y,
            random_state=RANDOM_STATE,
        )
    )

    # -------------------------------------------------------------
    # Second split
    # -------------------------------------------------------------
    # Temporary 30% is divided into:
    # 15% validation
    # 15% testing

    validation_ratio = (
        VALIDATION_SIZE
        / (
            VALIDATION_SIZE
            + TEST_SIZE
        )
    )

    (
        X_validation,
        X_test,
        y_validation,
        y_test,
    ) = train_test_split(
        X_temp,
        y_temp,
        test_size=(
            1 - validation_ratio
        ),
        stratify=y_temp,
        random_state=RANDOM_STATE,
    )

    print("\nSplit Shapes")

    print(
        f"Training   : {X_train.shape}"
    )

    print(
        f"Validation : {X_validation.shape}"
    )

    print(
        f"Testing    : {X_test.shape}"
    )

    return (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    )


# ---------------------------------------------------------------------
# Load Trained Model
# ---------------------------------------------------------------------

def load_model():
    """
    Load the selected trained model.
    """

    print(
        "\n" + "=" * 70
    )

    print(
        "Loading Trained Model"
    )

    print(
        "=" * 70
    )

    print(
        f"Model : {MODEL_NAME}"
    )

    print(
        f"Path  : {MODEL_FILE}"
    )

    if not MODEL_FILE.exists():

        raise FileNotFoundError(
            f"\nModel file not found:\n"
            f"{MODEL_FILE}"
        )

    model = joblib.load(
        MODEL_FILE
    )

    print(
        "\nModel loaded successfully."
    )

    return model


# ---------------------------------------------------------------------
# Evaluate Model on Test Set
# ---------------------------------------------------------------------

def evaluate_model(
    model,
    X_test,
    y_test,
):
    """
    Evaluate the trained model on the untouched testing dataset.

    Returns:
        metrics
        confusion matrix
    """

    print(
        "\n" + "=" * 70
    )

    print(
        "Evaluating Model on Test Set"
    )

    print(
        "=" * 70
    )

    print(
        f"Testing Rows : "
        f"{len(X_test):,}"
    )

    start_time = time.time()

    print(
        "\nGenerating predictions..."
    )

    y_pred = model.predict(
        X_test
    )

    # -------------------------------------------------------------
    # Convert XGBoost labels back to original Severity labels
    # -------------------------------------------------------------

    if MODEL_NAME == "xgboost":

        y_pred = (
            y_pred.astype(int)
            + 1
        )

    evaluation_time = (
        time.time() - start_time
    )

    # -------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred,
    )

    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    f1_weighted = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    f1_macro = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )

    # -------------------------------------------------------------
    # Classification Report
    # -------------------------------------------------------------

    report = classification_report(
        y_test,
        y_pred,
        zero_division=0,
    )

    report_dict = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0,
    )

    # -------------------------------------------------------------
    # Confusion Matrix
    # -------------------------------------------------------------

    matrix = confusion_matrix(
        y_test,
        y_pred,
        labels=[1, 2, 3, 4],
    )

    # -------------------------------------------------------------
    # Print Results
    # -------------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "Test Set Evaluation Results"
    )

    print(
        "=" * 70
    )

    print(
        f"Accuracy           : "
        f"{accuracy:.4f}"
    )

    print(
        f"Weighted Precision : "
        f"{precision:.4f}"
    )

    print(
        f"Weighted Recall    : "
        f"{recall:.4f}"
    )

    print(
        f"Weighted F1 Score  : "
        f"{f1_weighted:.4f}"
    )

    print(
        f"Macro F1 Score     : "
        f"{f1_macro:.4f}"
    )

    print(
        f"Evaluation Time    : "
        f"{evaluation_time:.2f} seconds"
    )

    print(
        "\nClassification Report"
    )

    print(
        "-" * 70
    )

    print(report)

    print(
        "\nConfusion Matrix"
    )

    print(
        "-" * 70
    )

    print(matrix)

    # -------------------------------------------------------------
    # Metrics Dictionary
    # -------------------------------------------------------------

    metrics = {

        "model": MODEL_NAME,

        "evaluation_dataset": "test",

        "test_rows": int(
            len(X_test)
        ),

        "accuracy": float(
            accuracy
        ),

        "precision_weighted": float(
            precision
        ),

        "recall_weighted": float(
            recall
        ),

        "f1_weighted": float(
            f1_weighted
        ),

        "f1_macro": float(
            f1_macro
        ),

        "evaluation_time_seconds": float(
            evaluation_time
        ),

        "classification_report":
            report_dict,
    }

    return (
        metrics,
        matrix,
    )


# ---------------------------------------------------------------------
# Save Evaluation Results
# ---------------------------------------------------------------------

def save_results(
    metrics,
    confusion_matrix_data,
):
    """
    Save test-set evaluation metrics as JSON
    and confusion matrix as CSV.
    """

    print(
        "\n" + "=" * 70
    )

    print(
        "Saving Evaluation Results"
    )

    print(
        "=" * 70
    )

    # -------------------------------------------------------------
    # Ensure metrics directory exists
    # -------------------------------------------------------------

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------
    # Save Metrics
    # -------------------------------------------------------------

    with open(
        METRICS_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4,
        )

    print(
        f"\nMetrics saved at:\n"
        f"{METRICS_FILE}"
    )

    # -------------------------------------------------------------
    # Save Confusion Matrix
    # -------------------------------------------------------------

    matrix_df = pd.DataFrame(
        confusion_matrix_data,
        index=[
            "Actual_1",
            "Actual_2",
            "Actual_3",
            "Actual_4",
        ],
        columns=[
            "Predicted_1",
            "Predicted_2",
            "Predicted_3",
            "Predicted_4",
        ],
    )

    matrix_df.to_csv(
        CONFUSION_MATRIX_FILE
    )

    print(
        f"\nConfusion matrix saved at:\n"
        f"{CONFUSION_MATRIX_FILE}"
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    """
    Execute the complete model evaluation pipeline.
    """

    print("\n")

    print(
        "=" * 70
    )

    print(
        "SafeRoute AI - Final Model Evaluation"
    )

    print(
        "=" * 70
    )

    # -------------------------------------------------------------
    # Load dataset
    # -------------------------------------------------------------

    df = load_dataset()

    # -------------------------------------------------------------
    # Prepare features and target
    # -------------------------------------------------------------

    X, y = prepare_data(df)

    # -------------------------------------------------------------
    # Create train/validation/test split
    # -------------------------------------------------------------

    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    ) = create_split(
        X,
        y,
    )

    # -------------------------------------------------------------
    # Load selected model
    # -------------------------------------------------------------

    model = load_model()

    # -------------------------------------------------------------
    # IMPORTANT:
    # Evaluate ONLY on the untouched TEST set.
    # -------------------------------------------------------------

    (
        metrics,
        confusion_matrix_data,
    ) = evaluate_model(
        model,
        X_test,
        y_test,
    )

    # -------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------

    save_results(
        metrics,
        confusion_matrix_data,
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "Final Model Evaluation "
        "Completed Successfully!"
    )

    print(
        "=" * 70
    )


# ---------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------

if __name__ == "__main__":
    main()