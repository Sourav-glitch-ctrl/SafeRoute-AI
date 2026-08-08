import sys
from pathlib import Path

import pandas as pd
import polars as pl

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

TRAINING_DIR = Path(__file__).resolve().parent

if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

### Import configuration constants from config.py
from config import (
    DATA_FILE,
    TARGET_COLUMN,
    RANDOM_STATE,
    TRAIN_SIZE,
    VALIDATION_SIZE,
    TEST_SIZE,
)

## Load the dataset and prepare it for training, validation, and testing.
def load_dataset() -> pd.DataFrame:
    """
    Load the feature-engineered dataset.
    The dataset is stored as a Parquet file and is converted
    to a pandas DataFrame for compatibility with scikit-learn.
    """
    print("=" * 70)
    print("Loading Feature Engineered Dataset")
    print("=" * 70)
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Feature dataset not found:\n{DATA_FILE}"
        )
    df = pl.read_parquet(DATA_FILE).to_pandas()
    print(
        f"Dataset Shape : {df.shape}"
    )

    return df

# Separate Features and Target

def separate_features_target(
    df: pd.DataFrame,
):
    """
    Separate the target variable from the input features.
    The target is Severity, which is a multiclass classification problem.
    X = input features, y = target variable.
    """
    print("\nSeparating features and target...")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' "
            f"not found in dataset."
        )

    # Create X by removing the target column.
    X = df.drop(columns=[TARGET_COLUMN])
    # Target is kept separately.
    y = df[TARGET_COLUMN]

    print(f"Features Shape : {X.shape}")

    print(f"Target Shape   : {y.shape}")

    return X, y

# Encode Categorical Features

def encode_categorical_features(
    X: pd.DataFrame,
):
    """
    Encode categorical features into numerical values.

    Currently Weather_Category is the main categorical feature.

    OrdinalEncoder is used because the dataset contains millions
    of rows and we want to avoid unnecessarily expanding the
    feature matrix through one-hot encoding.

    Unknown categories encountered later during inference are
    encoded as -1.
    """

    print("\nEncoding categorical features...")

    categorical_columns = (
        X.select_dtypes(
            include=["str", "category"]
        )
        .columns
        .tolist()
    )

    if not categorical_columns:
        print(
            "No categorical features found."
        )
        return X, None, []

    print(
        f"Categorical Features : "
        f"{categorical_columns}"
    )

    # Creating encoder
    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )

    # Fit and transform categorical columns

    # Convert categorical columns to a mutable dtype so we can assign
    # numeric encoded values into them (avoid Arrow string dtype issues).
    X[categorical_columns] = X[categorical_columns].astype(object)

    X.loc[:, categorical_columns] = (
        encoder.fit_transform(
            X[categorical_columns]
        )
    )

    return (
        X,
        encoder,
        categorical_columns,
    )

# Train / Validation / Test Split

def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
):
    """
    Split the dataset into training, validation and testing sets.
    """

    print("\nSplitting dataset...")
    # First split
    X_train, X_temp, y_train, y_temp = (
        train_test_split(
            X,
            y,
            test_size=(1 - TRAIN_SIZE),
            random_state=RANDOM_STATE,
            stratify=y,
        )
    )
    # Second split
    temp_test_ratio = (
        TEST_SIZE
        / (VALIDATION_SIZE + TEST_SIZE)
    )

    X_val, X_test, y_val, y_test = (
        train_test_split(
            X_temp,
            y_temp,
            test_size=temp_test_ratio,
            random_state=RANDOM_STATE,
            stratify=y_temp,
        )
    )

    X_train = X_train.reset_index(
        drop=True
    )

    X_val = X_val.reset_index(
        drop=True
    )

    X_test = X_test.reset_index(
        drop=True
    )

    y_train = y_train.reset_index(
        drop=True
    )

    y_val = y_val.reset_index(
        drop=True
    )

    y_test = y_test.reset_index(
        drop=True
    )

    # Display split sizes

    print("\nSplit Shapes")
    print("-" * 70)

    print(
        f"Training   : {X_train.shape}"
    )

    print(
        f"Validation : {X_val.shape}"
    )

    print(
        f"Testing    : {X_test.shape}"
    )

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    )

# Display Class Distribution


def display_class_distribution(
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
):
    """
    Display Severity class distribution in each dataset split.

    This confirms that stratified splitting has preserved the
    original class distribution.
    """

    print("\nSeverity Distribution")
    print("-" * 70)

    print("\nTraining:")

    print(
        y_train
        .value_counts(normalize=True)
        .sort_index()
    )

    print("\nValidation:")

    print(
        y_val
        .value_counts(normalize=True)
        .sort_index()
    )

    print("\nTesting:")

    print(
        y_test
        .value_counts(normalize=True)
        .sort_index()
    )


# ---------------------------------------------------------------------
# Main Dataset Preparation
# ---------------------------------------------------------------------

def prepare_dataset():
    """
    Execute the complete dataset preparation pipeline.
    """

    # Load the dataset

    df = load_dataset()
    # Separate features and target
    X, y = separate_features_target(df)

    # Encode categorical features

    (X, encoder, categorical_columns,) = encode_categorical_features(X)
    del df
    # Split the dataset into training, validation, and testing sets
    (X_train,X_val,X_test,y_train,y_val,y_test,) = split_dataset(X,y,)

    # Release the complete X and y objects after splitting.

    del X
    del y

    # Display class distribution

    display_class_distribution(
        y_train,
        y_val,
        y_test,
    )
    print(
        "\n" + "=" * 70
    )

    print(
        "Dataset Preparation "
        "Completed Successfully!"
    )

    print(
        "=" * 70
    )

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        encoder,
        categorical_columns,
    )


# ---------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------

if __name__ == "__main__":

    prepare_dataset()