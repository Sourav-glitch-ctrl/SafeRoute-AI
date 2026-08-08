from pathlib import Path
import polars as pl
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

#Importing configuration variables from config.py
from config import (
    DATA_FILE,
    TARGET_COLUMN,
    TRAIN_SIZE, 
    VALIDATION_SIZE,
    TEST_SIZE,
    RANDOM_STATE
)

## Loading the dataset
def load_dataset() -> pd.DataFrame:
    """ Here we will load the featured engineered 
    dataset from the processed data directory.
    """
    print("=" * 70)
    print("Loading featured Engineered dataset")
    print("=" * 70)

    df = pl.read_parquet(DATA_FILE).to_pandas()
    print(f"Dataset Shape: {df.shape}")
    return df

## Separate Features and Target Variable
def separate_features_target(
    df: pd.DataFrame,
):
    """
    Separate the target variable from the input features.
    So here target variable is 'Severity' and all other columns are features.
    X = input features, y = target variable
    """

    print("\nSeparating features and target...")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found in dataset."
        )

    X = df.drop(
        columns=[TARGET_COLUMN]
    )

    y = df[TARGET_COLUMN]

    print(f"Features Shape : {X.shape}")
    print(f"Target Shape   : {y.shape}")

    return X, y

## Encode categorical fetaures

def encode_categorical_features(
    X: pd.DataFrame,
):
    """
    Encode categorical features into numerical values.
    Currently Weather_Category is the main categorical feature.
    Unknown categories encountered later during inference are encoded safely using the 'use_encoded_value' strategy.
    """

    print("\nEncoding categorical features...")

    categorical_columns = X.select_dtypes(
        include=["str", "category"]
    ).columns.tolist()

    if not categorical_columns:
        print("No categorical features found.")

        return X, None, categorical_columns

    print(
        f"Categorical Features : {categorical_columns}"
    )

    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1
    )

    X[categorical_columns] = encoder.fit_transform(
        X[categorical_columns]
    )

    return X, encoder, categorical_columns

## Train, validate and test split

def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
):
   
    print("\nSplitting dataset...")

    # First split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=(1 - TRAIN_SIZE),
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Second split
    temp_test_ratio = (
        TEST_SIZE / (VALIDATION_SIZE + TEST_SIZE)
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=temp_test_ratio,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    print("\nSplit Shapes")
    print("-" * 70)

    print(f"Training   : {X_train.shape}")
    print(f"Validation : {X_val.shape}")
    print(f"Testing    : {X_test.shape}")

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
    Display Severity distribution in each dataset split.
    """

    print("\nSeverity Distribution")
    print("-" * 70)

    print("\nTraining:")
    print(y_train.value_counts(normalize=True).sort_index())

    print("\nValidation:")
    print(y_val.value_counts(normalize=True).sort_index())

    print("\nTesting:")
    print(y_test.value_counts(normalize=True).sort_index())


# ---------------------------------------------------------------------
# Main Dataset Preparation
# ---------------------------------------------------------------------

def prepare_dataset():
    """
    Execute the complete dataset preparation pipeline, and it returns the training, validation and testing datasets, together with the fitted categorical encoder.
    """
    df = load_dataset()
    X, y = separate_features_target(df)
    X, encoder, categorical_columns = (
        encode_categorical_features(X)
    )

    (X_train, X_val, X_test, y_train, y_val, y_test,) = split_dataset(X, y)

    display_class_distribution(
        y_train,
        y_val,
        y_test,
    )

    print("\n" + "=" * 70)
    print("Dataset Preparation Completed Successfully!")
    print("=" * 70)

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
