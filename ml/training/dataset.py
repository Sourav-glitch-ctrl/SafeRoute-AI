import sys
from pathlib import Path
import numpy as np
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

    Weather_Category is currently the main categorical feature.

    OrdinalEncoder is used because the dataset contains millions
    of rows and one-hot encoding would unnecessarily increase the
    feature matrix size.

    Unknown categories encountered later during inference
    are encoded as -1.
    """

    print("\nEncoding categorical features...")

    # ---------------------------------------------------------
    # Find categorical columns
    # ---------------------------------------------------------

    categorical_columns = (
        X.select_dtypes(
            include=["object", "string", "category", "str"]
        )
        .columns
        .tolist()
    )

    if not categorical_columns:
        print("No categorical features found.")

        return (
            X,
            None,
            [],
        )

    print(
        f"Categorical Features : "
        f"{categorical_columns}"
    )

    # ---------------------------------------------------------
    # Create encoder
    # ---------------------------------------------------------

    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )

    # ---------------------------------------------------------
    # Convert categorical columns to normal Python strings
    #
    # This avoids Pandas ArrowStringArray assignment errors.
    # ---------------------------------------------------------

    for column in categorical_columns:
        X[column] = X[column].astype("object")

    # ---------------------------------------------------------
    # Fit and transform categorical columns
    # ---------------------------------------------------------

    encoded_values = encoder.fit_transform(
        X[categorical_columns]
    )

    # ---------------------------------------------------------
    # Assign encoded values column-by-column
    #
    # This is safer than:
    # X.loc[:, categorical_columns] = encoded_values
    # because Pandas may preserve the original Arrow dtype.
    # ---------------------------------------------------------

    for index, column in enumerate(categorical_columns):

        X[column] = encoded_values[:, index].astype("float32")

    # ---------------------------------------------------------
    # Final verification
    # ---------------------------------------------------------

    print("\nCategorical encoding completed.")

    print("\nEncoded categorical dtypes:")

    for column in categorical_columns:
        print(
            f"{column} : "
            f"{X[column].dtype}"
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

# Create a Stratified Subset for Random Forest Training
def get_random_forest_training_data(
    X_train,
    y_train,
    sample_size=1_500_000,
    random_state=42
):
    """
    Create a stratified subset of the training data for Random Forest.

    Random Forest requires considerably more memory than the other
    models when trained on the complete 5.4M-row training dataset.

    Therefore:
    - The original training dataset remains unchanged.
    - Validation and testing datasets remain unchanged.
    - Only Random Forest uses this reduced training subset.
    - Stratified sampling preserves the original Severity distribution.

    Parameters
    ----------
    X_train :
        Complete training feature dataset.

    y_train :
        Complete training target dataset.

    sample_size :
        Number of samples to use for Random Forest training.

    random_state :
        Random seed for reproducibility.

    Returns
    -------
    X_rf :
        Stratified Random Forest training features.

    y_rf :
        Stratified Random Forest training target.
    """

    print("\nCreating Random Forest training subset...")

    total_rows = len(X_train)

    # -------------------------------------------------------------
    # Use complete training data if requested sample is larger
    # -------------------------------------------------------------

    if sample_size >= total_rows:

        print(
            "Requested sample size is greater than or equal "
            "to the full training dataset."
        )

        print("Using complete training dataset.")

        return X_train, y_train

    # -------------------------------------------------------------
    # Convert target to Pandas Series
    # -------------------------------------------------------------

    y_series = y_train.reset_index(drop=True)

    # Reset feature index so both datasets have matching indices
    X_reset = X_train.reset_index(drop=True)

    # -------------------------------------------------------------
    # Calculate number of samples for each Severity class
    # -------------------------------------------------------------

    class_counts = y_series.value_counts(
        sort=False
    )

    class_proportions = (
        class_counts / total_rows
    )

    class_sample_counts = (
        class_proportions * sample_size
    ).round().astype(int)

    # Make sure every class gets at least one sample
    class_sample_counts = class_sample_counts.clip(
        lower=1
    )

    # -------------------------------------------------------------
    # Correct rounding difference
    # -------------------------------------------------------------

    difference = (
        sample_size -
        class_sample_counts.sum()
    )

    if difference != 0:

        largest_class = class_counts.idxmax()

        class_sample_counts.loc[largest_class] += difference

    # -------------------------------------------------------------
    # Generate stratified indices
    # -------------------------------------------------------------

    sampled_indices = []

    for severity, n_samples in class_sample_counts.items():

        class_indices = y_series[
            y_series == severity
        ].index

        sampled_class_indices = (
            class_indices
            .to_series()
            .sample(
                n=min(
                    n_samples,
                    len(class_indices)
                ),
                random_state=random_state
            )
            .to_list()
        )

        sampled_indices.extend(
            sampled_class_indices
        )

    # -------------------------------------------------------------
    # Shuffle sampled indices
    # -------------------------------------------------------------

    sampled_indices = (
        pl.Series(sampled_indices)
        if False
        else sampled_indices
    )

    import numpy as np

    rng = np.random.default_rng(
        random_state
    )

    rng.shuffle(
        sampled_indices
    )

    # -------------------------------------------------------------
    # Create Random Forest dataset
    # -------------------------------------------------------------

    X_rf = X_reset.iloc[
        sampled_indices
    ].copy()

    y_rf = y_series.iloc[
        sampled_indices
    ].copy()

    # -------------------------------------------------------------
    # Reset indices
    # -------------------------------------------------------------

    X_rf = X_rf.reset_index(
        drop=True
    )

    y_rf = y_rf.reset_index(
        drop=True
    )

    # -------------------------------------------------------------
    # Display information
    # -------------------------------------------------------------

    print("\nRandom Forest Training Subset")
    print("--------------------------------")

    print(
        f"Original Training Rows : "
        f"{total_rows:,}"
    )

    print(
        f"Random Forest Rows     : "
        f"{len(X_rf):,}"
    )

    print("\nSeverity Distribution:")

    print(
        y_rf
        .value_counts(
            normalize=True
        )
        .sort_index()
    )

    print("\nRandom Forest subset created successfully.")

    return X_rf, y_rf



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