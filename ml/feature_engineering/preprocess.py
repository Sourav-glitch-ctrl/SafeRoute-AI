"""
    This module is used to clean the raw US Accidents dataset and save a
    processed dataset which will be further used for feature engineering
    and model training.
"""

from pathlib import Path

import polars as pl

# ---------------------------------------------------------------------
# Project Paths
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

RAW_DATA_FILE = RAW_DATA_DIR / "us_accidents.parquet"
OUTPUT_FILE = PROCESSED_DATA_DIR / "saferoute_dataset.parquet"

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Columns to Keep
# ---------------------------------------------------------------------
# We will only keep the following columns from the raw dataset
# for further processing and model training.

KEEP_COLUMNS = [
    "Severity",
    "Start_Time",
    "Start_Lat",
    "Start_Lng",
    "Distance(mi)",
    "Temperature(F)",
    "Humidity(%)",
    "Pressure(in)",
    "Visibility(mi)",
    "Wind_Speed(mph)",
    "Precipitation(in)",
    "Weather_Condition",
    "Junction",
    "Crossing",
    "Railway",
    "Traffic_Signal",
    "Stop",
    "Sunrise_Sunset",
]


# ---------------------------------------------------------------------
# Load Dataset
# ---------------------------------------------------------------------

def load_dataset_polars() -> pl.DataFrame:
    """
    Load the raw dataset from the local parquet file.
    """
    print("Loading raw dataset...")
    df = pl.read_parquet(RAW_DATA_FILE)
    print(f"Dataset Loaded Successfully : {df.height:,} rows")
    return df


# ---------------------------------------------------------------------
# Select Required Columns
# ---------------------------------------------------------------------

def select_columns(df: pl.DataFrame) -> pl.DataFrame:
    """
    Select only the required columns from the raw dataset.
    """
    print("Selecting required columns...")
    return df.select(KEEP_COLUMNS)


# ---------------------------------------------------------------------
# Remove Duplicate Records
# ---------------------------------------------------------------------

def remove_duplicates(df: pl.DataFrame) -> pl.DataFrame:
    """
    Remove duplicate rows from the dataset.
    """
    print("Removing duplicate rows...")
    return df.unique()


# ---------------------------------------------------------------------
# Handle Missing Values
# ---------------------------------------------------------------------

def handle_missing_values(df: pl.DataFrame) -> pl.DataFrame:
    """
    Handle missing values based on observations from EDA.
    Strategy:
    1. Temperature(F)       -> Median
    2. Humidity(%)          -> Median
    3. Pressure(in)         -> Median
    4. Visibility(mi)       -> Median
    5. Wind_Speed(mph)      -> Median
    6. Precipitation(in)    -> 0
    7. Weather_Condition    -> "Unknown"
    8. Sunrise_Sunset       -> "Unknown"
    """
    print("Handling missing values...")
    df = df.with_columns(
        pl.col("Temperature(F)").fill_null(pl.col("Temperature(F)").median()),

        pl.col("Humidity(%)").fill_null(pl.col("Humidity(%)").median()),

        pl.col("Pressure(in)").fill_null(pl.col("Pressure(in)").median()),

        pl.col("Visibility(mi)").fill_null(pl.col("Visibility(mi)").median()),

        pl.col("Wind_Speed(mph)").fill_null(pl.col("Wind_Speed(mph)").median()),

        pl.col("Precipitation(in)").fill_null(0),

        pl.col("Weather_Condition").fill_null("Unknown"),

        pl.col("Sunrise_Sunset").fill_null("Unknown"),
    )
    return df


# ---------------------------------------------------------------------
# Convert Datetime Columns
# ---------------------------------------------------------------------
def convert_datetime(df: pl.DataFrame) -> pl.DataFrame:
    """
    Convert Start_Time into Polars Datetime datatype.
    """

    print("Converting datetime columns...")

    df = df.with_columns(
        pl.col("Start_Time").str.strptime(
            pl.Datetime,
            "%Y-%m-%d %H:%M:%S%.f",
            strict=False,
        )
    )

    return df


# ---------------------------------------------------------------------
# Save Processed Dataset
# ---------------------------------------------------------------------

def save_dataset(df: pl.DataFrame) -> None:
    """
    Save the processed dataset to a parquet file.
    """

    print("Saving processed dataset...")

    df.write_parquet(OUTPUT_FILE)

    print(f"Dataset saved successfully at:\n{OUTPUT_FILE}")


# ---------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------

def main() -> None:
    """
    Execute the complete preprocessing pipeline.
    """

    df = load_dataset_polars()

    original_shape = (df.height, df.width)

    df = select_columns(df)

    df = remove_duplicates(df)

    df = handle_missing_values(df)

    df = convert_datetime(df)

    processed_shape = (df.height, df.width)

    save_dataset(df)

    print("\n" + "=" * 60)
    print("Preprocessing Summary")
    print("=" * 60)
    print(f"Original Shape  : {original_shape}")
    print(f"Processed Shape : {processed_shape}")
    print(f"Output File     : {OUTPUT_FILE}")
    print("=" * 60)

    print("\nPreprocessing Completed Successfully!")


# ---------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------

if __name__ == "__main__":

    try:
        main()

    except Exception as e:
        print(f"\nError: {e}")
        raise