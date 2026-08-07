"""
features.py

Purpose:
    This module is responsible for creating meaningful features from the
    cleaned accident dataset. The engineered features are later used for
    machine learning model training.
"""

from pathlib import Path

import polars as pl

# ---------------------------------------------------------------------
# Project Paths
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

INPUT_FILE = PROCESSED_DATA_DIR / "saferoute_dataset.parquet"

OUTPUT_FILE = PROCESSED_DATA_DIR / "saferoute_features.parquet"

# ---------------------------------------------------------------------
# Load Dataset
# ---------------------------------------------------------------------


def load_dataset() -> pl.DataFrame:
    """
    Load the processed dataset generated from preprocess.py.
    """

    print("Loading processed dataset...")

    df = pl.read_parquet(INPUT_FILE)

    print(f"Dataset Loaded Successfully : {df.height:,} rows")

    return df


# ---------------------------------------------------------------------
# Weather Categorization
# ---------------------------------------------------------------------


def categorize_weather(weather: str) -> str:
    """
    Convert 145+ detailed weather conditions into broader categories.
    """

    if weather is None:
        return "Unknown"

    weather = weather.lower()

    # ---------------- Clear ----------------

    if any(x in weather for x in [
        "fair",
        "clear",
        "sunny"
    ]):
        return "Clear"

    # ---------------- Cloudy ----------------

    elif any(x in weather for x in [
        "cloud",
        "overcast",
        "partly cloudy",
        "mostly cloudy",
        "scattered clouds"
    ]):
        return "Cloudy"

    # ---------------- Rain ----------------

    elif any(x in weather for x in [
        "rain",
        "drizzle",
        "shower"
    ]):
        return "Rain"

    # ---------------- Snow ----------------

    elif any(x in weather for x in [
        "snow",
        "snow grains"
    ]):
        return "Snow"

    # ---------------- Ice ----------------

    elif any(x in weather for x in [
        "freezing",
        "ice",
        "ice pellets",
        "sleet",
        "wintry mix"
    ]):
        return "Ice"

    # ---------------- Fog ----------------

    elif any(x in weather for x in [
        "fog",
        "mist",
        "haze",
        "shallow fog",
        "partial fog"
    ]):
        return "Fog"

    # ---------------- Storm ----------------

    elif any(x in weather for x in [
        "thunder",
        "storm",
        "t-storm",
        "hail",
        "tornado",
        "funnel cloud"
    ]):
        return "Storm"

    # ---------------- Wind ----------------

    elif any(x in weather for x in [
        "wind",
        "windy",
        "blowing",
        "squalls"
    ]):
        return "Wind"

    # ---------------- Smoke ----------------

    elif "smoke" in weather:
        return "Smoke"

    # ---------------- Dust ----------------

    elif any(x in weather for x in [
        "dust",
        "sand",
        "ash"
    ]):
        return "Dust"

    # ---------------- Unknown ----------------

    elif any(x in weather for x in [
        "unknown",
        "n/a precipitation"
    ]):
        return "Unknown"

    return "Other"

# ---------------------------------------------------------------------
# Time Features
# ---------------------------------------------------------------------

def create_time_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Create meaningful time-based features from Start_Time.
    """

    print("Creating time features...")

    return df.with_columns(

        # Hour of the day (0-23)
        pl.col("Start_Time")
        .dt.hour()
        .alias("Hour"),

        # Day of the week (Monday=1 ... Sunday=7)
        pl.col("Start_Time")
        .dt.weekday()
        .alias("DayOfWeek"),

        # Month (1-12)
        pl.col("Start_Time")
        .dt.month()
        .alias("Month"),

        # Quarter (1-4)
        pl.col("Start_Time")
        .dt.quarter()
        .alias("Quarter"),

        # Year
        pl.col("Start_Time")
        .dt.year()
        .alias("Year"),

        # Weekend
        (
            pl.col("Start_Time")
            .dt.weekday()
            .is_in([6, 7])
        ).alias("IsWeekend"),

        # Morning Rush Hour (7 AM - 10 AM)
        (
            (
                pl.col("Start_Time").dt.hour() >= 7
            )
            &
            (
                pl.col("Start_Time").dt.hour() <= 10
            )
        ).alias("MorningRushHour"),

        # Evening Rush Hour (4 PM - 7 PM)
        (
            (
                pl.col("Start_Time").dt.hour() >= 16
            )
            &
            (
                pl.col("Start_Time").dt.hour() <= 19
            )
        ).alias("EveningRushHour"),

        # Any Rush Hour
        (
            (
                (
                    pl.col("Start_Time").dt.hour() >= 7
                )
                &
                (
                    pl.col("Start_Time").dt.hour() <= 10
                )
            )
            |
            (
                (
                    pl.col("Start_Time").dt.hour() >= 16
                )
                &
                (
                    pl.col("Start_Time").dt.hour() <= 19
                )
            )
        ).alias("IsRushHour")
    )


# ---------------------------------------------------------------------
# Weather Features
# ---------------------------------------------------------------------

def create_weather_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Create simplified weather category feature.
    """

    print("Creating weather features...")

    return df.with_columns(

        pl.col("Weather_Condition")
        .map_elements(
            categorize_weather,
            return_dtype=pl.String
        )
        .alias("Weather_Category")

    )


# ---------------------------------------------------------------------
# Day / Night Feature
# ---------------------------------------------------------------------

def create_day_night_feature(df: pl.DataFrame) -> pl.DataFrame:
    """
    Convert Sunrise_Sunset into a binary feature.
    """

    print("Creating day/night feature...")

    return df.with_columns(

        (
            pl.col("Sunrise_Sunset") == "Night"
        ).alias("IsNight")

    )

# ---------------------------------------------------------------------
# Environmental Features
# ---------------------------------------------------------------------

def create_environmental_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Create additional environmental features.
    """

    print("Creating environmental features...")

    return df.with_columns(

        # -------------------------------------------------------------
        # Whether precipitation was present
        # -------------------------------------------------------------
        (
            pl.col("Precipitation(in)") > 0
        ).alias("HasPrecipitation"),

        # -------------------------------------------------------------
        # Low Visibility
        # Visibility less than 2 miles
        # -------------------------------------------------------------
        (
            pl.col("Visibility(mi)") < 2
        ).alias("LowVisibility")

    )


# ---------------------------------------------------------------------
# Road Features
# ---------------------------------------------------------------------

def create_road_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Create road-related features.
    """

    print("Creating road features...")

    return df.with_columns(

        (
            pl.col("Junction")
            |
            pl.col("Crossing")
            |
            pl.col("Railway")
            |
            pl.col("Traffic_Signal")
            |
            pl.col("Stop")
        ).alias("NearRoadInfrastructure")

    )


# ---------------------------------------------------------------------
# Remove Unnecessary Columns
# ---------------------------------------------------------------------

def drop_unused_columns(df: pl.DataFrame) -> pl.DataFrame:
    """
    Remove columns that are no longer required after
    feature engineering.
    """

    print("Dropping unnecessary columns...")

    return df.drop([
        "Start_Time",
        "Weather_Condition",
        "Sunrise_Sunset"
    ])

# ---------------------------------------------------------------------
# Save Feature Engineered Dataset
# ---------------------------------------------------------------------

def save_dataset(df: pl.DataFrame) -> None:
    """
    Save the feature engineered dataset to a parquet file.
    """

    print("Saving feature engineered dataset...")

    df.write_parquet(OUTPUT_FILE)

    print(f"Dataset saved successfully at:\n{OUTPUT_FILE}")


# ---------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------

def main() -> None:
    """
    Execute the complete feature engineering pipeline.
    """

    print("=" * 70)
    print("SafeRoute AI - Feature Engineering Pipeline")
    print("=" * 70)

    # Load processed dataset
    df = load_dataset()

    original_shape = (df.height, df.width)

    # -------------------------------------------------------------
    # Feature Engineering
    # -------------------------------------------------------------

    df = create_time_features(df)

    df = create_weather_features(df)

    df = create_day_night_feature(df)

    df = create_environmental_features(df)

    df = create_road_features(df)

    df = drop_unused_columns(df)

    processed_shape = (df.height, df.width)

    # -------------------------------------------------------------
    # Save Dataset
    # -------------------------------------------------------------

    save_dataset(df)

    # -------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------

    print("\n" + "=" * 70)
    print("Feature Engineering Summary")
    print("=" * 70)

    print(f"Original Shape  : {original_shape}")
    print(f"Processed Shape : {processed_shape}")

    print(f"\nNew Features Added :")

    print("- Hour")
    print("- DayOfWeek")
    print("- Month")
    print("- Quarter")
    print("- Year")
    print("- IsWeekend")
    print("- MorningRushHour")
    print("- EveningRushHour")
    print("- IsRushHour")
    print("- Weather_Category")
    print("- IsNight")
    print("- HasPrecipitation")
    print("- LowVisibility")
    print("- NearRoadInfrastructure")

    print(f"\nOutput File :")
    print(OUTPUT_FILE)

    print("\nFeature Engineering Completed Successfully!")

    print("=" * 70)


# ---------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------

if __name__ == "__main__":

    try:

        main()

    except Exception as e:

        print("\nFeature Engineering Failed!")

        print(f"Error : {e}")

        raise

