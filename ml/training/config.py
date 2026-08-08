"""
config.py

Central configuration for the SafeRoute AI machine learning pipeline.

This file contains:
- Dataset paths
- Model output paths
- Evaluation output paths
- Random seed
- Train/validation/test split ratios
"""

from pathlib import Path


# ---------------------------------------------------------------------
# Project Paths
# ---------------------------------------------------------------------

# Path of ml/
BASE_DIR = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------
# Dataset Paths
# ---------------------------------------------------------------------

# ml/data/processed/
PROCESSED_DATA_DIR = (
    BASE_DIR
    / "data"
    / "processed"
)


# Feature engineered dataset
DATA_FILE = (
    PROCESSED_DATA_DIR
    / "saferoute_features.parquet"
)


# ---------------------------------------------------------------------
# Model and Metrics Paths
# ---------------------------------------------------------------------

# ml/saved_models/
MODEL_DIR = (
    BASE_DIR
    / "saved_models"
)


# ml/metrics/
METRICS_DIR = (
    BASE_DIR
    / "metrics"
)


# Create directories if they don't exist
MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

METRICS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ---------------------------------------------------------------------
# Dataset Configuration
# ---------------------------------------------------------------------

# Target variable
TARGET_COLUMN = "Severity"


# ---------------------------------------------------------------------
# Dataset Split Configuration
# ---------------------------------------------------------------------

"""
For training we are using 70% of the dataset,
for testing and validation we are using 15% each.

This is a common practice in machine learning to ensure
that the model is trained on a sufficient amount of data
while also having enough data for testing and validation
to evaluate its performance.
"""

TRAIN_SIZE = 0.70
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------

RANDOM_STATE = 42


# ---------------------------------------------------------------------
# Model Configuration
# ---------------------------------------------------------------------

# Number of CPU threads where supported
N_JOBS = -1