"""
models.py

Defines the machine learning models used in the SafeRoute AI
road accident severity prediction pipeline.
So the models that we are going to use are:
Models:
    1. Logistic Regression
    2. Random Forest
    3. XGBoost
    4. LightGBM
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from config import RANDOM_STATE, N_JOBS


# ---------------------------------------------------------------------
# Logistic Regression
# ---------------------------------------------------------------------

def get_logistic_regression():
    """
    Create the Logistic Regression baseline model.
    class_weight='balanced' is used because the Severity target
    is highly imbalanced.
    """

    return LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )


# ---------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------

def get_random_forest():
    """
    Create the Random Forest classifier.
    Balanced class weights are used to give greater importance
    to minority Severity classes.
    """

    return RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        max_depth=None,
    )


# ---------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------

def get_xgboost():
    """
    Create the XGBoost multiclass classifier.
    XGBoost does not directly use class_weight like
    RandomForest and LogisticRegression.
    Class balancing will be handled separately during
    the training pipeline if required.
    """

    return XGBClassifier(
    objective="multi:softprob",
    num_class=4,
    n_estimators=300,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
    n_jobs=N_JOBS,
    eval_metric="mlogloss",
    tree_method="hist",
)

# ---------------------------------------------------------------------
# LightGBM
# ---------------------------------------------------------------------

def get_lightgbm():
    """
    Create the LightGBM multiclass classifier.

    LightGBM is included because it is highly efficient for
    large tabular datasets and will be compared against XGBoost.
    """

    return LGBMClassifier(
        objective="multiclass",
        num_class=4,
        n_estimators=300,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        verbosity=-1,
    )


# ---------------------------------------------------------------------
# Model Registry
# ---------------------------------------------------------------------

def get_models():
    """
    Return all models used in the SafeRoute AI experiment.

    Returns:
        dict:
            Dictionary containing model names and model objects.
    """

    models = {
        "Logistic Regression": get_logistic_regression(),
        "Random Forest": get_random_forest(),
        "XGBoost": get_xgboost(),
        "LightGBM": get_lightgbm(),
    }

    return models


# ---------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------

if __name__ == "__main__":

    print("=" * 70)
    print("SafeRoute AI - Model Configuration")
    print("=" * 70)

    models = get_models()

    for name, model in models.items():
        print(f"\n{name}")
        print("-" * 70)
        print(model)

    print("\n" + "=" * 70)
    print("Model Configuration Loaded Successfully!")
    print("=" * 70)