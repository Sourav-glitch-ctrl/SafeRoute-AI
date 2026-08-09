"""
SafeRoute AI
Random Forest Feature Importance

Extracts and saves feature importance from the trained
Random Forest model.
"""

import os
import joblib
import pandas as pd


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

MODEL_FILE = "ml/saved_models/random_forest.pkl"

OUTPUT_FILE = "ml/metrics/random_forest_feature_importance.csv"


# ---------------------------------------------------------
# Load Model
# ---------------------------------------------------------

def load_model():

    print("\nLoading Random Forest model...")

    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(
            f"Model not found:\n{MODEL_FILE}"
        )

    model = joblib.load(MODEL_FILE)

    print("Model loaded successfully.")

    return model


# ---------------------------------------------------------
# Extract Feature Importance
# ---------------------------------------------------------

def get_feature_importance(model):

    print("\nExtracting feature importance...")

    if not hasattr(model, "feature_importances_"):
        raise AttributeError(
            "The loaded model does not provide "
            "feature_importances_."
        )

    importance = model.feature_importances_

    # Try to recover feature names from the model.
    # sklearn RandomForest normally stores them when
    # trained using a DataFrame.
    if hasattr(model, "feature_names_in_"):

        feature_names = model.feature_names_in_

    else:

        raise AttributeError(
            "Feature names were not stored inside the model."
        )

    if len(feature_names) != len(importance):

        raise ValueError(
            "Number of feature names does not match "
            "number of feature importance values."
        )

    importance_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": importance,
        }
    )

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False,
    ).reset_index(drop=True)

    return importance_df


# ---------------------------------------------------------
# Save Results
# ---------------------------------------------------------

def save_results(importance_df):

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True,
    )

    importance_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"\nFeature importance saved at:\n"
        f"{OUTPUT_FILE}"
    )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("\n" + "=" * 70)
    print("SafeRoute AI - Random Forest Feature Importance")
    print("=" * 70)

    model = load_model()

    importance_df = get_feature_importance(model)

    print("\n## Feature Importance")

    print(
        importance_df.to_string(index=False)
    )

    save_results(importance_df)

    print("\n## Top 10 Features")

    print(
        importance_df.head(10).to_string(index=False)
    )

    print("\nFeature importance analysis completed.")


if __name__ == "__main__":
    main()