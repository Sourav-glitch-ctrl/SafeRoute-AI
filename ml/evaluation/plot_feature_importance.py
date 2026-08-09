import os

import pandas as pd
import matplotlib.pyplot as plt


INPUT_FILE = "ml/metrics/random_forest_feature_importance.csv"
OUTPUT_FILE = "ml/metrics/random_forest_feature_importance.png"


def main():

    print("\nLoading feature importance...")

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Feature importance file not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    df = df.sort_values(
        by="Importance",
        ascending=True,
    )

    top_n = 15

    plot_df = df.tail(top_n)

    plt.figure(figsize=(10, 7))

    plt.barh(
        plot_df["Feature"],
        plot_df["Importance"],
    )

    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")

    plt.title(
        "Random Forest - Top 15 Feature Importance"
    )

    plt.tight_layout()

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True,
    )

    plt.savefig(
        OUTPUT_FILE,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"\nFeature importance plot saved at:\n"
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()