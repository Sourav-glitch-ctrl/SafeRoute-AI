from pathlib import Path

import polars as pl
from datasets import load_dataset

# -----------------------------
# Paths
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent

RAW_DATA_DIR = BASE_DIR / "data" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = RAW_DATA_DIR / "us_accidents.parquet"

# -----------------------------
# Download Dataset
# -----------------------------

print("Downloading dataset...")

dataset = load_dataset("yuvidhepe/us-accidents-updated")

print("Converting to Polars...")

df = pl.from_arrow(dataset["train"].data.table)

print("Saving raw dataset...")

df.write_parquet(OUTPUT_FILE)

print(f"Dataset saved to:\n{OUTPUT_FILE}")

print(f"\nRows    : {df.height:,}")
print(f"Columns : {df.width}")