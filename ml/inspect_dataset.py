from pathlib import Path
import polars as pl

BASE_DIR = Path(__file__).resolve().parent

RAW_FILE = (
    BASE_DIR /
    "data" /
    "raw" /
    "us_accidents.parquet"
)

df = pl.read_parquet(RAW_FILE)

print("=" * 80)
print("Loading dataset...")
print("=" * 80)

print("\nDataset Summary")
print("-" * 80)
print(df)

print("\nNumber of rows:")
print(len(df))

print("\nNumber of columns:")
print(len(df.columns))

print("\nColumn Names:")
for col in df.columns:
    print(f"- {col}")

print("\nFirst Record:")
print(df[0])