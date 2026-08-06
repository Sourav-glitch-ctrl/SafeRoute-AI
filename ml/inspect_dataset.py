from datasets import load_dataset

print("=" * 80)
print("Loading dataset...")
print("=" * 80)

dataset = load_dataset("yuvidhepe/us-accidents-updated")

train = dataset["train"]

print("\nDataset Summary")
print("-" * 80)
print(dataset)

print("\nNumber of rows:")
print(len(train))

print("\nNumber of columns:")
print(len(train.column_names))

print("\nColumn Names:")
for col in train.column_names:
    print(f"- {col}")

print("\nFirst Record:")
print(train[0])