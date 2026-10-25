
from pathlib import Path
import pandas as pd

# Locate the CSV beside this script.
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Car_Data.csv"

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

print("=" * 60)
print("1. DATASET OVERVIEW")
print("=" * 60)

print("Shape (rows, columns):", df.shape)
print("\nColumn names:")
print(df.columns.tolist())

print("\nData types and non-null counts:")
df.info()

print("\nFirst five rows:")
print(df.head())

print("\n" + "=" * 60)
print("2. MISSING VALUES")
print("=" * 60)

missing = df.isnull().sum()
missing_percent = (missing / len(df)) * 100

missing_report = pd.DataFrame({
    "missing_count": missing,
    "missing_percent": missing_percent.round(2),
})

print(missing_report)

print("\n" + "=" * 60)
print("3. DUPLICATE ROWS")
print("=" * 60)

duplicate_count = df.duplicated().sum()
duplicate_group_rows = df.duplicated(keep=False).sum()
deduplicated_df = df.drop_duplicates()

print("Total rows:", len(df))
print("Repeated copies beyond the first:", duplicate_count)
print("Rows belonging to duplicate groups:", duplicate_group_rows)
print("Rows after removing exact duplicates:", len(deduplicated_df))

print("\nExamples of duplicated records:")
duplicate_examples = df[df.duplicated(keep=False)].sort_values(
    by=["name", "year", "selling_price"]
)
print(duplicate_examples.head(15).to_string(index=False))


duplicate_count = df.duplicated().sum()

# Count every row that belongs to a duplicate group.
duplicate_group_rows = df.duplicated(keep=False).sum()

# Calculate how many rows would remain after removing exact duplicates.
deduplicated_df = df.drop_duplicates()

print("Total rows:", len(df))
print("Repeated copies beyond the first:", duplicate_count)
print("Rows belonging to duplicate groups:", duplicate_group_rows)
print("Rows after removing exact duplicates:", len(deduplicated_df))
print("Unique rows removed:", len(df) - len(deduplicated_df))

print("\nExamples of duplicated records:")
duplicate_examples = df[df.duplicated(keep=False)].sort_values(
    by=["name", "year", "selling_price"]
)
print(duplicate_examples.head(15).to_string(index=False))


print("\n" + "=" * 60)
print("4. NUMERIC DATA SUMMARY")
print("=" * 60)

print(df.describe().T)

print("\n" + "=" * 60)
print("5. TARGET PRICE CHECK")
print("=" * 60)

if "selling_price" in df.columns:
    prices = pd.to_numeric(df["selling_price"], errors="coerce")

    print("Invalid or missing prices:", prices.isna().sum())
    print("Zero or negative prices:", (prices <= 0).sum())
    print("Minimum price:", prices.min())
    print("Median price:", prices.median())
    print("Maximum price:", prices.max())

    print("\nLowest 10 prices:")
    print(prices.nsmallest(10))

    print("\nHighest 10 prices:")
    print(prices.nlargest(10))

print("\n" + "=" * 60)
print("6. CATEGORICAL VALUE COUNTS")
print("=" * 60)

for column in df.select_dtypes(include=["object", "str"]).columns:
    print(f"\n{column}: {df[column].nunique()} unique values")
    print(df[column].value_counts(dropna=False).head(10))

print("\nAudit complete. No rows have been modified or deleted.")
