
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
df = pd.read_csv(BASE_DIR / "Car_Data.csv")
df.columns = df.columns.str.strip()

numeric_columns = ["year", "selling_price", "km_driven"]

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

print("=" * 60)
print("1. EXTREME VALUES")
print("=" * 60)

for column in numeric_columns:
    print(f"\n{column}")
    print(df[column].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]))

print("\n" + "=" * 60)
print("2. IQR OUTLIER COUNTS")
print("=" * 60)

for column in ["selling_price", "km_driven"]:
    values = df[column].dropna()

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = df[
        (df[column] < lower_bound) |
        (df[column] > upper_bound)
    ]

    print(f"\n{column}")
    print(f"Q1: {q1:,.2f}")
    print(f"Q3: {q3:,.2f}")
    print(f"IQR: {iqr:,.2f}")
    print(f"Lower bound: {lower_bound:,.2f}")
    print(f"Upper bound: {upper_bound:,.2f}")
    print(f"Outlier rows: {len(outliers)}")

print("\n" + "=" * 60)
print("3. MOST EXPENSIVE CARS")
print("=" * 60)

print(
    df.nlargest(15, "selling_price")[
        ["name", "year", "selling_price", "km_driven",
         "fuel", "transmission", "owner"]
    ].to_string(index=False)
)

print("\n" + "=" * 60)
print("4. HIGHEST MILEAGE CARS")
print("=" * 60)

print(
    df.nlargest(15, "km_driven")[
        ["name", "year", "selling_price", "km_driven",
         "fuel", "transmission", "owner"]
    ].to_string(index=False)
)

print("\n" + "=" * 60)
print("5. YEAR VALIDATION")
print("=" * 60)

current_year = pd.Timestamp.now().year

print("Years before 1990:", (df["year"] < 1990).sum())
print("Years after current year:", (df["year"] > current_year).sum())

print("\nNo rows have been deleted or modified.")
