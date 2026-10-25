
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).resolve().parent
df = pd.read_csv(BASE_DIR / "Car_Data.csv")
df.columns = df.columns.str.strip()

numeric_columns = ["year", "selling_price", "km_driven"]

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

df = df.dropna(subset=numeric_columns)
df = df[df["selling_price"] > 0]
df = df.drop_duplicates().copy()

# Calculate IQR boundaries using this cleaned dataset.
def get_iqr_bounds(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr

price_low, price_high = get_iqr_bounds(df["selling_price"])
km_low, km_high = get_iqr_bounds(df["km_driven"])

# Flag a row if price OR mileage falls outside its IQR bounds.
outlier_mask = (
    (df["selling_price"] < price_low)
    | (df["selling_price"] > price_high)
    | (df["km_driven"] < km_low)
    | (df["km_driven"] > km_high)
)

datasets = {
    "All deduplicated rows": df,
    "Without IQR-flagged rows": df.loc[~outlier_mask].copy(),
}

numeric_features = ["year", "km_driven"]
categorical_features = [
    "name", "fuel", "seller_type", "transmission", "owner"
]

def create_pipeline():
    preprocessor = ColumnTransformer([
        (
            "numeric",
            SimpleImputer(strategy="median"),
            numeric_features,
        ),
        (
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categorical_features,
        ),
    ])

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        min_samples_leaf=2,
        n_jobs=-1,
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])

for label, data in datasets.items():
    X = data.drop(columns=["selling_price"])
    y = data["selling_price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    model = create_pipeline()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    print("\n" + "=" * 60)
    print(label)
    print("=" * 60)
    print("Dataset rows:", len(data))
    print("Training rows:", len(X_train))
    print("Test rows:", len(X_test))
    print(f"Test MAE:  ₹{mae:,.2f}")
    print(f"Test RMSE: ₹{rmse:,.2f}")
    print(f"Test R²:   {r2:.4f}")

print("\nIQR price bounds:", f"₹{price_low:,.2f}", "to", f"₹{price_high:,.2f}")
print("IQR mileage bounds:", f"{km_low:,.0f}", "to", f"{km_high:,.0f}")
print("Rows flagged by either rule:", int(outlier_mask.sum()))
print("No source data has been modified.")
