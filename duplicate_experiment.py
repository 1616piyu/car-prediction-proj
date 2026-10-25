
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).resolve().parent
df = pd.read_csv(BASE_DIR / "Car_Data.csv")
df.columns = df.columns.str.strip()

required_columns = [
    "name", "year", "selling_price", "km_driven",
    "fuel", "seller_type", "transmission", "owner"
]

df = df[required_columns].copy()

# Apply the same basic cleaning to both experiments.
for column in ["year", "selling_price", "km_driven"]:
    df[column] = pd.to_numeric(df[column], errors="coerce")

df = df.dropna(subset=["selling_price"])
df = df[df["selling_price"] > 0]

# Create the two datasets.
datasets = {
    "Original dataset": df,
    "Deduplicated dataset": df.drop_duplicates().copy(),
}

numeric_features = ["year", "km_driven"]
categorical_features = [
    "name", "fuel", "seller_type", "transmission", "owner"
]

def create_preprocessor():
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])

results = []

for dataset_name, data in datasets.items():
    X = data.drop(columns=["selling_price"])
    y = data["selling_price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    models = {
        "Median baseline": DummyRegressor(strategy="median"),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            min_samples_leaf=2,
            n_jobs=-1,
        ),
    }

    print("\n" + "=" * 60)
    print(dataset_name)
    print("=" * 60)
    print("Total rows:", len(data))
    print("Training rows:", len(X_train))
    print("Test rows:", len(X_test))
    print("Exact duplicate rows remaining:", data.duplicated().sum())

    for model_name, estimator in models.items():
        pipeline = Pipeline([
            ("preprocessor", create_preprocessor()),
            ("model", estimator),
        ])

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = mean_squared_error(y_test, predictions) ** 0.5
        r2 = r2_score(y_test, predictions)

        results.append({
            "Dataset": dataset_name,
            "Model": model_name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
        })

        print(f"\n{model_name}")
        print(f"MAE:  {mae:,.2f}")
        print(f"RMSE: {rmse:,.2f}")
        print(f"R²:   {r2:.4f}")

print("\n" + "=" * 60)
print("COMPARISON SUMMARY")
print("=" * 60)

summary = pd.DataFrame(results)
print(summary.to_string(index=False, formatters={
    "MAE": "{:,.2f}".format,
    "RMSE": "{:,.2f}".format,
    "R2": "{:.4f}".format,
}))