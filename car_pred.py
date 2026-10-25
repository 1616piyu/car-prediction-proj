
from pathlib import Path
import pickle

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor


# --------------------------------------------------
# 1. Paths and settings
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Car_Data.csv"
MODEL_PATH = BASE_DIR / "model.pkl"

# Keep consistent with the Streamlit app and feature engineering.
REFERENCE_YEAR = 2020
TEST_SIZE = 0.20
RANDOM_STATE = 42
N_SPLITS = 5


# --------------------------------------------------
# 2. Load and validate data
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

required_columns = [
    "name",
    "year",
    "selling_price",
    "km_driven",
    "fuel",
    "seller_type",
    "transmission",
    "owner",
]

missing_columns = set(required_columns) - set(df.columns)

if missing_columns:
    raise ValueError(
        f"Missing columns: {sorted(missing_columns)}"
    )

df = df[required_columns].copy()

# Convert numeric columns safely.
for column in ["year", "selling_price", "km_driven"]:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# Prices are required to train a regression model.
df = df.dropna(subset=["selling_price"])
df = df[df["selling_price"] > 0].copy()

if len(df) < 10:
    raise ValueError("Not enough valid rows to train a model.")

# Reset the index so positional splits are easy to follow.
df = df.reset_index(drop=True)

print("=" * 60)
print("DATASET SUMMARY")
print("=" * 60)
print("Rows available:", len(df))
print("Exact duplicate rows:", df.duplicated().sum())
print("Unique original feature combinations:",
      df.drop(columns=["selling_price"]).drop_duplicates().shape[0])


# --------------------------------------------------
# 3. Feature engineering
# --------------------------------------------------

# Calculate age relative to the dataset's reference year.
df["car_age"] = (REFERENCE_YEAR - df["year"]).clip(lower=0)

# Avoid division by zero for cars with age zero.
df["km_per_year"] = (
    df["km_driven"] / df["car_age"].clip(lower=1)
)

# Extract the first word from the car name as a basic brand feature.
df["brand"] = (
    df["name"]
    .fillna("Unknown")
    .astype(str)
    .str.strip()
    .str.split()
    .str[0]
)

target = "selling_price"

original_numeric = ["year", "km_driven"]

original_categorical = [
    "name",
    "fuel",
    "seller_type",
    "transmission",
    "owner",
]

engineered_numeric = [
    "car_age",
    "km_driven",
    "km_per_year",
]

engineered_categorical = [
    "brand",
    "name",
    "fuel",
    "seller_type",
    "transmission",
    "owner",
]

X_original = df[original_numeric + original_categorical].copy()

X_engineered = df[
    engineered_numeric + engineered_categorical
].copy()

y = df[target].copy()


# --------------------------------------------------
# 4. Create duplicate-safe train/test split
# --------------------------------------------------

# Rows with the same original input features receive the same group ID.
# This prevents identical input records from crossing the train/test
# boundary, even if their selling_price labels differ.
groups = pd.util.hash_pandas_object(
    X_original,
    index=False,
).astype("uint64")

number_of_groups = groups.nunique()

if number_of_groups < N_SPLITS + 1:
    raise ValueError(
        "Not enough unique feature groups for the requested "
        "train/test split and cross-validation."
    )

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
)

train_positions, test_positions = next(
    splitter.split(X_original, y, groups=groups)
)

X_train_original = X_original.iloc[train_positions]
X_test_original = X_original.iloc[test_positions]

X_train_engineered = X_engineered.iloc[train_positions]
X_test_engineered = X_engineered.iloc[test_positions]

y_train = y.iloc[train_positions]
y_test = y.iloc[test_positions]

groups_train = groups.iloc[train_positions]
groups_test = groups.iloc[test_positions]

# Verify that no original feature group appears in both sets.
shared_groups = set(groups_train) & set(groups_test)

if shared_groups:
    raise RuntimeError(
        "Data leakage detected: feature groups overlap "
        "between training and test sets."
    )

print("\n" + "=" * 60)
print("DUPLICATE-SAFE SPLIT")
print("=" * 60)
print("Training rows:", len(train_positions))
print("Testing rows:", len(test_positions))
print("Unique training groups:", groups_train.nunique())
print("Unique testing groups:", groups_test.nunique())
print("Groups shared across train/test:", len(shared_groups))


# --------------------------------------------------
# 5. Baseline model
# --------------------------------------------------

# A simple benchmark that predicts the training-set median price
# for every car in the test set.
baseline = DummyRegressor(strategy="median")
baseline.fit(X_train_original, y_train)

baseline_predictions = baseline.predict(X_test_original)

baseline_mae = mean_absolute_error(
    y_test, baseline_predictions
)
baseline_rmse = mean_squared_error(
    y_test, baseline_predictions
) ** 0.5
baseline_r2 = r2_score(y_test, baseline_predictions)

print("\n" + "=" * 60)
print("MEDIAN BASELINE")
print("=" * 60)
print(f"Test MAE:  ₹{baseline_mae:,.2f}")
print(f"Test RMSE: ₹{baseline_rmse:,.2f}")
print(f"Test R²:   {baseline_r2:.4f}")


# --------------------------------------------------
# 6. Preprocessing and candidate models
# --------------------------------------------------

def create_pipeline(
    numeric_features,
    categorical_features,
    estimator,
):
    """Build a pipeline that preprocesses data before modeling."""

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        (
            "numeric",
            numeric_pipeline,
            numeric_features,
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features,
        ),
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", estimator),
    ])


# Each model is freshly created for each experiment.
def create_models():
    return {
        "Linear Regression": LinearRegression(),

        "Decision Tree": DecisionTreeRegressor(
            random_state=RANDOM_STATE,
            min_samples_leaf=2,
        ),

        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            random_state=RANDOM_STATE,
            min_samples_leaf=2,
            n_jobs=-1,
        ),
    }


experiments = {
    "Original features": (
        X_train_original,
        X_test_original,
        original_numeric,
        original_categorical,
    ),

    "Engineered features": (
        X_train_engineered,
        X_test_engineered,
        engineered_numeric,
        engineered_categorical,
    ),
}

results = {}

# Use grouped folds so duplicate feature combinations also remain
# within the same fold during cross-validation.
group_cv = GroupKFold(n_splits=N_SPLITS)


# --------------------------------------------------
# 7. Compare models with duplicate-safe cross-validation
# --------------------------------------------------

for feature_set, (
    X_train,
    X_test,
    numeric_features,
    categorical_features,
) in experiments.items():

    print("\n" + "=" * 60)
    print(feature_set.upper())
    print("=" * 60)

    for model_name, estimator in create_models().items():

        pipeline = create_pipeline(
            numeric_features,
            categorical_features,
            estimator,
        )

        # Cross-validation uses training rows only.
        # Passing groups keeps matching input records together.
        cv_scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            groups=groups_train,
            cv=group_cv,
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
        )

        cv_mae = -cv_scores.mean()

        # Fit on the full training partition and evaluate on holdout.
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = mean_squared_error(y_test, predictions) ** 0.5
        r2 = r2_score(y_test, predictions)

        key = (feature_set, model_name)

        results[key] = {
            "pipeline": pipeline,
            "cv_mae": cv_mae,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "feature_set": feature_set,
            "model_name": model_name,
        }

        print(f"\n{feature_set} | {model_name}")
        print(f"Cross-validation MAE: ₹{cv_mae:,.2f}")
        print(f"Test MAE:             ₹{mae:,.2f}")
        print(f"Test RMSE:            ₹{rmse:,.2f}")
        print(f"Test R²:              {r2:.4f}")


# --------------------------------------------------
# 8. Select the best model using cross-validation
# --------------------------------------------------

best_key = min(
    results,
    key=lambda key: results[key]["cv_mae"],
)

best_result = results[best_key]
best_pipeline = best_result["pipeline"]

model_bundle = {
    "pipeline": best_pipeline,
    "feature_set": best_result["feature_set"],
    "model_name": best_result["model_name"],
    "reference_year": REFERENCE_YEAR,
}

# Save the model bundle in the existing project folder.
with open(MODEL_PATH, "wb") as file:
    pickle.dump(model_bundle, file)


# --------------------------------------------------
# 9. Final report
# --------------------------------------------------

print("\n" + "=" * 60)
print("BEST MODEL — DUPLICATE-SAFE EVALUATION")
print("=" * 60)

print("Feature set:", best_result["feature_set"])
print("Model:", best_result["model_name"])
print(f"Cross-validation MAE: ₹{best_result['cv_mae']:,.2f}")
print(f"Test MAE:             ₹{best_result['mae']:,.2f}")
print(f"Test RMSE:            ₹{best_result['rmse']:,.2f}")
print(f"Test R²:               {best_result['r2']:.4f}")

print("\nBaseline comparison:")
print(f"Baseline Test MAE: ₹{baseline_mae:,.2f}")
print(f"Best Model Test MAE: ₹{best_result['mae']:,.2f}")

if baseline_mae > 0:
    improvement = (
        (baseline_mae - best_result["mae"]) / baseline_mae
    ) * 100

    print(f"MAE improvement vs baseline: {improvement:.2f}%")

print("\nSaved model bundle to:", MODEL_PATH)
print("Your existing app.py can load this model bundle.")

# --------------------------------------------------
# 10. Prediction diagnostics
# --------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np

# Use the selected model on the duplicate-safe test set.
if best_result["feature_set"] == "Engineered features":
    X_diagnostic = X_test_engineered
else:
    X_diagnostic = X_test_original

actual = y_test.to_numpy()
predicted = best_pipeline.predict(X_diagnostic)


diagnostics = X_diagnostic.reset_index(drop=True).copy()

diagnostics["Actual Price"] = actual
diagnostics["Predicted Price"] = predicted

diagnostics["Absolute Error"] = (
    diagnostics["Actual Price"] - diagnostics["Predicted Price"]
).abs()

diagnostics["Prediction Error"] = (
    diagnostics["Actual Price"] - diagnostics["Predicted Price"]
)

diagnostics["Percentage Error"] = (
    diagnostics["Absolute Error"]
    / diagnostics["Actual Price"].clip(lower=1)
) * 100

# Export the held-out test predictions for Power BI.
diagnostics.to_csv(
    BASE_DIR / "car_price_predictions.csv",
    index=False,
)

print("Power BI prediction data saved to:",
      BASE_DIR / "car_price_predictions.csv")


diagnostics["Absolute Error"] = (
    diagnostics["Actual Price"] - diagnostics["Predicted Price"]
).abs()

diagnostics["Percentage Error"] = (
    diagnostics["Absolute Error"]
    / diagnostics["Actual Price"].clip(lower=1)
) * 100

print("\n" + "=" * 60)
print("PREDICTION DIAGNOSTICS")
print("=" * 60)

print(f"Mean actual price: ₹{diagnostics['Actual Price'].mean():,.2f}")
print(f"Mean predicted price: ₹{diagnostics['Predicted Price'].mean():,.2f}")
print(f"Median absolute error: ₹{diagnostics['Absolute Error'].median():,.2f}")
print(
    "Predictions within ₹50,000:",
    f"{(diagnostics['Absolute Error'] <= 50_000).mean() * 100:.2f}%"
)
print(
    "Predictions within ₹1,00,000:",
    f"{(diagnostics['Absolute Error'] <= 100_000).mean() * 100:.2f}%"
)

print("\n10 largest prediction errors:")
print(
    diagnostics.nlargest(10, "Absolute Error")
    .round(2)
    .to_string(index=False)
)

# Save the diagnostics table in memory for inspection in this run.
# Charts are displayed interactively by Matplotlib.

# Chart 1: Actual vs predicted prices.
plt.figure(figsize=(8, 6))
plt.scatter(actual, predicted, alpha=0.6)

plot_min = min(actual.min(), predicted.min())
plot_max = max(actual.max(), predicted.max())

plt.plot(
    [plot_min, plot_max],
    [plot_min, plot_max],
    linestyle="--",
)

plt.xlabel("Actual Selling Price (₹)")
plt.ylabel("Predicted Selling Price (₹)")
plt.title("Actual vs Predicted Car Prices")
plt.tight_layout()
plt.show()

# Chart 2: Distribution of prediction errors.
errors = actual - predicted

plt.figure(figsize=(8, 6))
plt.hist(errors, bins=30)
plt.axvline(0, linestyle="--")
plt.xlabel("Prediction Error: Actual - Predicted (₹)")
plt.ylabel("Number of Cars")
plt.title("Distribution of Prediction Errors")
plt.tight_layout()
plt.show()

# Chart 3: Absolute error vs actual price.
plt.figure(figsize=(8, 6))
plt.scatter(actual, diagnostics["Absolute Error"], alpha=0.6)
plt.xlabel("Actual Selling Price (₹)")
plt.ylabel("Absolute Prediction Error (₹)")
plt.title("Prediction Error vs Actual Car Price")
plt.tight_layout()
plt.show()
