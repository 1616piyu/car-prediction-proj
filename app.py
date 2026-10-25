
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
DATA_PATH = BASE_DIR / "Car_Data.csv"
PREDICTIONS_PATH = BASE_DIR / "car_price_predictions.csv"

st.set_page_config(
    page_title="Car Price Predictor & Analytics",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

REQUIRED_COLUMNS = {
    "name",
    "year",
    "selling_price",
    "km_driven",
    "fuel",
    "seller_type",
    "transmission",
    "owner",
}

CATEGORICAL_COLUMNS = [
    "name",
    "fuel",
    "seller_type",
    "transmission",
    "owner",
]


# --------------------------------------------------
# Utility functions
# --------------------------------------------------

def money(value):
    """Format a finite numeric value as Indian rupees."""
    try:
        value = float(value)
        return f"₹{value:,.0f}" if np.isfinite(value) else "N/A"
    except (TypeError, ValueError):
        return "N/A"


def fail(message):
    """Display a fatal error and stop the Streamlit script."""
    st.error(message)
    st.stop()


@st.cache_resource(show_spinner="Loading trained model...")
def load_model(path, modified_time):
    """
    Load the trusted model bundle.

    modified_time is included in the cache key so that replacing
    model.pkl causes Streamlit to load the new file.
    """
    del modified_time

    with open(path, "rb") as handle:
        return pickle.load(handle)


@st.cache_data(show_spinner=False)
def load_csv(path, modified_time):
    """Load a CSV and refresh the cache when the file changes."""
    del modified_time

    data = pd.read_csv(path)
    data.columns = data.columns.astype(str).str.strip()
    return data


def clean_numeric_columns(data, columns):
    """Convert specified columns to numbers and normalize infinities."""
    for column in columns:
        if column in data.columns:
            data[column] = pd.to_numeric(
                data[column], errors="coerce"
            )
            data[column] = data[column].replace(
                [np.inf, -np.inf], np.nan
            )

    return data


def build_prediction_input(
    name,
    year,
    km_driven,
    fuel,
    seller_type,
    transmission,
    owner,
):
    """
    Build input features matching the training script.

    The feature set and reference year are taken from model.pkl.
    """
    input_data = pd.DataFrame(
        [
            {
                "name": str(name),
                "year": int(year),
                "km_driven": int(km_driven),
                "fuel": str(fuel),
                "seller_type": str(seller_type),
                "transmission": str(transmission),
                "owner": str(owner),
            }
        ]
    )

    car_age = max(reference_year - int(year), 0)

    if feature_set == "Engineered features":
        input_data["car_age"] = car_age
        input_data["km_per_year"] = (
            int(km_driven) / max(car_age, 1)
        )
        input_data["brand"] = (
            input_data["name"]
            .str.strip()
            .str.split()
            .str[0]
        )

    return input_data, car_age


# --------------------------------------------------
# Load and validate model
# --------------------------------------------------

try:
    if not MODEL_PATH.is_file():
        fail("`model.pkl` is missing. Run `python car_pred.py` first.")

    bundle = load_model(
        str(MODEL_PATH),
        MODEL_PATH.stat().st_mtime_ns,
    )

    if not isinstance(bundle, dict):
        raise ValueError("The saved model bundle must be a dictionary.")

    required_bundle_keys = {
        "pipeline",
        "feature_set",
        "model_name",
        "reference_year",
    }

    missing_bundle_keys = required_bundle_keys - set(bundle.keys())

    if missing_bundle_keys:
        raise ValueError(
            "Missing model bundle keys: "
            + ", ".join(sorted(missing_bundle_keys))
        )

    model = bundle["pipeline"]
    feature_set = str(bundle["feature_set"])
    model_name = str(bundle["model_name"])
    reference_year = int(bundle["reference_year"])

    if not hasattr(model, "predict"):
        raise ValueError("The saved pipeline does not support prediction.")

    if feature_set not in {
        "Original features",
        "Engineered features",
    }:
        raise ValueError(
            f"Unsupported feature set: {feature_set}"
        )

    if not 1990 <= reference_year <= 2100:
        raise ValueError("The model reference year is invalid.")

except FileNotFoundError:
    fail("`model.pkl` was not found. Run `python car_pred.py` first.")
except Exception as exc:
    fail(
        f"Could not load the trained model: {exc}. "
        "Check the model bundle and training script."
    )


# --------------------------------------------------
# Load and validate dataset
# --------------------------------------------------

try:
    if not DATA_PATH.is_file():
        fail("`Car_Data.csv` is missing from the project folder.")

    df = load_csv(
        str(DATA_PATH),
        DATA_PATH.stat().st_mtime_ns,
    )

except Exception as exc:
    fail(f"Could not read `Car_Data.csv`: {exc}")


missing_columns = REQUIRED_COLUMNS - set(df.columns)

if missing_columns:
    fail(
        "Missing required dataset columns: "
        + ", ".join(sorted(missing_columns))
    )


df = clean_numeric_columns(
    df,
    ["year", "selling_price", "km_driven"],
)

for column in CATEGORICAL_COLUMNS:
    df[column] = (
        df[column]
        .astype("string")
        .str.strip()
        .replace("", pd.NA)
    )


# Positive selling prices are required for market analysis.
market_df = df.dropna(
    subset=["selling_price"]
).copy()

market_df = market_df[
    market_df["selling_price"] > 0
].copy()

if market_df.empty:
    fail(
        "The dataset contains no valid positive selling prices."
    )


# Use complete records for dropdown choices.
choices_df = df.dropna(
    subset=["year", "km_driven"] + CATEGORICAL_COLUMNS
).copy()

choices_df = choices_df[
    (choices_df["km_driven"] >= 0)
    & (choices_df["year"] >= 1990)
    & (choices_df["year"] <= reference_year)
].copy()


# --------------------------------------------------
# Load held-out prediction diagnostics
# --------------------------------------------------

predictions_df = None

if PREDICTIONS_PATH.is_file():
    try:
        diagnostics = load_csv(
            str(PREDICTIONS_PATH),
            PREDICTIONS_PATH.stat().st_mtime_ns,
        ).copy()

        required_prediction_columns = {
            "Actual Price",
            "Predicted Price",
        }

        if required_prediction_columns.issubset(
            diagnostics.columns
        ):
            diagnostics = clean_numeric_columns(
                diagnostics,
                ["Actual Price", "Predicted Price"],
            )

            diagnostics = diagnostics.dropna(
                subset=["Actual Price", "Predicted Price"]
            ).copy()

            diagnostics = diagnostics[
                (diagnostics["Actual Price"] > 0)
                & (diagnostics["Predicted Price"] > 0)
            ].copy()

            if not diagnostics.empty:
                diagnostics["Prediction Error"] = (
                    diagnostics["Actual Price"]
                    - diagnostics["Predicted Price"]
                )

                diagnostics["Absolute Error"] = (
                    diagnostics["Prediction Error"].abs()
                )

                diagnostics["Percentage Error"] = (
                    100
                    * diagnostics["Absolute Error"]
                    / diagnostics["Actual Price"]
                )

                predictions_df = diagnostics

    except Exception as exc:
        st.warning(
            "Prediction diagnostics could not be loaded. "
            f"Check `car_price_predictions.csv`. Details: {exc}"
        )


# --------------------------------------------------
# Sidebar navigation
# --------------------------------------------------

st.sidebar.title("🚗 Car Analytics")
st.sidebar.caption(
    "Used-car price prediction and model diagnostics"
)

page = st.sidebar.radio(
    "Navigate",
    [
        "Price Predictor",
        "Analytics Dashboard",
        "Model Evaluation",
    ],
)

st.sidebar.divider()
st.sidebar.write(f"**Model:** {model_name}")
st.sidebar.write(f"**Features:** {feature_set}")


# --------------------------------------------------
# PAGE 1: PRICE PREDICTOR
# --------------------------------------------------

if page == "Price Predictor":

    st.title("🚗 Used Car Price Predictor")

    st.write(
        "Estimate a used car's selling price using its listing details."
    )

    st.info(
        f"This model uses historical training data and a "
        f"reference year of {reference_year}. "
        "The estimate is not a live market quote."
    )

    if choices_df.empty:
        st.error(
            "No complete vehicle records are available for the "
            "prediction form. Check your dataset."
        )

    else:
        options = {
            column: sorted(
                choices_df[column]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
            for column in CATEGORICAL_COLUMNS
        }

        if any(not values for values in options.values()):
            st.error(
                "Some vehicle categories are missing. "
                "Check `Car_Data.csv`."
            )

        else:
            with st.form("car_prediction_form"):

                st.subheader("Vehicle Details")

                name = st.selectbox(
                    "Car Model",
                    options["name"],
                )

                left, right = st.columns(2)

                with left:
                    year = st.number_input(
                        "Manufacture Year",
                        min_value=1990,
                        max_value=reference_year,
                        value=reference_year,
                        step=1,
                    )

                    km_driven = st.number_input(
                        "Kilometers Driven",
                        min_value=0,
                        max_value=2_000_000,
                        value=30_000,
                        step=1_000,
                    )

                    fuel = st.selectbox(
                        "Fuel Type",
                        options["fuel"],
                    )

                with right:
                    seller_type = st.selectbox(
                        "Seller Type",
                        options["seller_type"],
                    )

                    transmission = st.selectbox(
                        "Transmission",
                        options["transmission"],
                    )

                    owner = st.selectbox(
                        "Owner Type",
                        options["owner"],
                    )

                submitted = st.form_submit_button(
                    "Predict Selling Price",
                    use_container_width=True,
                )

            if submitted:
                try:
                    input_data, car_age = build_prediction_input(
                        name=name,
                        year=year,
                        km_driven=km_driven,
                        fuel=fuel,
                        seller_type=seller_type,
                        transmission=transmission,
                        owner=owner,
                    )

                    prediction = float(
                        model.predict(input_data)[0]
                    )

                    if (
                        not np.isfinite(prediction)
                        or prediction <= 0
                    ):
                        st.warning(
                            "The model returned an invalid price. "
                            "Try another combination."
                        )

                    else:
                        st.success("Prediction complete")

                        st.metric(
                            "Estimated Selling Price",
                            money(prediction),
                        )

                        st.caption(
                            "This estimate may not reflect vehicle "
                            "condition, location, service history, "
                            "accident history, or current market changes."
                        )

                        st.subheader("Selected Car Summary")

                        summary = pd.DataFrame(
                            {
                                "Detail": [
                                    "Car Model",
                                    "Manufacture Year",
                                    "Age at Model Reference Year",
                                    "Kilometers Driven",
                                    "Fuel",
                                    "Seller Type",
                                    "Transmission",
                                    "Ownership",
                                ],
                                "Value": [
                                    name,
                                    str(int(year)),
                                    f"{car_age} years",
                                    f"{int(km_driven):,} km",
                                    fuel,
                                    seller_type,
                                    transmission,
                                    owner,
                                ],
                            }
                        )

                        st.dataframe(
                            summary,
                            hide_index=True,
                            use_container_width=True,
                        )

                except Exception as exc:
                    st.error(
                        f"Prediction failed: {exc}. "
                        "Check that the model bundle and feature "
                        "engineering match `car_pred.py`."
                    )


# --------------------------------------------------
# PAGE 2: ANALYTICS DASHBOARD
# --------------------------------------------------

elif page == "Analytics Dashboard":

    st.title("📊 Used Car Analytics Dashboard")

    st.write(
        "Explore historical prices and compare listings "
        "across vehicle categories."
    )

    st.sidebar.subheader("Market Filters")

    filters = {}

    for column, label in [
        ("fuel", "Fuel Type"),
        ("transmission", "Transmission"),
        ("seller_type", "Seller Type"),
    ]:
        values = sorted(
            market_df[column]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        filters[column] = st.sidebar.multiselect(
            label,
            options=values,
            default=values,
        )

    filtered_df = market_df.copy()

    for column, selected in filters.items():
        filtered_df = filtered_df[
            filtered_df[column].astype(str).isin(selected)
        ]

    if filtered_df.empty:

        st.warning(
            "No cars match the selected filters. "
            "Select more categories in the sidebar."
        )

    else:
        a, b, c, d = st.columns(4)

        a.metric(
            "Listings",
            f"{len(filtered_df):,}",
        )

        b.metric(
            "Median Price",
            money(filtered_df["selling_price"].median()),
        )

        c.metric(
            "Average Price",
            money(filtered_df["selling_price"].mean()),
        )

        median_km = filtered_df["km_driven"].median()

        d.metric(
            "Median Kilometers",
            (
                f"{median_km:,.0f} km"
                if pd.notna(median_km)
                else "N/A"
            ),
        )

        st.divider()

        st.subheader("Median Selling Price by Fuel Type")

        fuel_data = (
            filtered_df.dropna(
                subset=["fuel", "selling_price"]
            )
            .groupby("fuel")["selling_price"]
            .median()
            .sort_values(ascending=False)
        )

        if not fuel_data.empty:
            st.bar_chart(fuel_data)
        else:
            st.info("No fuel-type data is available for this filter.")

        left, right = st.columns(2)

        with left:
            st.subheader("Price by Transmission")

            transmission_data = (
                filtered_df.dropna(
                    subset=["transmission", "selling_price"]
                )
                .groupby("transmission")["selling_price"]
                .median()
                .sort_values(ascending=False)
            )

            if not transmission_data.empty:
                st.bar_chart(transmission_data)
            else:
                st.info("No transmission data is available.")

        with right:
            st.subheader("Price by Seller Type")

            seller_data = (
                filtered_df.dropna(
                    subset=["seller_type", "selling_price"]
                )
                .groupby("seller_type")["selling_price"]
                .median()
                .sort_values(ascending=False)
            )

            if not seller_data.empty:
                st.bar_chart(seller_data)
            else:
                st.info("No seller-type data is available.")

        st.subheader("Median Price by Manufacture Year")

        year_data = (
            filtered_df.dropna(
                subset=["year", "selling_price"]
            )
            .groupby("year")["selling_price"]
            .median()
            .sort_index()
        )

        if not year_data.empty:
            st.line_chart(year_data)
        else:
            st.info("No manufacture-year data is available.")

        st.subheader("Kilometers Driven vs Selling Price")

        mileage = filtered_df.dropna(
            subset=["km_driven", "selling_price"]
        )

        if not mileage.empty:
            st.scatter_chart(
                mileage,
                x="km_driven",
                y="selling_price",
            )
        else:
            st.info("No mileage data is available.")

        st.subheader("Top 10 Car Models by Median Price")

        model_summary = (
            filtered_df.dropna(
                subset=["name", "selling_price"]
            )
            .groupby("name")
            .agg(
                median_price=("selling_price", "median"),
                listings=("selling_price", "size"),
            )
        )

        # Require repeated listings to reduce the influence of rare cars.
        top_models = (
            model_summary[model_summary["listings"] >= 3]
            .nlargest(10, "median_price")
            .sort_values("median_price")
        )

        if top_models.empty:
            st.info(
                "There are not enough repeated listings to "
                "compare models reliably."
            )
        else:
            st.bar_chart(top_models["median_price"])
            st.caption(
                "Only models with at least three listings "
                "in the filtered dataset are shown."
            )

        st.caption(
            "These charts describe the historical dataset, not "
            "today's used-car market. Repeated listings may be present."
        )


# --------------------------------------------------
# PAGE 3: MODEL EVALUATION
# --------------------------------------------------

elif page == "Model Evaluation":

    st.title("🧪 Model Evaluation Dashboard")

    st.write(
        "Review performance on held-out test records exported "
        "by `car_pred.py`."
    )

    if predictions_df is None:

        st.warning(
            "Valid prediction diagnostics were not found. "
            "Run `python car_pred.py` to generate "
            "`car_price_predictions.csv`, then refresh the app."
        )

    else:
        actual = predictions_df["Actual Price"]
        predicted = predictions_df["Predicted Price"]
        errors = predictions_df["Absolute Error"]

        mae = float(errors.mean())

        rmse = float(
            np.sqrt(
                np.mean((actual - predicted) ** 2)
            )
        )

        total_variation = float(
            ((actual - actual.mean()) ** 2).sum()
        )

        r2 = (
            1
            - float(((actual - predicted) ** 2).sum())
            / total_variation
            if total_variation > 0
            else float("nan")
        )

        a, b, c = st.columns(3)

        a.metric("Test MAE", money(mae))
        b.metric("Test RMSE", money(rmse))
        c.metric(
            "Test R²",
            f"{r2:.4f}" if np.isfinite(r2) else "N/A",
        )

        a, b, c = st.columns(3)

        a.metric(
            "Test Records",
            f"{len(predictions_df):,}",
        )

        b.metric(
            "Within ₹50,000",
            f"{(errors <= 50_000).mean() * 100:.2f}%",
        )

        c.metric(
            "Within ₹1 Lakh",
            f"{(errors <= 100_000).mean() * 100:.2f}%",
        )

        st.caption(
            "MAE is the average absolute prediction error. "
            "RMSE penalizes large errors more heavily. "
            "R² measures fit relative to predicting the mean."
        )

        st.divider()

        # Actual versus predicted plot.
        st.subheader("Actual vs Predicted Prices")

        fig, ax = plt.subplots(figsize=(9, 5))

        ax.scatter(
            actual,
            predicted,
            alpha=0.5,
            s=18,
            label="Test predictions",
        )

        low = min(
            float(actual.min()),
            float(predicted.min()),
        )

        high = max(
            float(actual.max()),
            float(predicted.max()),
        )

        ax.plot(
            [low, high],
            [low, high],
            "--",
            color="gray",
            label="Perfect prediction",
        )

        ax.set_xlabel("Actual Selling Price (₹)")
        ax.set_ylabel("Predicted Selling Price (₹)")
        ax.legend()
        ax.grid(alpha=0.2)
        fig.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            "Above the dashed line means overprediction; "
            "below it means underprediction."
        )

        # Prediction-error histogram.
        st.subheader("Distribution of Prediction Errors")

        fig, ax = plt.subplots(figsize=(9, 4))

        ax.hist(
            predictions_df["Prediction Error"],
            bins=25,
            edgecolor="white",
        )

        ax.axvline(
            0,
            linestyle="--",
            color="gray",
            label="Zero error",
        )

        ax.set_xlabel("Actual − Predicted (₹)")
        ax.set_ylabel("Number of Cars")
        ax.legend()
        fig.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            "Positive error means underprediction; "
            "negative error means overprediction."
        )

        # Largest individual prediction errors.
        st.subheader("10 Largest Prediction Errors")

        top_errors = predictions_df.nlargest(
            10,
            "Absolute Error",
        )

        display_columns = [
            column
            for column in [
                "name",
                "fuel",
                "year",
                "km_driven",
                "Actual Price",
                "Predicted Price",
                "Absolute Error",
                "Percentage Error",
            ]
            if column in top_errors.columns
        ]

        st.dataframe(
            top_errors[display_columns].round(2),
            hide_index=True,
            use_container_width=True,
        )

        # Error grouped by actual price.
        st.subheader("Error by Actual Price Range")

        by_price = predictions_df.copy()

        by_price["Price Band"] = pd.cut(
            by_price["Actual Price"],
            bins=[
                0,
                200_000,
                500_000,
                1_000_000,
                2_000_000,
                np.inf,
            ],
            labels=[
                "Up to ₹2 lakh",
                "₹2–5 lakh",
                "₹5–10 lakh",
                "₹10–20 lakh",
                "Above ₹20 lakh",
            ],
            include_lowest=True,
        )

        summary = (
            by_price.groupby(
                "Price Band",
                observed=False,
            )
            .agg(
                Cars=("Absolute Error", "size"),
                MAE=("Absolute Error", "mean"),
            )
        )

        nonempty = summary[summary["Cars"] > 0]

        if not nonempty.empty:
            st.bar_chart(nonempty["MAE"])

            display_summary = nonempty.copy()
            display_summary["Cars"] = (
                display_summary["Cars"].map("{:,.0f}".format)
            )
            display_summary["MAE"] = (
                display_summary["MAE"].map(money)
            )

            st.dataframe(
                display_summary,
                use_container_width=True,
            )
        else:
            st.info("No price-band evaluation data is available.")

        st.caption(
            "Test-set performance is not a guarantee of accuracy "
            "on new listings. Unusual or expensive cars may have "
            "larger prediction errors."
        )


# --------------------------------------------------
# MODEL INFORMATION
# --------------------------------------------------

with st.sidebar.expander("About the Model"):

    st.write(f"**Algorithm:** {model_name}")
    st.write(f"**Feature set:** {feature_set}")
    st.write(f"**Training reference year:** {reference_year}")

    st.write(
        "The model uses historical car listings. "
        "Only load model.pkl from a trusted source."
    )


    