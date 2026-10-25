
# Used Car Price Prediction & Analytics

An end-to-end machine-learning project that estimates used-car selling prices and provides an interactive dashboard for exploring historical listings and evaluating model performance.

## Features

- **Price Predictor:** Estimate a car's selling price using its model, manufacture year, kilometers driven, fuel type, seller type, transmission, and ownership.
- **Analytics Dashboard:** Explore historical prices by fuel type, transmission, seller type, manufacture year, and mileage.
- **Model Evaluation:** Review MAE, RMSE, R², prediction-error distribution, largest errors, and errors across price ranges.
- **Feature Engineering:** Supports the feature set recorded in the saved model bundle, including car age, kilometers per year, and brand for the engineered model.
- **Data Validation:** Handles common invalid-input, missing-file, and empty-data cases.

## Tech Stack

- Python
- pandas and NumPy
- scikit-learn
- Matplotlib
- Streamlit

## Project Structure

```text
car-prediction-proj/
├── app.py
├── car_pred.py
├── Car_Data.csv
├── model.pkl
├── car_price_predictions.csv
├── requirements.txt
└── README.md
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/1616piyu/car-prediction-proj.git
cd car-prediction-proj
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Generate the model artifacts if needed

The app expects `model.pkl` and `car_price_predictions.csv`. If they are missing, run:

```bash
python car_pred.py
```

This executes the project's training and evaluation workflow using the available dataset.

### 5. Launch the app

```bash
streamlit run app.py
```

Open the local URL printed in the terminal, usually `http://localhost:8501`.

## Dataset

The app expects `Car_Data.csv` with these columns:

| Column | Description |
|---|---|
| `name` | Car model/name |
| `year` | Manufacture year |
| `selling_price` | Historical selling price in INR |
| `km_driven` | Kilometers driven |
| `fuel` | Fuel category |
| `seller_type` | Seller category |
| `transmission` | Transmission category |
| `owner` | Ownership category |

The data represents historical listings, not live market prices. The model may not account for location, vehicle condition, service history, accident history, or current market changes.

## Evaluation Metrics

- **MAE:** Average absolute difference between actual and predicted prices, in INR.
- **RMSE:** Error metric that penalizes larger prediction errors more heavily.
- **R²:** Measures performance relative to predicting the mean; it can be negative.
- **Within-threshold percentage:** Share of held-out predictions within selected INR error thresholds.

Use the results produced by the current training script when reporting model performance. Historical test results are not a guarantee of accuracy on future listings.

## Important Notes

- Only load `model.pkl` from a trusted source. Python pickle files can execute code when loaded.
- Keep the model bundle, feature engineering, dataset columns, and app consistent.
- Do not commit virtual environments, cache folders, credentials, or unrelated files.
- Add screenshots of the running application to make the project easier for recruiters to review.

## Future Improvements

- Add unit tests for preprocessing and prediction inputs.
- Compare performance across car-age and price segments.
- Pin dependency versions after validating the environment.
- Deploy the app and add screenshots.

## Author

**Piyush Pratap Singh**  
GitHub: https://github.com/1616piyu
