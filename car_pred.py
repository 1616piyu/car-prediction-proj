import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv("Car_Data.csv")  # adjust path if needed

# Clean column names (in case there are extra spaces)
df.columns = df.columns.str.strip()

# Show column names for reference
print("Columns:", df.columns)

# Drop any missing values
df.dropna(inplace=True)

# Encode categorical features
label_encoders = {}
for column in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Features and Target
X = df.drop("selling_price", axis=1)
y = df["selling_price"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Evaluation:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.2f}")


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor

# Try Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
print("Linear Regression R2:", lr.score(X_test, y_test))

#Data Visualization (EDA)
import seaborn as sns
import matplotlib.pyplot as plt

# Plot price distribution
sns.histplot(df['selling_price'], bins=50)
plt.title('Selling Price Distribution')
plt.show()

# Correlation heatmap
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation')
plt.show()

import pickle

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save encoders
with open("encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

