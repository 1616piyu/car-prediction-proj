
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent
PLOT_DIR = BASE_DIR / "eda_plots"
PLOT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(BASE_DIR / "Car_Data.csv")
df.columns = df.columns.str.strip()

sns.set_theme(style="whitegrid")

# 1. Selling price distribution
plt.figure(figsize=(9, 5))
sns.histplot(data=df, x="selling_price", bins=50)
plt.title("Distribution of Used Car Selling Prices")
plt.xlabel("Selling Price (₹)")
plt.ylabel("Number of Cars")
plt.tight_layout()
plt.savefig(PLOT_DIR / "price_distribution.png", dpi=150)
plt.close()

# 2. Price distribution after log transformation for visualization
plt.figure(figsize=(9, 5))
sns.histplot(x=df["selling_price"].clip(lower=1).map(lambda x: __import__("math").log10(x)), bins=40)
plt.title("Selling Price Distribution on a Log10 Scale")
plt.xlabel("Log10(Selling Price)")
plt.ylabel("Number of Cars")
plt.tight_layout()
plt.savefig(PLOT_DIR / "price_log_distribution.png", dpi=150)
plt.close()

# 3. Selling price versus kilometres driven
plt.figure(figsize=(9, 5))
sns.scatterplot(
    data=df,
    x="km_driven",
    y="selling_price",
    alpha=0.5,
    s=25,
)
plt.title("Selling Price vs Kilometres Driven")
plt.xlabel("Kilometres Driven")
plt.ylabel("Selling Price (₹)")
plt.tight_layout()
plt.savefig(PLOT_DIR / "price_vs_km.png", dpi=150)
plt.close()

# 4. Selling price by fuel type
plt.figure(figsize=(9, 5))
sns.boxplot(data=df, x="fuel", y="selling_price")
plt.title("Selling Price by Fuel Type")
plt.xlabel("Fuel Type")
plt.ylabel("Selling Price (₹)")
plt.tight_layout()
plt.savefig(PLOT_DIR / "price_by_fuel.png", dpi=150)
plt.close()

print("EDA complete.")
print("Plots saved in:", PLOT_DIR)
print("\nMedian price by fuel type:")
print(df.groupby("fuel")["selling_price"].median().sort_values(ascending=False))

print("\nMedian price by transmission:")
print(df.groupby("transmission")["selling_price"].median().sort_values(ascending=False))
