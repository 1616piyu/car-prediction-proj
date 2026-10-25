
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

BASE_DIR = Path(__file__).resolve().parent
df = pd.read_csv(BASE_DIR / "Car_Data.csv")
df.columns = df.columns.str.strip()

# Identify exact duplicate rows using every available column.
groups = pd.util.hash_pandas_object(df, index=False)

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42,
)

train_idx, test_idx = next(
    splitter.split(df, groups=groups)
)

train_df = df.iloc[train_idx]
test_df = df.iloc[test_idx]

train_groups = set(groups.iloc[train_idx])
test_groups = set(groups.iloc[test_idx])

print("Original dataset rows:", len(df))
print("Training rows:", len(train_df))
print("Test rows:", len(test_df))
print("Training unique groups:", len(train_groups))
print("Test unique groups:", len(test_groups))
print("Groups shared between train and test:", len(train_groups & test_groups))

print("\nTraining target median:", train_df["selling_price"].median())
print("Test target median:", test_df["selling_price"].median())
