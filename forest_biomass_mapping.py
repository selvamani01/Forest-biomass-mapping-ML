import pandas as pd
import numpy as np

# -----------------------------
# Step 1: Load the dataset
# -----------------------------
file_path = "deforestation/merged_forest_data.csv"
df = pd.read_csv(file_path, low_memory=False)
print(" Dataset loaded successfully!")
print("Shape:", df.shape)

# -----------------------------
# Step 2: Quick data info
# -----------------------------
print("\n--- Dataset Info ---")
print(df.info())
print("\nMissing values:\n", df.isnull().sum().sort_values(ascending=False).head(20))

# -----------------------------
# Step 3: Decide potential target columns
# -----------------------------
target_candidates = [
    "gfw_aboveground_carbon_stocks_2000__Mg_C",
    "avg_gfw_aboveground_carbon_stocks_2000__Mg_C_ha-1"
]

for col in target_candidates:
    if col in df.columns:
        print(f"\n Checking target column: {col}")
        print("Missing values:", df[col].isna().sum())
        print("Unique values:", df[col].nunique())
        print(df[col].describe())

# Choose your preferred target column
target_col = "gfw_aboveground_carbon_stocks_2000__Mg_C"
print(f"\n Using target column: {target_col}")

# -----------------------------
# Step 4: Drop columns not useful for ML
# -----------------------------
# Drop country names, region text columns, etc.
drop_cols = ["country", "country_x", "country_y", "subnational1", "Level"]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# -----------------------------
# Step 5: Handle missing values
# -----------------------------
# Drop columns with more than 50% missing
missing_threshold = 0.5
df = df[df.columns[df.isnull().mean() < missing_threshold]]

# Fill remaining missing numeric values with median
df = df.fillna(df.median(numeric_only=True))

print("\n Missing values handled.")
print("New shape:", df.shape)

# -----------------------------
# Step 6: Check for low variation columns
# -----------------------------
low_var_cols = [col for col in df.columns if df[col].nunique() <= 1]
if low_var_cols:
    print("\n Dropping low variation columns:", low_var_cols)
    df = df.drop(columns=low_var_cols)

# -----------------------------
# Step 7: Feature-Target Split
# -----------------------------
X = df.drop(columns=[target_col])
y = df[target_col]

print("\n Data prepared successfully.")
print("Features shape:", X.shape)
print("Target shape:", y.shape)

# -----------------------------
# Step 8: Save cleaned data (optional)
# -----------------------------
df.to_csv("cleaned_forest_data.csv", index=False)
print("\n Cleaned dataset saved as 'cleaned_forest_data.csv'")
