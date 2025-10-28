import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

# =============================
# STEP 1 — Load merged dataset
# =============================
folder = r"D:\energy\deforestation"
data_path = os.path.join(folder, "merged_forest_data.csv")

print(" Loading dataset...")
df = pd.read_csv(data_path)
print(" Data loaded successfully!")
print("Shape:", df.shape)
print("Columns:\n", df.columns.tolist()[:10], "...")  # show first 10 column names

# =============================
# STEP 2 — Handle missing values
# =============================
print("\n Checking missing values...")
missing_count = df.isnull().sum().sum()
print(f"Total missing values before cleaning: {missing_count}")

# Fill numeric missing values with column mean
df = df.fillna(df.mean(numeric_only=True))
print("Missing numeric values replaced with mean.")

# =============================
# STEP 3 — Select input and target features
# =============================

# 🎯 Target feature — represents above-ground biomass (carbon stock)
target_feature = 'gfw_aboveground_carbon_stocks_2000__Mg_C'

# 📥 Input features (forest structure, extent, and change)
input_features = [
    'umd_tree_cover_density_2000__threshold',
    'umd_tree_cover_extent_2000__ha',
    'extent_2010_ha',
    'gain_2000-2020_ha',
    'tc_loss_ha_2001', 'tc_loss_ha_2005', 'tc_loss_ha_2010',
    'tc_loss_ha_2015', 'tc_loss_ha_2020'
]

# Filter only columns that exist (avoid KeyError)
available_features = [col for col in input_features if col in df.columns]
if target_feature not in df.columns:
    raise KeyError(f"Target feature '{target_feature}' not found in dataset!")

df = df[available_features + [target_feature]]
print("\n Selected features:")
print("Inputs:", available_features)
print("Target:", target_feature)

# =============================
# STEP 4 — Normalize input features
# =============================
print("\n Normalizing numeric input features...")
scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[available_features] = scaler.fit_transform(df_scaled[available_features])
print("Features normalized successfully.")

# =============================
# STEP 5 — Save preprocessed data
# =============================
processed_path = os.path.join(folder, "preprocessed_forest_data.csv")
df_scaled.to_csv(processed_path, index=False)
print(f"\n Preprocessed dataset saved as:\n{processed_path}")

# =============================
# STEP 6 — Final summary
# =============================
print("\n Final Data Overview:")
print(df_scaled.head())
print("\n Preprocessing complete! Ready for model training.")
