import pandas as pd
import joblib
import numpy as np
import os

# === STEP 1: Load model + training data + predictions ===
train_data_path = r"D:\energy\deforestation\preprocessed_forest_data.csv"
predictions_path = r"D:\energy\deforestation\carbon_predictions.csv"

df_train = pd.read_csv(train_data_path)
df_pred = pd.read_csv(predictions_path)

target_col = "gfw_aboveground_carbon_stocks_2000__Mg_C"

if target_col not in df_train.columns:
    raise ValueError(f"Target column '{target_col}' not found in training dataset!")

# === STEP 2: Basic statistics of actual and predicted ===
actual_min, actual_max = df_train[target_col].min(), df_train[target_col].max()
pred_min, pred_max = df_pred["Predicted_Carbon_Stock_MgC"].min(), df_pred["Predicted_Carbon_Stock_MgC"].max()

print("\n=== Actual Carbon Stock Range (Training Data) ===")
print(f"Min: {actual_min:,.2f}")
print(f"Max: {actual_max:,.2f}")

print("\n=== Predicted Carbon Stock Range (New Data) ===")
print(f"Min: {pred_min:,.2f}")
print(f"Max: {pred_max:,.2f}")

# === STEP 3: Check realism ===
if pred_max > actual_max * 1.5:
    print("\n Prediction range is unusually high compared to training data.")
elif pred_min < actual_min * 0.5:
    print("\n Prediction range is unusually low compared to training data.")
else:
    print("\n Predictions fall within a realistic range of training data.")

# === STEP 4: Optional: Compare means ===
actual_mean = df_train[target_col].mean()
pred_mean = df_pred["Predicted_Carbon_Stock_MgC"].mean()
print(f"\nTraining Mean: {actual_mean:,.2f}")
print(f"Prediction Mean: {pred_mean:,.2f}")

if abs(pred_mean - actual_mean) / actual_mean < 0.3:
    print("Mean prediction is consistent with training data.")
else:
    print(" Mean prediction differs significantly — check input scaling.")
