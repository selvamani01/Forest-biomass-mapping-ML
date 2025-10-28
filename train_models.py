# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import json

# Step 1: Load cleaned dataset
print(" Loading cleaned dataset...")
df = pd.read_csv("cleaned_forest_data.csv")
print(" Dataset loaded successfully! Shape:", df.shape)

# Step 2: Define target column and features
target_col = 'gfw_aboveground_carbon_stocks_2000__Mg_C'
X = df.drop(columns=[target_col])
y = df[target_col]

# Step 3: Split into training and testing sets
print("\n Splitting data into train & test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f" Train size: {X_train.shape}, Test size: {X_test.shape}")

# Step 4: Initialize and train Random Forest model
print("\n Training Random Forest model...")
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=25,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)
print(" Model training complete!")

# Step 5: Evaluate model performance
print("\n Evaluating model performance...")
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f" Mean Squared Error (MSE): {mse:.2f}")
print(f" R² Score: {r2:.4f}")

# Step 6: Save the trained model
model_filename = "random_forest_biomass_model.pkl"
joblib.dump(model, model_filename)
print(f"\n Model saved as '{model_filename}'")

# Step 7: Save feature names
features_filename = "model_features.json"
with open(features_filename, "w") as f:
    json.dump(list(X.columns), f)
print(f" Feature names saved as '{features_filename}'")

# Step 8: Save evaluation metrics for reference
metrics = {"MSE": mse, "R2": r2}
metrics_filename = "model_metrics.json"
with open(metrics_filename, "w") as f:
    json.dump(metrics, f)
print(f" Metrics saved as '{metrics_filename}'")

# Step 9: Test predictions on few samples
print("\n Testing predictions on few random samples:")
sample_inputs = X_test.sample(5, random_state=42)
sample_predictions = model.predict(sample_inputs)

for i, pred in enumerate(sample_predictions):
    print(f"Sample {i+1}: Predicted Carbon Stock = {pred:.2f}")

print("\n Training pipeline completed successfully!")

import numpy as np

# Check correlation of all features with the target
corr_matrix = df.corr()[target_col].sort_values(ascending=False)
print("\nTop 10 correlations with target:\n")
print(corr_matrix.head(10))

