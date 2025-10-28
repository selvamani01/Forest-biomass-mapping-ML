import pandas as pd
import joblib
import json

# Step 1: Load the trained model
print("Loading trained model...")
model = joblib.load("random_forest_biomass_model.pkl")
print("Model loaded successfully!")

# Step 2: Load feature names
with open("model_features.json", "r") as f:
    model_features = json.load(f)
print(f"Loaded {len(model_features)} model features.")

# Step 3: Load new input data (for prediction)
input_file = "new_forest_data.csv"
print(f"\nLoading input file: {input_file}")
new_data = pd.read_csv(input_file)
print("Input data loaded successfully! Shape:", new_data.shape)

# Step 4: Match columns to training features
missing_features = set(model_features) - set(new_data.columns)
extra_features = set(new_data.columns) - set(model_features)

if missing_features:
    print(f"Missing features in input data: {missing_features}")
if extra_features:
    print(f"Extra features in input data (will be ignored): {extra_features}")

# Keep only required model features
new_data = new_data.reindex(columns=model_features, fill_value=0)

# Step 5: Predict carbon stock
print("\nPredicting Carbon Stock...")
predictions = model.predict(new_data)

# Step 6: Show predictions in console
new_data["Predicted_Carbon_Stock_MgC"] = predictions

print("\nPredictions completed successfully!")
print("\nSample Results (first 10):")
print(new_data[["Predicted_Carbon_Stock_MgC"]].head(10))
