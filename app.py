from flask import Flask, render_template, request, send_file
import pandas as pd
import joblib
import json
import os

app = Flask(__name__)

# ==========================
# Load model and metadata
# ==========================
MODEL_PATH = "random_forest_biomass_model.pkl"
FEATURES_PATH = "model_features.json"

print("Loading trained model...")
model = joblib.load(MODEL_PATH)
print("Model loaded successfully!")

with open(FEATURES_PATH, "r") as f:
    model_features = json.load(f)
print(f"Loaded {len(model_features)} model features.")

# ==========================
# Column mapping for readable input
# ==========================
COLUMN_MAPPING = {
    "Tree Cover Density (2000)": "umd_tree_cover_density_2000__threshold",
    "Tree Cover Extent (2000)": "umd_tree_cover_extent_2000__ha",
    "Average Carbon Stock (MgC/ha)": "avg_gfw_aboveground_carbon_stocks_2000__Mg_C_ha-1",
    "Threshold": "threshold",
    "Area (ha)": "area_ha",
    "Extent 2000 (ha)": "extent_2000_ha",
    "Extent 2010 (ha)": "extent_2010_ha",
    "Gain 2000–2020 (ha)": "gain_2000-2020_ha",
    "Tree Loss 2001": "tc_loss_ha_2001",
    "Tree Loss 2002": "tc_loss_ha_2002",
    "Tree Loss 2003": "tc_loss_ha_2003",
    "Tree Loss 2004": "tc_loss_ha_2004",
    "Tree Loss 2005": "tc_loss_ha_2005",
    "Tree Loss 2006": "tc_loss_ha_2006",
    "Tree Loss 2007": "tc_loss_ha_2007",
    "Tree Loss 2008": "tc_loss_ha_2008",
    "Tree Loss 2009": "tc_loss_ha_2009",
    "Tree Loss 2010": "tc_loss_ha_2010",
    "Tree Loss 2011": "tc_loss_ha_2011",
    "Tree Loss 2012": "tc_loss_ha_2012",
    "Tree Loss 2013": "tc_loss_ha_2013",
    "Tree Loss 2014": "tc_loss_ha_2014",
    "Tree Loss 2015": "tc_loss_ha_2015",
    "Tree Loss 2016": "tc_loss_ha_2016",
    "Tree Loss 2017": "tc_loss_ha_2017",
    "Tree Loss 2018": "tc_loss_ha_2018",
    "Tree Loss 2019": "tc_loss_ha_2019",
    "Tree Loss 2020": "tc_loss_ha_2020",
    "Tree Loss 2021": "tc_loss_ha_2021",
    "Tree Loss 2022": "tc_loss_ha_2022"
}

# Reverse map for display back
REVERSE_MAPPING = {v: k for k, v in COLUMN_MAPPING.items()}

# ==========================
# Home Route
# ==========================
@app.route('/')
def home():
    return render_template('index.html')

# ==========================
# Prediction Route
# ==========================
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    # Load uploaded CSV
    input_df = pd.read_csv(file)
    print(f"Input data loaded successfully! Shape: {input_df.shape}")

    # Rename columns using mapping
    renamed_df = input_df.rename(columns=COLUMN_MAPPING)

    # Check for missing model features
    missing_cols = set(model_features) - set(renamed_df.columns)
    for col in missing_cols:
        renamed_df[col] = 0  # Fill missing features if needed

    # Keep only model features
    X = renamed_df[model_features]

    # Predict
    print("Predicting Carbon Stock...")
    predictions = model.predict(X)
    print("Predictions completed successfully!")

    # Prepare result DataFrame
    result_df = input_df.copy()
    result_df["Predicted_Carbon_Stock_MgC"] = predictions

    # Save output CSV
    output_file = "predicted_carbon_stock.csv"
    result_df.to_csv(output_file, index=False)

    # Display first few results
    return render_template(
    'result.html',
    table=result_df.head(10).to_html(classes='data', index=False),
    titles=result_df.columns.values,
    download_link=output_file
)


# ==========================
# Download Route
# ==========================
@app.route('/download')
def download_file():
    output_file = "predicted_carbon_stock.csv"
    if os.path.exists(output_file):
        return send_file(output_file, as_attachment=True)
    else:
        return "No prediction file found.", 404

# ==========================
# Run Flask App
# ==========================
if __name__ == '__main__':
    app.run(debug=True)
