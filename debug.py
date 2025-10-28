import pandas as pd

# Load your dataset
df = pd.read_csv("deforestation/merged_forest_data.csv")

# Step 1: Show dataset info
print("\n--- Dataset Info ---")
print(df.info())

# Step 2: Check for missing values
print("\n--- Missing Values per Column ---")
print(df.isnull().sum().sort_values(ascending=False).head(20))

# Step 3: Check variation (summary statistics)
print("\n--- Summary Statistics ---")
print(df.describe().T)

# Step 4: Define your target column
target_col = 'gfw_aboveground_carbon_stocks_2000__Mg_C'

# Step 5: Check unique target values
if target_col in df.columns:
    unique_count = df[target_col].nunique()
    print(f"\nUnique values in target '{target_col}':", unique_count)
    print("\nTarget value counts (Top 10):\n", df[target_col].value_counts().head(10))
else:
    print(f"\n Target column '{target_col}' not found in dataset!")

# Step 6: Check correlation with the target (numeric columns only)
print("\n--- Correlation with Target ---")
numeric_df = df.select_dtypes(include=['number'])
if target_col in numeric_df.columns:
    corr = numeric_df.corr()[target_col].sort_values(ascending=False)
    print(corr)
else:
    print(f"\n Cannot compute correlation: target '{target_col}' is not numeric or missing.")

# Step 7: Identify constant or near-constant columns (no variation)
print("\n--- Columns with Low Variation ---")
low_var = df.nunique().sort_values()
print(low_var.head(20))
