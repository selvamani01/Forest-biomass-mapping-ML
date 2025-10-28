import os
import pandas as pd

folder = r"D:\energy\deforestation"

# File paths
country_carbon = os.path.join(folder, "Country carbon data.csv")
country_tree = os.path.join(folder, "Country tree cover loss.csv")
sub_carbon = os.path.join(folder, "Subnational 1 carbon data.csv")
sub_tree = os.path.join(folder, "Subnational 1 tree cover loss.csv")

# Load CSV files
df_country_carbon = pd.read_csv(country_carbon)
df_country_tree = pd.read_csv(country_tree)
df_sub_carbon = pd.read_csv(sub_carbon)
df_sub_tree = pd.read_csv(sub_tree)

print(" Files loaded successfully!")

# Merge country-level data (based on 'country' only)
df_country = pd.merge(df_country_carbon, df_country_tree, on="country", how="outer")

# Merge subnational-level data (based on 'subnational1' only)
df_subnational = pd.merge(df_sub_carbon, df_sub_tree, on="subnational1", how="outer")

# Add a level column
df_country["Level"] = "Country"
df_subnational["Level"] = "Subnational"

# Combine both
merged_df = pd.concat([df_country, df_subnational], ignore_index=True)

# Save merged data
output_path = os.path.join(folder, "merged_forest_data.csv")
merged_df.to_csv(output_path, index=False)

print(" All datasets merged successfully!")
print(f" Merged file saved as: {output_path}")
print("\n Merged Data Preview:")
print(merged_df.head())
