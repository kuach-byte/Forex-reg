import pandas as pd
import matplotlib.pyplot as plt
import os

# Load your final labeled data
df = pd.read_csv("data/final_vol.csv")

# Get all one-hot encoded pair columns
pair_columns = [col for col in df.columns if col.startswith('pair_') and col != 'pair_name']

# Compute NaN percentages correctly per active pair
nan_percentages = {}

for pair_col in pair_columns:
    pair_df = df[df[pair_col] == 1]  # Only rows for this pair
    if len(pair_df) == 0:
        nan_percentages[pair_col] = float('nan')
    else:
        nan_percentage = pair_df['volatility_label'].isna().mean() * 100
        nan_percentages[pair_col] = round(nan_percentage, 2)

# Convert to Series and print
nan_series = pd.Series(nan_percentages).sort_values()
print("Correct % of NaN values in 'trend_label' per active pair:")
print(nan_series)


