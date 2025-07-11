import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


df = pd.read_csv("data/ohlcv.csv") 

# Display the first few rows
print("Head of the dataset:")
print(df.head())

print("\nColumn names:")
print(df.columns)

# Ensure timestamp is datetime type
df['time'] = pd.to_datetime(df['time'])


print(df.describe())

#dropping row with volume <=0 some indicator depend on it
df = df[df['tick_volume'] > 0]

# UNIVARIATE ANALYSIS
def analyze_forex_pairs(df, output_dir="Visualizations_1"):
    # OHLCV columns to plot
    ohlcv_cols = ['open', 'high', 'low', 'close', 'tick_volume']
    
    # Create output folder if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Detect all currency pair columns
    pair_cols = [col for col in df.columns if col.startswith('pair_')]
    print(f"Found {len(pair_cols)} currency pairs:\n{pair_cols}")

    # Loop through each pair
    for pair_col in pair_cols:
        pair_name = pair_col.replace("pair_", "")
        print(f"\nAnalyzing {pair_name} ...")

        # Filter rows where this pair is active
        pair_df = df[df[pair_col] == 1]
        
        if pair_df.empty:
            print(f"No rows with {pair_col} == 1. Skipping...")
            continue

        for col in ohlcv_cols:
            if col not in pair_df.columns:
                print(f"Column '{col}' not found in data. Skipping...")
                continue

            plt.figure(figsize=(10, 4))
            sns.histplot(pair_df[col], bins=50, kde=True, color='skyblue')
            plt.title(f"{pair_name}: Distribution of {col.capitalize()}")
            plt.xlabel(col.capitalize())
            plt.ylabel('Frequency')
            plt.grid(True)
            plt.tight_layout()

            # Save plot
            filename = f"{pair_name}_{col}.png"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath)
            plt.close()  # Close the figure to avoid memory leaks

            print(f"Saved plot: {filepath}")

analyze_forex_pairs(df)
