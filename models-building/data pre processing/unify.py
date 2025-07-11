import pandas as pd
import glob
import os
from sklearn.preprocessing import OneHotEncoder

# Directory containing CSV files
DATA_DIR = "data/separate data" 

# Get all H4 CSV files
files = glob.glob(os.path.join(DATA_DIR, "*_H4.csv"))
print(f"Found {len(files)} files.")

dataframes = []

for file in files:
    try:
        pair = os.path.basename(file).split("_")[0]
        df = pd.read_csv(file)
        df['pair'] = pair

        # Keep essential columns
        expected_cols = {'time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume'}
        keep_cols = [col for col in df.columns if col in expected_cols]
        keep_cols.append('pair')
        df = df[keep_cols]

        # Drop missing values
        df.dropna(inplace=True)

        dataframes.append(df)

    except Exception as e:
        print(f"Error reading {file}: {e}")

# Combine all into one DataFrame
combined_df = pd.concat(dataframes, ignore_index=True)
print("Combined all dataframes.")

# One-hot encode 'pair' column
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
pair_encoded = encoder.fit_transform(combined_df[['pair']])
pair_encoded_df = pd.DataFrame(pair_encoded, columns=encoder.get_feature_names_out(['pair']))

# Remove 'pair' and join one-hot encoded version
combined_df = combined_df.drop(columns=['pair']).reset_index(drop=True)
combined_df = pd.concat([combined_df, pair_encoded_df], axis=1)

# Handle 'time' 
if 'time' in combined_df.columns:
    time_col = combined_df['time']
    combined_df = combined_df.drop(columns=['time'])
else:
    time_col = None



# Restore 'time' 
if time_col is not None:
    combined_df.insert(0, 'time', time_col)

# Save the final result
combined_df.to_csv("ohlcv.csv", index=False)
print("Saved encoded data to ohlcv.csv")
