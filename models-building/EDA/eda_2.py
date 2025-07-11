import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
import os

# Load your combined dataframe
df = pd.read_csv("data/ohlcv.csv")  
df['time'] = pd.to_datetime(df['time'])

# Folder to save visualizations
output_dir = "visualizations_2"
os.makedirs(output_dir, exist_ok=True)

# Detect pair columns like 'pair_EURUSD', 'pair_USDJPY', etc.
pair_cols = [col for col in df.columns if col.startswith('pair_')]

# Loop through each currency pair
for pair_col in pair_cols:
    pair_name = pair_col.replace("pair_", "")
    pair_df = df[df[pair_col] == 1].copy()

    if pair_df.empty:
        print(f"No data for {pair_name}. Skipping...")
        continue

    # Prepare time index
    pair_df.set_index('time', inplace=True)

    # Calculate rolling stats
    for window in [10, 20, 50, 200]:
        pair_df[f'rolling_mean_{window}'] = pair_df['close'].rolling(window).mean()
        pair_df[f'rolling_std_{window}'] = pair_df['close'].rolling(window).std()

    # 1. OHLC line plot
    pair_df[['open', 'high', 'low', 'close']].plot(figsize=(15, 6), title=f'{pair_name} - OHLC Over Time')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{pair_name}_ohlc_trend.png")
    plt.close()

    # 2. Volume plot
    pair_df['tick_volume'].plot(figsize=(15, 3), title=f'{pair_name} - Tick Volume Over Time', color='orange')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{pair_name}_volume_trend.png")
    plt.close()

    # 3. Rolling Means
    pair_df[['close', 'rolling_mean_50', 'rolling_mean_200']].plot(figsize=(15, 5), title=f'{pair_name} - Close with Rolling Means')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{pair_name}_rolling_means.png")
    plt.close()

    # 4. Autocorrelation (ACF) plot
    fig, ax = plt.subplots(figsize=(10, 4))
    plot_acf(pair_df['close'].dropna(), lags=40, ax=ax)
    plt.title(f'{pair_name} - ACF of Close Price')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{pair_name}_acf.png")
    plt.close()

    print(f" Saved all plots for {pair_name}")

print(f"\n All charts saved in: {output_dir}")
