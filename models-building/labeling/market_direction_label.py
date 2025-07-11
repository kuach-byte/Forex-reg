import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# Config
CSV_PATH = "data/semifinal_ohlcv_2.csv"
OUTPUT_CSV = "final_trend_direction.csv"
TREND_VIS_DIR = "visualizations/market_trend_vis"
DIST_VIS_DIR = "visualizations/market_trend_vis/distribution"
USE_EMA = True
SHORT_WINDOW = 5
LONG_WINDOW = 25
ATR_WINDOW = 14
THRESHOLD_MULTIPLIER = 0.5  # Multiplied with ATR for adaptive margin

# Create output folders
os.makedirs(TREND_VIS_DIR, exist_ok=True)
os.makedirs(DIST_VIS_DIR, exist_ok=True)

# Load dataset
df = pd.read_csv(CSV_PATH)
df['time'] = pd.to_datetime(df['time'])

# === Labeling logic ===
def label_trend_per_pair(group):
    group = group.sort_values(by='time').copy()

    # Moving averages
    if USE_EMA:
        group['ma_short'] = group['close'].ewm(span=SHORT_WINDOW, adjust=False).mean()
        group['ma_long'] = group['close'].ewm(span=LONG_WINDOW, adjust=False).mean()
    else:
        group['ma_short'] = group['close'].rolling(window=SHORT_WINDOW).mean()
        group['ma_long'] = group['close'].rolling(window=LONG_WINDOW).mean()

    # ATR: using High-Low range (simplified ATR)
    group['hl_range'] = group['high'] - group['low']
    group['atr'] = group['hl_range'].rolling(window=ATR_WINDOW).mean()

    def compute_label(row):
        if pd.isna(row['ma_short']) or pd.isna(row['ma_long']) or pd.isna(row['atr']):
            return np.nan
        diff = row['ma_short'] - row['ma_long']
        threshold = row['atr'] * THRESHOLD_MULTIPLIER
        if abs(diff) < threshold:
            return 0
        return 1 if diff > 0 else -1

    group['trend_label'] = group.apply(compute_label, axis=1)
    return group.drop(columns=['ma_short', 'ma_long', 'atr', 'hl_range'])

# Apply labeling
df_labeled = df.groupby('pair_name', group_keys=False).apply(label_trend_per_pair)

# === Plotting ===
for pair in df_labeled['pair_name'].dropna().unique():
    pair_df = df_labeled[df_labeled['pair_name'] == pair]

    # Skip if no valid labels
    if pair_df['trend_label'].dropna().empty:
        continue

    # Plot close price with trend label
    plt.figure(figsize=(14, 4))
    plt.plot(pair_df['time'], pair_df['close'], label='Close Price', alpha=0.6)
    plt.plot(pair_df['time'], pair_df['trend_label'], label='Trend Label', linewidth=1.2)
    plt.title(f"Trend Label (ATR-Adjusted, {'EMA' if USE_EMA else 'SMA'}) for {pair}")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(TREND_VIS_DIR, f"{pair}_trend.png"))
    plt.close()

    # Distribution bar chart
    label_counts = pair_df['trend_label'].value_counts(dropna=True).sort_index()
    label_counts = label_counts.reindex([-1, 0, 1], fill_value=0)

    plt.figure(figsize=(6, 4))
    label_counts.plot(kind='bar', color=['red', 'gray', 'green'])
    plt.title(f"Trend Label Distribution: {pair}")
    plt.xlabel("Trend Label (-1: Down, 0: Range, 1: Up)")
    plt.ylabel("Count")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(DIST_VIS_DIR, f"{pair}_distribution.png"))
    plt.close()

# Save labeled dataset
df_labeled.to_csv(OUTPUT_CSV, index=False)
print(f"\nLabeled dataset saved as: {OUTPUT_CSV}")
print(f"Trend plots saved in: {TREND_VIS_DIR}")
print(f"Distributions saved in: {DIST_VIS_DIR}")
