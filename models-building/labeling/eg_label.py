import pandas as pd
import matplotlib.pyplot as plt
import os

# === Configurable Parameters ===
short_window = 5
long_window = 15
margin_ratio = 0.005  # 0.5% threshold
use_ema = True        # ✅ Toggle this between EMA or SMA
debug_pair = "AUDUSD"  # ✅ Set to a pair name to visualize moving averages

# === Load and prepare dataset ===
df = pd.read_csv("data/semifinal_ohlcv_2.csv")
df['time'] = pd.to_datetime(df['time'])

# === Output folders ===
trend_vis_dir = "market_trend_vis"
dist_vis_dir = os.path.join(trend_vis_dir, "distribution")
debug_vis_dir = os.path.join(trend_vis_dir, "debug")
os.makedirs(trend_vis_dir, exist_ok=True)
os.makedirs(dist_vis_dir, exist_ok=True)
os.makedirs(debug_vis_dir, exist_ok=True)

# === Labeling function ===
def label_trend_per_pair(group):
    group = group.sort_values(by='time').copy()
    
    if use_ema:
        group['ma_short'] = group['close'].ewm(span=short_window, adjust=False).mean()
        group['ma_long'] = group['close'].ewm(span=long_window, adjust=False).mean()
    else:
        group['ma_short'] = group['close'].rolling(window=short_window).mean()
        group['ma_long'] = group['close'].rolling(window=long_window).mean()
    
    def compute_label(row):
        if pd.isna(row['ma_short']) or pd.isna(row['ma_long']):
            return None
        diff = row['ma_short'] - row['ma_long']
        if abs(diff) < margin_ratio * row['ma_long']:
            return 0  # Ranging
        elif diff > 0:
            return 1  # Uptrend
        else:
            return -1  # Downtrend

    group['trend_label'] = group.apply(compute_label, axis=1)

    # Debug plot for selected pair
    if group['pair_name'].iloc[0] == debug_pair:
        plt.figure(figsize=(14, 5))
        plt.plot(group['time'], group['close'], label="Close", alpha=0.6)
        plt.plot(group['time'], group['ma_short'], label=f"{'EMA' if use_ema else 'SMA'} {short_window}")
        plt.plot(group['time'], group['ma_long'], label=f"{'EMA' if use_ema else 'SMA'} {long_window}")
        plt.title(f"Moving Averages for {debug_pair}")
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(debug_vis_dir, f"{debug_pair}_ma_plot.png"))
        plt.close()

    return group.drop(columns=['ma_short', 'ma_long'])

# === Apply labeling logic ===
df_labeled = df.groupby('pair_name', group_keys=False).apply(label_trend_per_pair)

# === Plot and save trend labels + distributions ===
for pair in df_labeled['pair_name'].dropna().unique():
    pair_df = df_labeled[df_labeled['pair_name'] == pair]
    if pair_df['trend_label'].notna().sum() == 0:
        continue

    # Price and label time series
    plt.figure(figsize=(14, 4))
    plt.plot(pair_df['time'], pair_df['close'], label='Close Price', alpha=0.6)
    plt.plot(pair_df['time'], pair_df['trend_label'], label='Trend Label', linewidth=1.5)
    plt.title(f"Trend Label ({'EMA' if use_ema else 'SMA'}) for {pair}")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(trend_vis_dir, f"{pair}_trend.png"))
    plt.close()

    # Label distribution
    label_counts = pair_df['trend_label'].value_counts(dropna=True).sort_index()
    plt.figure(figsize=(6, 4))
    label_counts.plot(kind='bar', color=['red', 'gray', 'green'])
    plt.title(f"Trend Label Distribution for {pair}")
    plt.xlabel("Label (-1: Down, 0: Range, 1: Up)")
    plt.ylabel("Count")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(dist_vis_dir, f"{pair}_distribution.png"))
    plt.close()

# === Save final labeled data ===
df_labeled.to_csv("final_trend_direction.csv", index=False)
print("Trend labeling complete!")
print("Time series plots saved in:", trend_vis_dir)
print("Label distribution plots saved in:", dist_vis_dir)
print("Debug MA plot saved in:", debug_vis_dir)
