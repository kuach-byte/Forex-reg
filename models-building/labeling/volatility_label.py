import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the dataset
df = pd.read_csv("data/semifinal_ohlcv_2.csv")

def label_volatility_per_pair(group, vol_column='atr_14'):
    group = group.copy()

    # Drop NaNs if any in volatility column
    group = group.dropna(subset=[vol_column])

    # Calculate percentiles
    low_thresh = group[vol_column].quantile(0.33)
    high_thresh = group[vol_column].quantile(0.66)

    def compute_vol_label(value):
        if value < low_thresh:
            return 0  # Low volatility
        elif value < high_thresh:
            return 1  # Medium volatility
        else:
            return 2  # High volatility

    group['volatility_label'] = group[vol_column].apply(compute_vol_label)
    return group

# Apply labeling per pair
def add_volatility_labels(df, vol_column='atr_14'):
    labeled_df = df.groupby('pair_name').apply(label_volatility_per_pair, vol_column=vol_column)
    labeled_df = labeled_df.reset_index(drop=True)
    return labeled_df

df = add_volatility_labels(df, vol_column='atr_14')  

os.makedirs('volatility_vis', exist_ok=True)

# Mapping for labels (optional for readability)
label_names = {0: 'Low', 1: 'Medium', 2: 'High'}

def plot_volatility_label_distribution(df, label_column='volatility_label', pair_column='pair_name'):
    for pair_name, group in df.groupby(pair_column):
        # Count the labels
        label_counts = group[label_column].value_counts().sort_index()
        label_counts = label_counts.reindex([0, 1, 2], fill_value=0)  # Ensure all labels exist

        # Bar plot
        plt.figure(figsize=(6, 4))
        bars = plt.bar(
            [label_names[i] for i in label_counts.index],
            label_counts.values,
            color=['green', 'orange', 'red']
        )
        plt.title(f'Volatility Label Distribution - {pair_name}')
        plt.xlabel('Volatility Label')
        plt.ylabel('Count')
        plt.grid(axis='y')
        plt.tight_layout()

        # Save plot
        filename = f'volatility_vis/{pair_name}_volatility_label_dist.png'.replace("/", "_")
        output_dir = "volatility_vis"
        plt.savefig(os.path.join(output_dir,filename))
        plt.close()

plot_volatility_label_distribution(df)

df.to_csv("final_vol.csv", index=False)
