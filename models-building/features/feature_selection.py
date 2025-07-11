import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("data/semifinal_ohlcv.csv")
print(df.columns)


# Step 1: Identify and separate columns to exclude
pair_columns = [col for col in df.columns if col.startswith("pair_")]
columns_to_exclude = pair_columns + ['time', 'pair_name']
df_excluded = df[columns_to_exclude].copy()

# Step 2: Drop them from the feature set
df_features = df.drop(columns=columns_to_exclude)

# Step 3: Compute correlation matrix
correlation_matrix = df_features.corr()

# Optional: Visualize the correlation heatmap
plt.figure(figsize=(18, 14))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title("Correlation Heatmap (Without Time & Pair Columns)")
plt.tight_layout()
plt.show()

# Step 4: Identify redundant features (correlation > 0.9) — but protect essential OHLC columns
threshold = 0.9
protected_features = ['open', 'high', 'low', 'close']  # These will not be dropped

upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
to_drop = [
    column for column in upper_triangle.columns
    if any(abs(upper_triangle[column]) > threshold) and column not in protected_features
]

print("Highly correlated (redundant) features to drop (excluding protected OHLC columns):")
print(to_drop)

# Step 5: Drop the selected redundant features
df_selected = df_features.drop(columns=to_drop)

# Step 6: Add back time and pair columns
final_df = pd.concat([df_selected, df_excluded], axis=1)

# Step 7: Save the final DataFrame
final_df.to_csv("semifinal_ohlcv_2.csv", index=False)
print(f"\nFinal dataset saved as 'semifinal_ohlcv_2.csv'")
print(f"Final shape: {final_df.shape}")