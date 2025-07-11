import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

# Load labeled dataset
df = pd.read_csv("data/final_vol.csv")

# Target label column (adjust if using 'trend_label' instead)
target_col = 'volatility_label'  # or 'trend_label'
exclude_cols = ['time', 'pair_name', target_col] + [col for col in df.columns if col.startswith('pair_')]
features = [col for col in df.columns if col not in exclude_cols]

# Output folder for confusion matrices
output_dir = "visualizations/confusion_matrices"
os.makedirs(output_dir, exist_ok=True)

# Process each pair
for pair in df['pair_name'].dropna().unique():
    print(f"[INFO] Generating confusion matrix for {pair}...")

    # Filter data for this pair and sort chronologically
    pair_df = df[df['pair_name'] == pair].sort_values(by='time').dropna(subset=[target_col])
    
    if len(pair_df) < 100:
        print(f"[SKIPPED] {pair}: Not enough labeled data.")
        continue

    # Time-aware split
    split_idx = int(0.8 * len(pair_df))
    test_df = pair_df.iloc[split_idx:]
    X_test = test_df[features]
    y_test = test_df[target_col]

    # Load the corresponding saved model
    model_path = f"models/{pair}_vol_model.joblib"
    if not os.path.exists(model_path):
        print(f"[SKIPPED] {pair}: Model file not found at {model_path}.")
        continue

    model = joblib.load(model_path)

    # Predict on test data
    y_pred = model.predict(X_test)

    # Generate and save confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Low', 'Medium', 'High'])

    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    plt.title(f"Confusion Matrix - {pair} (Volatility)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{pair}_conf_matrix.png"))
    plt.close()

print("\nConfusion matrices saved in:", output_dir)
