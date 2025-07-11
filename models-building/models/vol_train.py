import pandas as pd
import os
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from lightgbm import LGBMClassifier
from joblib import dump, Parallel, delayed
import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv("data/final_vol.csv")

# Column setup
target_col = 'volatility_label'
exclude_cols = ['time', target_col, 'pair_name'] + [col for col in df.columns if col.startswith('pair_')]
features = [col for col in df.columns if col not in exclude_cols]

# Output directories
os.makedirs("model_logs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Search space for randomized tuning
param_dist = {
    'num_leaves': [31, 63, 127],
    'max_depth': [-1, 10, 20, 30],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 300],
    'min_child_samples': [10, 20, 50],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

# Function to process each pair
def process_pair(pair):
    print(f"\n[INFO] Processing {pair}...")
    result = {}

    # Prepare data for this pair
    pair_df = df[df['pair_name'] == pair].sort_values('time').dropna(subset=[target_col])
    if len(pair_df) < 100:
        print(f"[SKIPPED] {pair}: Not enough labeled samples.")
        return None

    # Time-based train/test split
    split_idx = int(0.8 * len(pair_df))
    train_df, test_df = pair_df.iloc[:split_idx], pair_df.iloc[split_idx:]
    X_train, y_train = train_df[features], train_df[target_col]
    X_test, y_test = test_df[features], test_df[target_col]

    # Initialize base model
    base_model = LGBMClassifier(objective='multiclass', num_class=3, random_state=42)

    # Randomized search
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=30,
        scoring='f1_macro',
        cv=3,
        verbose=0,
        n_jobs=-1,
        random_state=42
    )
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    report = classification_report(y_test, y_pred, digits=3)

    print(f"[RESULT] {pair} - Accuracy: {acc:.4f}, F1-macro: {f1:.4f}")

    # Save model and report
    dump(best_model, f"models/{pair}_vol_model.joblib")
    with open(f"model_logs/{pair}_vol_report.txt", "w") as f:
        f.write(f"== {pair} (Volatility) ==\n")
        f.write(f"Accuracy: {acc:.4f}\nMacro-F1: {f1:.4f}\n")
        f.write(f"Best Params: {search.best_params_}\n\n")
        f.write(report)

    # Return result
    result['pair'] = pair
    result['accuracy'] = acc
    result['f1_macro'] = f1
    return result

if __name__ == "__main__":
    # Run in parallel with limited cores (safe for Windows/laptops)
    pairs = df['pair_name'].dropna().unique()
    results = Parallel(n_jobs=4)(delayed(process_pair)(pair) for pair in pairs)  # limit to 4 cores

    # Filter out None results (skipped pairs)
    results = [res for res in results if res is not None]

    # Save summary
    summary_df = pd.DataFrame(results).sort_values(by='f1_macro', ascending=False)
    summary_df.to_csv("model_logs/volatility_summary_metrics.csv", index=False)
    print("\nAll volatility models saved to /models and logs in /model_logs.")
