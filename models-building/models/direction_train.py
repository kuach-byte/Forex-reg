import pandas as pd
import os
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMClassifier
from joblib import dump

# Load the dataset
df = pd.read_csv("data/final_trend_direction.csv")

# Define columns
target_col = 'trend_label'
exclude_cols = ['time', 'trend_label', 'pair_name'] + [col for col in df.columns if col.startswith('pair_')]
features = [col for col in df.columns if col not in exclude_cols]

# Create output folders
os.makedirs("model_logs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Grid search space
param_grid = {
    'num_leaves': [31, 63],
    'max_depth': [-1, 10, 20],
    'learning_rate': [0.05, 0.1],
    'n_estimators': [100, 200],
    'min_child_samples': [20, 50],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# Store results
results = {}

for pair in df['pair_name'].dropna().unique():
    print(f"\n[INFO] Processing {pair}...")

    # Prepare data
    pair_df = df[df['pair_name'] == pair].sort_values('time').dropna(subset=[target_col])

    if len(pair_df) < 100:
        print(f"[SKIPPED] {pair}: Not enough labeled samples.")
        continue

    # Time-aware split
    split_idx = int(0.8 * len(pair_df))
    train_df = pair_df.iloc[:split_idx]
    test_df = pair_df.iloc[split_idx:]

    X_train = train_df[features]
    y_train = train_df[target_col]
    X_test = test_df[features]
    y_test = test_df[target_col]

    # Base model for tuning
    base_model = LGBMClassifier(objective='multiclass', num_class=3, random_state=42)

    # Grid search
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring='f1_macro',
        cv=3,
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)

    # Best model
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # Evaluation
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    report = classification_report(y_test, y_pred, digits=3)

    print(f"[RESULT] Accuracy: {acc:.4f}, Macro-F1: {f1:.4f}")
    print(f"[BEST PARAMS] {grid_search.best_params_}")

    # Save results
    results[pair] = {
        'accuracy': acc,
        'f1_macro': f1,
        'report': report,
        'best_params': grid_search.best_params_
    }

    # Save report
    with open(f"model_logs/{pair}_report.txt", "w") as f:
        f.write(f"== {pair} ==\n")
        f.write(f"Accuracy: {acc:.4f}\nMacro-F1: {f1:.4f}\n")
        f.write(f"Best Params: {grid_search.best_params_}\n\n")
        f.write(report)

    # Save model
    dump(best_model, f"models/{pair}_model.joblib")

# Save summary
summary = pd.DataFrame([
    {'pair': pair, 'accuracy': res['accuracy'], 'f1_macro': res['f1_macro']}
    for pair, res in results.items()
]).sort_values(by='f1_macro', ascending=False)

summary.to_csv("model_logs/summary_metrics.csv", index=False)
print("\nAll tuned models saved in /models and reports in /model_logs.")
