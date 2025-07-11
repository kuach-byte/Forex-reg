import pandas as pd
import numpy as np
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler

# === CONFIGURATION ===
INPUT_FILE = "data/ohlcv.csv"
OUTPUT_FILE = "data/semifinal_ohlcv.csv"
MIN_ROWS = 10  # Minimum required rows per pair to compute indicators


# === STEP 1: Load Data ===
df = pd.read_csv(INPUT_FILE)
print("Loaded dataset with columns:\n", df.columns)

print("\nInitial Pair Row Counts (Raw Data):")
for col in df.columns:
    if col.startswith("pair_"):
        count = df[df[col] == 1].shape[0]
        print(f"{col}: {count} rows")


# === STEP 2: Replace Outliers in OHLCV-related Columns Only ===
def replace_outliers_with_mean(df):
    df_cleaned = df.copy()
    
    target_cols = ['open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']
    for col in target_cols:
        if col in df_cleaned.columns:
            Q1 = df_cleaned[col].quantile(0.25)
            Q3 = df_cleaned[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            is_outlier = (df_cleaned[col] < lower_bound) | (df_cleaned[col] > upper_bound)
            non_outlier_mean = df_cleaned.loc[~is_outlier, col].mean()
            df_cleaned.loc[is_outlier, col] = non_outlier_mean

    return df_cleaned

df_clean = replace_outliers_with_mean(df)

print("\nPair Row Counts After Outlier Replacement:")
for col in df_clean.columns:
    if col.startswith("pair_"):
        count = df_clean[df_clean[col] == 1].shape[0]
        print(f"{col}: {count} rows")


# === STEP 3: Compute 20 Technical Indicators ===
def compute_clean_20_indicators(df):
    df = df.copy()

    def safe_add(col_name, result, key=None):
        if result is not None:
            try:
                if isinstance(result, pd.Series):
                    df[col_name] = result
                    print(f"  Added: {col_name}")
                elif isinstance(result, pd.DataFrame) and key in result.columns:
                    df[col_name] = result[key]
                    print(f"  Added: {col_name}")
                else:
                    print(f"  Skipped: {col_name} (missing key '{key}')")
            except Exception as e:
                print(f"  Failed: {col_name} ({e})")
        else:
            print(f"  Null result for: {col_name}")

    # Trend indicators
    safe_add('sma_14', ta.sma(df['close'], length=14))
    safe_add('ema_20', ta.ema(df['close'], length=20))
    safe_add('adx_14', ta.adx(df['high'], df['low'], df['close'], length=14), 'ADX_14')

    # Momentum indicators
    stoch = ta.stoch(df['high'], df['low'], df['close'])
    safe_add('stoch_k', stoch, 'STOCHk_14_3_3')
    safe_add('stoch_d', stoch, 'STOCHd_14_3_3')
    safe_add('rsi_14', ta.rsi(df['close'], length=14))
    safe_add('cci_20', ta.cci(df['high'], df['low'], df['close'], length=20))
    safe_add('roc_10', ta.roc(df['close'], length=10))
    safe_add('willr_14', ta.willr(df['high'], df['low'], df['close'], length=14))
    safe_add('cmo_14', ta.cmo(df['close'], length=14))

    # Volatility indicators
    safe_add('atr_14', ta.atr(df['high'], df['low'], df['close'], length=14))
    bb = ta.bbands(df['close'], length=20, std=2)
    safe_add('bb_width', bb, 'BBB_20_2.0')

    # Volume indicators
    safe_add('obv', ta.obv(df['close'], df['tick_volume']))
    safe_add('mfi_14', ta.mfi(df['high'], df['low'], df['close'], df['tick_volume'], length=14))

    # Price action
    macd = ta.macd(df['close'])
    safe_add('macd_line', macd, 'MACD_12_26_9')
    safe_add('macd_hist', macd, 'MACDh_12_26_9')
    safe_add('bb_upper', bb, 'BBU_20_2.0')
    safe_add('bb_lower', bb, 'BBL_20_2.0')

    # Derived features
    df['candle_body'] = df['close'] - df['open']
    df['candle_range'] = df['high'] - df['low']

    print(f"    Rows before dropna: {len(df)}")
    df.dropna(inplace=True)
    print(f"    Rows after dropna: {len(df)}")

    return df


# === STEP 4: Normalize Only New Indicator Columns ===
def normalize_indicators(df, original_cols):
    df = df.copy()
    new_cols = [col for col in df.columns if col not in original_cols]
    if new_cols:
        scaler = MinMaxScaler()
        df[new_cols] = scaler.fit_transform(df[new_cols])
    return df


# === STEP 5: Process Each Currency Pair Group ===
def process_all_pairs(df, min_rows=10):
    pair_columns = [col for col in df.columns if col.startswith("pair_")]
    original_cols = list(df.columns)
    result_frames = []

    print("\nProcessing Each Currency Pair")

    for pair_col in pair_columns:
        pair_df = df[df[pair_col] == 1].copy().sort_values("time")

        if pair_df.empty or len(pair_df) < min_rows:
            print(f"Skipping {pair_col}: only {len(pair_df)} rows")
            continue

        try:
            print(f"\nProcessing {pair_col} with {len(pair_df)} rows")
            enriched_df = compute_clean_20_indicators(pair_df)

            if enriched_df.empty:
                print(f"All rows dropped after dropna in {pair_col}")
                continue

            normalized_df = normalize_indicators(enriched_df, original_cols)
            normalized_df["pair_name"] = pair_col.replace("pair_", "")
            result_frames.append(normalized_df)

        except Exception as e:
            print(f"Error processing {pair_col}: {e}")

    if result_frames:
        final_df = pd.concat(result_frames, ignore_index=True).sort_values("time")
        print(f"\nFinal dataset shape: {final_df.shape}")
    else:
        final_df = pd.DataFrame()
        print("\nFinal dataset is empty. No pairs were processed successfully.")

    return final_df


# === STEP 6: Execute Pipeline and Save Final Output ===
df_final = process_all_pairs(df_clean, min_rows=MIN_ROWS)

if not df_final.empty:
    df_final.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved processed data to: {OUTPUT_FILE}")
else:
    print("\nNo data was saved. Final dataset is empty.")
