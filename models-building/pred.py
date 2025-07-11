import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import pandas_ta as ta
import time
import os
import sqlite3
from joblib import load
from datetime import datetime

# === Configuration ===
PAIRS = [
    'AUDUSD', 'EURUSD', 'GBPUSD', 'NZDUSD', 'USDCAD',
    'USDCHF', 'USDHKD', 'USDNOK', 'USDSEK'
]
TIMEFRAME = mt5.TIMEFRAME_H4
CANDLES = 500
MODEL_PATH = "models"
DB_PATH = "prediction_logs.db"

# === Initialize MetaTrader 5 ===
if not mt5.initialize():
    print("MetaTrader5 initialization failed")
    quit()

# === SQLite Setup ===
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            pair TEXT,
            trend_class INTEGER,
            trend_label TEXT,
            vol_class INTEGER,
            vol_label TEXT
        )
    ''')
    conn.commit()
    conn.close()

# === Compute technical indicators ===
def compute_indicators(df):
    df.ta.sma(length=14, append=True)
    df.ta.adx(length=14, append=True)
    df.ta.stoch(k=14, d=3, smooth_k=3, append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.cci(length=20, append=True)
    df.ta.roc(length=10, append=True)
    df.ta.atr(length=14, append=True)

    # Bollinger Bands
    bb = df.ta.bbands(length=20, std=2.0, append=True)
    df['bb_width'] = bb['BBU_20_2.0'] - bb['BBL_20_2.0']

    # OBV needs volume, we map tick_volume
    df['volume'] = df['tick_volume']
    df.ta.obv(append=True)

    df.ta.mfi(length=14, append=True)
    df.ta.macd(append=True)

    # Candle features
    df['candle_body'] = abs(df['close'] - df['open'])
    df['candle_range'] = df['high'] - df['low']

    # Rename to match model features
    rename_map = {
        'SMA_14': 'sma_14',
        'ADX_14': 'adx_14',
        'STOCHk_14_3_3': 'stoch_k',
        'RSI_14': 'rsi_14',
        'CCI_20_0.015': 'cci_20',
        'ROC_10': 'roc_10',
        'ATRr_14': 'atr_14',
        'OBV': 'obv',
        'MFI_14': 'mfi_14',
        'MACD_12_26_9': 'macd_line',
        'MACDh_12_26_9': 'macd_hist'
    }
    df.rename(columns=rename_map, inplace=True)

    return df

# === Insert into SQLite DB ===
def log_to_db(entry):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO predictions (timestamp, pair, trend_class, trend_label, vol_class, vol_label)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (
        entry['timestamp'], entry['pair'],
        entry['trend_class'], entry['trend_label'],
        entry['vol_class'], entry['vol_label']
    ))
    conn.commit()
    conn.close()

# === Feature columns used for model prediction ===
FEATURE_COLS = [
    'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume',
    'sma_14', 'adx_14', 'stoch_k', 'rsi_14', 'cci_20', 'roc_10', 'atr_14',
    'bb_width', 'obv', 'mfi_14', 'macd_line', 'macd_hist', 'candle_body', 'candle_range'
]

# === Label maps ===
trend_map = {-1: "Downtrend", 0: "Ranging", 1: "Uptrend"}
vol_map = {0: "Low", 1: "Medium", 2: "High"}

# === Initialize database if not present ===
init_db()

# === Run prediction once ===
print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting prediction test run...\n")

for pair in PAIRS:
    rates = mt5.copy_rates_from_pos(pair, TIMEFRAME, 0, CANDLES)
    if rates is None or len(rates) < 60:
        print(f"[WARNING] Not enough data for {pair}")
        continue

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df = compute_indicators(df)
    print(f"[DEBUG] {pair}: Rows before dropna = {len(df)}, after dropna = {len(df.dropna())}")
    df.dropna(inplace=True)

    if df.empty:
        print(f"[SKIPPED] {pair}: Indicator calculation resulted in empty data.")
        continue

    latest = df.iloc[-1]
    try:
        X = latest[FEATURE_COLS].values.reshape(1, -1)
    except KeyError as e:
        print(f"[ERROR] {pair}: Missing feature column(s): {e}")
        continue

    trend_model_file = os.path.join(MODEL_PATH, f"{pair}_model.joblib")
    vol_model_file = os.path.join(MODEL_PATH, f"{pair}_vol_model.joblib")

    if not os.path.exists(trend_model_file) or not os.path.exists(vol_model_file):
        print(f"[ERROR] Model file(s) missing for {pair}")
        continue

    try:
        trend_model = load(trend_model_file)
        vol_model = load(vol_model_file)
    except Exception as e:
        print(f"[ERROR] Could not load model for {pair}: {e}")
        continue

    trend_pred = trend_model.predict(X)[0]
    vol_pred = vol_model.predict(X)[0]

    entry = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'pair': pair,
        'trend_class': int(trend_pred),
        'trend_label': trend_map.get(trend_pred, "Unknown"),
        'vol_class': int(vol_pred),
        'vol_label': vol_map.get(vol_pred, "Unknown")
    }

    log_to_db(entry)
    print(f"[LOGGED] {pair}: Trend={entry['trend_label']}, Volatility={entry['vol_label']}")

# === Shutdown MetaTrader 5 ===
mt5.shutdown()
