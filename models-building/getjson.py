import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import pandas_ta as ta
from datetime import datetime
import json

# === Configuration ===
TIMEFRAME = mt5.TIMEFRAME_H4
CANDLES = 500

# === Prompt User for Pair ===
user_pair = "EURUSD"

# === Initialize MetaTrader 5 ===
if not mt5.initialize():
    print("MetaTrader5 initialization failed")
    quit()

# === Technical Indicator Computation Function ===
def compute_indicators(df):
    df.ta.sma(length=14, append=True)
    df.ta.adx(length=14, append=True)
    df.ta.stoch(k=14, d=3, smooth_k=3, append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.cci(length=20, append=True)
    df.ta.roc(length=10, append=True)
    df.ta.atr(length=14, append=True)

    bb = df.ta.bbands(length=20, std=2.0, append=True)
    df['bb_width'] = bb['BBU_20_2.0'] - bb['BBL_20_2.0']

    df['volume'] = df['tick_volume']
    df.ta.obv(append=True)
    df.ta.mfi(length=14, append=True)
    df.ta.macd(append=True)

    df['candle_body'] = abs(df['close'] - df['open'])
    df['candle_range'] = df['high'] - df['low']

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

# === Fetch & Process Data ===
rates = mt5.copy_rates_from_pos(user_pair, TIMEFRAME, 0, CANDLES)

if rates is None or len(rates) < 60:
    print(f"[ERROR] Not enough data found for {user_pair}.")
else:
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df = compute_indicators(df)
    df.dropna(inplace=True)
    df['pair'] = user_pair

    if not df.empty:
        df.to_json("sample.json", orient="records", date_format="iso")
        print(f"[SUCCESS] Saved {len(df)} cleaned rows with indicators to sample.json")
    else:
        print("[WARNING] All rows dropped after computing indicators. Try a different pair or timeframe.")

# === Shutdown MetaTrader 5 ===
mt5.shutdown()
