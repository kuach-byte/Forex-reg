import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime

# Account credentials
LOGIN = ***
PASSWORD = "***"
SERVER = "***"

# USD-related forex pairs
usd_pairs = [
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "NZDUSD",
    "USDCAD", "USDCHF", "USDSEK", "USDNOK", "USDHKD"
]

# Connect to MetaTrader 5
print("Connecting to MetaTrader 5...")
if not mt5.initialize(login=LOGIN, password=PASSWORD, server=SERVER):
    print("Login failed:", mt5.last_error())
    quit()
print("Login successful!")

# Parameters
timeframe = mt5.TIMEFRAME_H4
bars = 3000
utc_to = datetime.now()
data_dict = {}

# Fetch data for each pair
for symbol in usd_pairs:
    if not mt5.symbol_select(symbol, True):
        print(f"Warning: Could not enable {symbol} in Market Watch")
        continue

    rates = mt5.copy_rates_from(symbol, timeframe, utc_to, bars)
    if rates is None or len(rates) == 0:
        print(f"No data retrieved for {symbol}")
        continue

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    data_dict[symbol] = df
    print(f"{symbol}: {len(df)} H4 candles fetched")

    # Save to CSV
    df.to_csv(f"{symbol}_H4.csv", index=False)
    print(f"Saved {symbol}_H4.csv")

# Disconnect from MT5
mt5.shutdown()
print("Disconnected from MetaTrader 5")

# Sample preview of EURUSD
if "EURUSD" in data_dict:
    print("\nSample from EURUSD:")
    print(data_dict["EURUSD"].head())
