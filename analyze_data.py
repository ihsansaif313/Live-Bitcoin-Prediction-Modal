import pandas as pd
import numpy as np

# Load historical data
df = pd.read_csv('btc_historical.csv')

with open('data_analysis_report.txt', 'w') as f:
    f.write(f"Total rows: {len(df)}\n")
    f.write(f"\nColumns: {df.columns.tolist()}\n")
    
    # Convert timeOpen to datetime
    df['timeOpen'] = pd.to_datetime(df['timeOpen'])
    
    f.write(f"\nDate range: {df['timeOpen'].min()} to {df['timeOpen'].max()}\n")
    
    # Check for price anomalies
    f.write("\n=== PRICE STATISTICS ===\n")
    f.write(str(df[['open', 'high', 'low', 'close']].describe()))
    
    # Check for outliers (candles with extreme ranges)
    df['candle_range'] = df['high'] - df['low']
    df['candle_range_pct'] = (df['candle_range'] / df['close']) * 100
    
    f.write("\n\n=== CANDLE RANGE STATISTICS ===\n")
    f.write(str(df['candle_range'].describe()))
    f.write(f"\n\nCandle range as % of price:\n")
    f.write(str(df['candle_range_pct'].describe()))
    
    # Find extreme candles
    f.write("\n\n=== TOP 10 LARGEST CANDLES ===\n")
    largest = df.nlargest(10, 'candle_range')[['timeOpen', 'open', 'high', 'low', 'close', 'candle_range', 'candle_range_pct']]
    f.write(str(largest))
    
    # Check for data quality issues
    f.write("\n\n=== DATA QUALITY CHECKS ===\n")
    f.write(f"Rows with zero prices: {(df[['open', 'high', 'low', 'close']] == 0).any(axis=1).sum()}\n")
    f.write(f"Rows with negative prices: {(df[['open', 'high', 'low', 'close']] < 0).any(axis=1).sum()}\n")
    f.write(f"Rows where high < low: {(df['high'] < df['low']).sum()}\n")
    f.write(f"Rows where close > high: {(df['close'] > df['high']).sum()}\n")
    f.write(f"Rows where close < low: {(df['close'] < df['low']).sum()}\n")
    f.write(f"\nDuplicate timestamps: {df['timeOpen'].duplicated().sum()}\n")

print("Analysis complete. Check data_analysis_report.txt")
