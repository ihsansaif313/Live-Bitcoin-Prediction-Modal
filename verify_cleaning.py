import pandas as pd
import numpy as np
import os
from data_cleaner import detect_outliers, remove_noise, clean_historical_data

def verify_cleaning():
    print("--- Starting Data Cleaning Verification ---")
    
    # 1. Create dummy data with outliers
    data = {
        'timeOpen': pd.date_range(start='2023-01-01', periods=200, freq='1min'),
        'open': np.random.normal(50000, 100, 200),
        'high': np.random.normal(50100, 100, 200),
        'low': np.random.normal(49900, 100, 200),
        'close': np.random.normal(50000, 100, 200),
        'volume': np.random.normal(10, 2, 200)
    }
    df = pd.DataFrame(data)
    
    # Inject Price Outlier (Z-score)
    df.loc[100, 'close'] = 70000 
    
    # Inject Price Jump (Noise)
    df.loc[150, 'close'] = df.loc[149, 'close'] * 1.15
    
    # Inject Volume Spike
    df.loc[50, 'volume'] = 10000
    
    print(f"Injected Outliers:\n"
          f" - Row 100 Close: {df.loc[100, 'close']}\n"
          f" - Row 150 Close (Jump): {df.loc[150, 'close']}\n"
          f" - Row 50 Volume: {df.loc[50, 'volume']}")
    
    # Test Clean
    df_clean = detect_outliers(df)
    df_clean = remove_noise(df_clean)
    
    print("\nCleaned Results:")
    print(f" - Row 100 Close (Winsorized/Clipped?): {df_clean.loc[100, 'close']}")
    print(f" - Row 150 Close (Jump Fixed?): {df_clean.loc[150, 'close']}")
    print(f" - Row 50 Volume (Spike Fixed?): {df_clean.loc[50, 'volume']}")
    
    # Verify
    assert df_clean.loc[150, 'close'] < 60000, "Price jump not corrected"
    assert df_clean.loc[50, 'volume'] < 500, "Volume spike not corrected"
    print("\nSUCCESS: Verification passed!")

if __name__ == "__main__":
    verify_cleaning()
