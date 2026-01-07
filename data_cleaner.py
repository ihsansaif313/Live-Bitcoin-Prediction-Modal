import pandas as pd
import numpy as np
import logging
import os
from typing import List, Tuple

# Configure logging
if not os.path.exists("logs"):
    os.makedirs("logs")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/data_cleaning.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def log_cleaning(actions: List[str]) -> None:
    """Log cleaning actions for transparency."""
    for action in actions:
        logger.info(action)

def detect_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect outliers in price and volume using statistical methods.
    - Z-score (>3 sigma)
    - IQR (values outside Q1-1.5*IQR or Q3+1.5*IQR)
    - Percentile clipping (1% tails)
    """
    df_clean = df.copy()
    actions = []
    
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col not in df_clean.columns:
            continue
            
        # 1. Z-score (Rolling to handle trends)
        window = 60 # 1 hour
        rolling_mean = df_clean[col].rolling(window=window).mean()
        rolling_std = df_clean[col].rolling(window=window).std()
        z_scores = (df_clean[col] - rolling_mean) / rolling_std
        z_outliers = z_scores.abs() > 3
        
        # 2. IQR
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        iqr_outliers = (df_clean[col] < (Q1 - 1.5 * IQR)) | (df_clean[col] > (Q3 + 1.5 * IQR))
        
        # 3. Percentile Clipping (Extreme tails)
        lower_tail = df_clean[col].quantile(0.005)
        upper_tail = df_clean[col].quantile(0.995)
        percentile_outliers = (df_clean[col] < lower_tail) | (df_clean[col] > upper_tail)
        
        # Combine (only flag if multiple methods agree or extreme jump)
        # For simplicity and robustness, we will use a weighted approach or just flag any extreme
        total_outliers = z_outliers | percentile_outliers
        
        count = total_outliers.sum()
        if count > 0:
            actions.append(f"Detected {count} outliers in {col} using Z-score/Percentile check.")
            # Winsorize: Clip to tails instead of removing to maintain time continuity
            df_clean[col] = df_clean[col].clip(lower=lower_tail, upper=upper_tail)
            actions.append(f"Winsorized {col} to [0.5%, 99.5%] range.")

    log_cleaning(actions)
    return df_clean

def remove_noise(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect and fix anomalous candles.
    - Price jumps > 10% in 1 minute.
    - Volume spikes compared to rolling median.
    """
    df_clean = df.copy()
    actions = []
    
    # 1. Price Jump detection
    df_clean['price_change'] = df_clean['close'].pct_change().abs()
    jumps = df_clean['price_change'] > 0.10 # 10% jump
    jump_count = jumps.sum()
    
    if jump_count > 0:
        actions.append(f"Detected {jump_count} unrealistic price jumps (>10%).")
        # Correct jumps by smoothing with previous value (interpolation)
        df_clean.loc[jumps, 'close'] = df_clean['close'].shift(1)
        df_clean.loc[jumps, 'high'] = df_clean['close']
        df_clean.loc[jumps, 'low'] = df_clean['close']
        df_clean.loc[jumps, 'open'] = df_clean['close']
        actions.append("Corrected jumps using previous candle values.")
    
    # 2. Volume Spike smoothing
    window = 120 # 2 hours
    volume_median = df_clean['volume'].rolling(window=window).median()
    volume_spikes = df_clean['volume'] > (volume_median * 50) # 50x median volume
    spike_count = volume_spikes.sum()
    
    if spike_count > 0:
        actions.append(f"Detected {spike_count} extreme volume spikes (>50x median).")
        df_clean.loc[volume_spikes, 'volume'] = volume_median * 5
        actions.append("Capped volume spikes to 5x rolling median.")

    df_clean.drop(columns=['price_change'], inplace=True, errors='ignore')
    log_cleaning(actions)
    return df_clean

def clean_historical_data(input_path: str, output_path: str):
    """Integrate cleaning methods for historical data."""
    if not os.path.exists(input_path):
        logger.error(f"Input file {input_path} not found.")
        return

    logger.info(f"Starting cleaning on {input_path}...")
    df = pd.read_csv(input_path, low_memory=False)
    initial_rows = len(df)
    
    # Apply cleaning rules
    df = detect_outliers(df)
    df = remove_noise(df)
    
    # Ensure time sorted
    df['timeOpen'] = pd.to_datetime(df['timeOpen'], utc=True, errors='coerce')
    df = df.dropna(subset=['timeOpen'])
    df = df.sort_values('timeOpen').reset_index(drop=True)
    
    def safe_replace(tmp, target):
        max_retries = 5
        for i in range(max_retries):
            try:
                if os.name == 'nt' and os.path.exists(target):
                    os.remove(target) # Windows sometimes needs explicit remove
                    os.rename(tmp, target)
                else:
                    os.replace(tmp, target)
                return True
            except PermissionError:
                if i < max_retries - 1:
                    time.sleep(0.5)
                    continue
                raise
        return False

    tmp_out = output_path + ".tmp"
    df.to_csv(tmp_out, index=False)
    safe_replace(tmp_out, output_path)
    logger.info(f"Cleaned data saved to {output_path}. Rows: {len(df)} (Original: {initial_rows})")

if __name__ == "__main__":
    # For testing
    HIST_PATH = "btc_historical.csv"
    CLEAN_PATH = "btc_historical_clean.csv"
    if os.path.exists(HIST_PATH):
        clean_historical_data(HIST_PATH, CLEAN_PATH)
