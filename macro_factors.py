"""
Macro Factors Collector
Fetches S&P 500 (^GSPC) and US Dollar Index (DX-Y.NYB) data.
Calculates log returns and z-scores for macro-economic correlation features.
"""

import os
import time
import logging
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timezone, timedelta

# --- Configuration ---
LOG_FILE = "logs/macro_factors.log"
CSV_FILE = "macro_factors.csv"
WINDOW_SIZE = 60 # For z-score

# --- Logging ---
if not os.path.exists("logs"):
    os.makedirs("logs")

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def calculate_z_score(series):
    if len(series) < WINDOW_SIZE:
        return 0.0
    rolling = series.tail(WINDOW_SIZE)
    mean = rolling.mean()
    std = rolling.std()
    if std == 0:
        return 0.0
    return (rolling.iloc[-1] - mean) / std

def fetch_macro_data():
    """
    Fetch macro data: 1m for live returns, and 1d for 20-day z-score.
    """
    tickers = {
        'spx': '^GSPC',
        'dxy': 'DX-Y.NYB'
    }
    
    results = {}
    
    for key, ticker_symbol in tickers.items():
        try:
            ticker = yf.Ticker(ticker_symbol)
            
            # 1. Get Live Return (1m bars)
            data_1m = ticker.history(period='5d', interval='1m')
            last_close = 0.0
            log_return = 0.0
            last_time = datetime.now(timezone.utc)
            
            if not data_1m.empty:
                last_close = float(data_1m.iloc[-1]['Close'])
                if len(data_1m) > 1:
                    prev_close = float(data_1m.iloc[-2]['Close'])
                    log_return = float(np.log(last_close / prev_close)) if prev_close > 0 else 0
                last_time = data_1m.index[-1].to_pydatetime()

            # 2. Get Stable Z-Score (Daily bars, 20-day window)
            data_1d = ticker.history(period='2mo', interval='1d')
            z_score = 0.0
            if len(data_1d) >= 20:
                closes_1d = data_1d['Close'].astype(float)
                # Calculate daily log returns
                returns_1d = np.log(closes_1d / closes_1d.shift(1)).dropna()
                # Z-score of the last daily return relative to the last 20 days
                rolling_window = returns_1d.tail(20)
                mean = rolling_window.mean()
                std = rolling_window.std()
                if std > 0:
                    z_score = float((rolling_window.iloc[-1] - mean) / std)
            
            results[key] = {
                'close': last_close,
                'return': log_return,
                'z_score': z_score,
                'timestamp': last_time
            }
                
        except Exception as e:
            logger.error(f"Error fetching {ticker_symbol}: {e}")
            results[key] = None
            
    return results

def bootstrap_history(lookback_days: int = 365):
    """
    Fetch historical macro data for training.
    YFinance permits up to 730 days of 1h data.
    """
    logger.info(f"Bootstrapping {lookback_days} days of macro history...")
    tickers = {'spx': '^GSPC', 'dxy': 'DX-Y.NYB'}
    
    start_date = datetime.now() - timedelta(days=lookback_days)
    
    combined_dfs = []
    for key, ticker in tickers.items():
        try:
            # 1h data is the best resolution for long-term history
            df = yf.download(ticker, start=start_date, interval='1h', progress=False)
            if df.empty: continue
            
            df = df[['Close']].copy()
            df.columns = [f'{key}_close']
            df.index = pd.to_datetime(df.index, utc=True)
            
            # Calculate returns on 1h bars
            df[f'returns_{key}'] = np.log(df[f'{key}_close'] / df[f'{key}_close'].shift(1))
            
            # Calculate rolling z-score (20-period on 1h bars)
            df[f'z_{key}'] = (df[f'returns_{key}'] - df[f'returns_{key}'].rolling(20).mean()) / df[f'returns_{key}'].rolling(20).std()
            
            combined_dfs.append(df)
        except Exception as e:
            logger.error(f"Bootstrap error for {ticker}: {e}")

    if not combined_dfs:
        logger.warning("No macro history bootstrapped.")
        return

    # Merge and interpolate to 1m
    history = pd.concat(combined_dfs, axis=1).ffill().dropna()
    history.index.name = 'timeOpen'
    
    # Create 1m range
    full_range = pd.date_range(start=history.index.min(), end=history.index.max(), freq='1min', tz='UTC')
    history = history.reindex(full_range).ffill()
    history.index.name = 'timeOpen'
    history.reset_index(inplace=True)
    history['timeOpen'] = history['timeOpen'].dt.strftime('%Y-%m-%dT%H:%M:00')
    
    # Save
    if os.path.exists(CSV_FILE):
        existing = pd.read_csv(CSV_FILE)
        history = pd.concat([history, existing]).drop_duplicates(subset=['timeOpen'], keep='last')
    
    history.sort_values('timeOpen').to_csv(CSV_FILE, index=False)
    logger.info(f"Bootstrapped {len(history)} macro rows.")

def init_csv():
    if not os.path.exists(CSV_FILE):
        cols = ['timeOpen', 'spx_close', 'dxy_close', 'returns_spx', 'returns_dxy', 'z_spx', 'z_dxy']
        pd.DataFrame(columns=cols).to_csv(CSV_FILE, index=False)
        logger.info(f"Initialized {CSV_FILE}")

def run_collector():
    logger.info("Starting Macro Factors Collector...")
    init_csv()
    
    last_min = -1
    
    while True:
        current_time = datetime.now(timezone.utc)
        current_min = current_time.minute
        
        # Run once per minute
        if current_min != last_min:
            logger.info(f"Collecting macro data for {current_time.strftime('%H:%M')} UTC")
            data = fetch_macro_data()
            
            if data['spx'] and data['dxy']:
                row = {
                    'timeOpen': current_time.replace(second=0, microsecond=0).strftime('%Y-%m-%dT%H:%M:00'),
                    'spx_close': data['spx']['close'],
                    'dxy_close': data['dxy']['close'],
                    'returns_spx': data['spx']['return'],
                    'returns_dxy': data['dxy']['return'],
                    'z_spx': data['spx']['z_score'],
                    'z_dxy': data['dxy']['z_score']
                }
                
                df = pd.DataFrame([row])
                tmp_out = CSV_FILE + ".tmp"
                # If file exists, we append properly by reading and writing or just using mode='a' with replace
                if os.path.exists(CSV_FILE):
                    try:
                        existing = pd.read_csv(CSV_FILE)
                        # Filter out empty/NA frames to avoid FutureWarning in concat
                        to_concat = [d for d in [existing, df] if not d.empty]
                        if to_concat:
                            combined = pd.concat(to_concat).drop_duplicates(subset=['timeOpen'], keep='last')
                            combined.to_csv(tmp_out, index=False)
                        else:
                            df.to_csv(tmp_out, index=False)
                    except:
                        df.to_csv(tmp_out, index=False)
                else:
                    df.to_csv(tmp_out, index=False)
                
                def safe_replace(tmp, target):
                    max_retries = 10
                    for i in range(max_retries):
                        try:
                            if os.path.exists(tmp):
                                os.replace(tmp, target)
                            return True
                        except PermissionError:
                            if i < max_retries - 1:
                                time.sleep(1.0) # Wait a bit longer
                                continue
                            logger.warning(f"Macro: Could not update CSV due to persistent lock. Skipping this minute.")
                    return False

                safe_replace(tmp_out, CSV_FILE)
                last_min = current_min
                logger.info("Macro data saved atomically.")
            else:
                logger.warning("Incomplete macro data fetched. Waiting...")
        
        time.sleep(10) # Check every 10s

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", type=int, default=0, help="Days to bootstrap history")
    args = parser.parse_args()

    if args.bootstrap > 0:
        bootstrap_history(args.bootstrap)
    else:
        # Auto-bootstrap if file is too small (e.g. less than 100 rows)
        if not os.path.exists(CSV_FILE) or len(pd.read_csv(CSV_FILE)) < 100:
            bootstrap_history(365)
            
        run_collector()
