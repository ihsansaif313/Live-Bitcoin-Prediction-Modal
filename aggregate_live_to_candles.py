"""
Bitcoin Live Trade Aggregation Script (Windowed Version)
Reads the tail of btc_trades_live.csv and produces complete 1-minute candles.
"""

import pandas as pd
import logging
import time
import os
import signal
from datetime import datetime, timedelta, timezone
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
INPUT_CSV = "btc_trades_live.csv"
OUTPUT_CSV = "btc_live_candles.csv"
REFRESH_INTERVAL = 1.0  # seconds
WINDOW_SIZE_BYTES = 100000 # Read last 100KB which is ~2000 trades
ROTATION_INTERVAL_SECONDS = 7200  # 2 hours
MAX_TRADES_TO_KEEP = 200000  # Keep ~2 hours of trades (typical: 100-150 trades/min)

running = True
last_rotation_time = time.time()

def safe_replace(tmp, target):
    """Atomic replace with retry logic for Windows file locks."""
    max_retries = 5
    for i in range(max_retries):
        try:
            if os.path.exists(target):
                os.remove(target)
            os.rename(tmp, target)
            return True
        except PermissionError:
            if i < max_retries - 1:
                time.sleep(0.5)
                continue
            raise
        except Exception as e:
            if i < max_retries - 1:
                time.sleep(0.5)
                continue
            raise
    return False

def get_last_trades(path: str, size: int) -> pd.DataFrame:
    """Reads the tail of the trade file directly."""
    if not os.path.exists(path): return pd.DataFrame()
    try:
        with open(path, 'rb') as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - size), os.SEEK_SET)
            lines = f.read().decode('utf-8', errors='ignore').splitlines()
            if len(lines) <= 1: return pd.DataFrame()
            
            # Skip first partial line if we aren't at the start of file
            data = [l.split(',') for l in lines[1:] if len(l.split(',')) == 6]
            if not data: 
                logger.debug("No valid 6-column rows found in window.")
                return pd.DataFrame()
            
            df = pd.DataFrame(data, columns=['tradeId', 'price', 'qty', 'quoteQty', 'time', 'isBuyerMaker'])
            df['price'] = pd.to_numeric(df['price'], errors='coerce')
            df['qty'] = pd.to_numeric(df['qty'], errors='coerce')
            df['time'] = pd.to_datetime(df['time'], format='ISO8601', utc=True)
            return df.dropna(subset=['price', 'time'])
    except Exception as e:
        logger.error(f"Error reading tail: {e}")
        return pd.DataFrame()

def rotate_trades_file(path: str) -> None:
    """
    Rotate the trades CSV file by keeping only the last MAX_TRADES_TO_KEEP rows.
    This prevents the file from growing indefinitely.
    """
    if not os.path.exists(path):
        return
    
    try:
        # Get file size
        file_size = os.path.getsize(path)
        file_size_mb = file_size / (1024 * 1024)
        
        logger.info(f"Rotating {path} (current size: {file_size_mb:.2f} MB)")
        
        # Read the entire file
        df = pd.read_csv(path)
        original_count = len(df)
        
        # Keep only the last MAX_TRADES_TO_KEEP rows
        if len(df) > MAX_TRADES_TO_KEEP:
            df = df.tail(MAX_TRADES_TO_KEEP)
            
            # Write back to file
            df.to_csv(path, index=False)
            
            new_size = os.path.getsize(path)
            new_size_mb = new_size / (1024 * 1024)
            removed_count = original_count - len(df)
            
            logger.info(f"Rotation complete: Removed {removed_count} old trades")
            logger.info(f"New size: {new_size_mb:.2f} MB (saved {file_size_mb - new_size_mb:.2f} MB)")
        else:
            logger.info(f"No rotation needed (only {len(df)} trades)")
            
    except Exception as e:
        logger.error(f"Error rotating file: {e}")

def aggregate_and_update():
    global running, last_rotation_time
    logger.info("Starting Windowed Aggregation (Zero-Gap Mode)")
    
    while running:
        try:
            # Check if rotation is needed (every 2 hours)
            current_time = time.time()
            if current_time - last_rotation_time >= ROTATION_INTERVAL_SECONDS:
                logger.info("Rotation interval reached, rotating trades file...")
                rotate_trades_file(INPUT_CSV)
                last_rotation_time = current_time
            
            df = get_last_trades(INPUT_CSV, WINDOW_SIZE_BYTES)
            if not df.empty:
                logger.info(f"Read {len(df)} trades from tail.")
                # 1. Floor to minutes
                df['minute'] = df['time'].dt.floor('1min')
                
                # 2. Filter out CURRENT minute
                current_minute = datetime.now(timezone.utc).replace(second=0, microsecond=0)
                completed_df = df[df['minute'] < current_minute]
                
                if not completed_df.empty:
                    logger.info(f"Processing {completed_df['minute'].nunique()} completed minutes.")
                    # 3. Aggregate
                    candles = completed_df.groupby('minute').agg(
                        open=('price', 'first'),
                        high=('price', 'max'),
                        low=('price', 'min'),
                        close=('price', 'last'),
                        volume=('qty', 'sum'),
                        numberOfTrades=('tradeId', 'count')
                    ).reset_index()
                    candles.rename(columns={'minute': 'timeOpen'}, inplace=True)
                    candles['timeClose'] = candles['timeOpen'] + timedelta(seconds=59, milliseconds=999)

                    # 4. Save/Update CSV
                    if os.path.exists(OUTPUT_CSV):
                        try:
                            existing = pd.read_csv(OUTPUT_CSV)
                            existing['timeOpen'] = pd.to_datetime(existing['timeOpen'], utc=True, errors='coerce')
                            existing = existing.dropna(subset=['timeOpen'])
                            
                            combined = pd.concat([existing, candles]).drop_duplicates('timeOpen', keep='last')
                            combined = combined.sort_values('timeOpen').reset_index(drop=True)
                            
                            # Atomic Write
                            tmp_out = OUTPUT_CSV + ".tmp"
                            combined.tail(500).to_csv(tmp_out, index=False)
                            safe_replace(tmp_out, OUTPUT_CSV)
                            logger.info(f"Updated {OUTPUT_CSV} with latest candles.")
                        except Exception as e:
                            logger.error(f"Error updating {OUTPUT_CSV}: {e}. Overwriting with fresh data.")
                            tmp_out = OUTPUT_CSV + ".tmp"
                            candles.to_csv(tmp_out, index=False)
                            safe_replace(tmp_out, OUTPUT_CSV)
                    else:
                        tmp_out = OUTPUT_CSV + ".tmp"
                        candles.to_csv(tmp_out, index=False)
                        safe_replace(tmp_out, OUTPUT_CSV)
                        logger.info(f"Created {OUTPUT_CSV}.")
                else:
                    logger.info("No completed minutes found in trades window.")
            else:
                logger.debug("Tail read returned empty DF.")
            
            time.sleep(REFRESH_INTERVAL)
        except Exception as e:
            logger.error(f"Loop error: {e}")
            time.sleep(1)

def signal_handler(sig, frame):
    global running
    running = False

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    aggregate_and_update()
