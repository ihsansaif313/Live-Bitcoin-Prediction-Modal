"""
Bitcoin Historical Data Collection Script
Fetches 6 months of BTC/USDT 1-minute klines from Binance and saves to CSV.
"""

import requests
import pandas as pd
import logging
import time
from datetime import datetime, timedelta, timezone
import yaml
import sys
import json
from typing import List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
BINANCE_API_URL = "https://api.binance.com/api/v3/klines"
OUTPUT_CSV = "btc_historical.csv"
MAX_RETRIES = 5
RETRY_DELAY = 2  # seconds


def fetch_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> List[List]:
    """
    Fetch klines (candlestick data) from Binance API.
    
    Args:
        symbol: Trading pair (e.g., "BTCUSDT")
        interval: Kline interval (e.g., "1m")
        start_ms: Start time in milliseconds
        end_ms: End time in milliseconds
    
    Returns:
        List of klines, each kline is a list of values
    
    Raises:
        Exception: If all retries fail
    """
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": 1000  # Max allowed by Binance
    }
    
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(BINANCE_API_URL, params=params, timeout=10)
            
            # Handle rate limiting
            if response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', RETRY_DELAY * 2))
                logger.warning(f"Rate limited. Waiting {retry_after} seconds...")
                time.sleep(retry_after)
                continue
            
            response.raise_for_status()
            data = response.json()
            
            logger.info(f"Fetched {len(data)} klines from {datetime.fromtimestamp(start_ms/1000, tz=timezone.utc)}")
            return data
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"Attempt {attempt + 1}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))  # Exponential backoff
            else:
                raise Exception(f"Failed to fetch klines after {MAX_RETRIES} attempts") from e
    
    return []


def normalize_klines(raw: List[List]) -> pd.DataFrame:
    """
    Convert raw klines data to a normalized DataFrame.
    
    Args:
        raw: List of raw kline data from Binance API
    
    Returns:
        DataFrame with normalized columns and types
    """
    if not raw:
        return pd.DataFrame()
    
    # Binance klines format:
    # [0] Open time, [1] Open, [2] High, [3] Low, [4] Close, [5] Volume,
    # [6] Close time, [7] Quote asset volume, [8] Number of trades, ...
    
    df = pd.DataFrame(raw, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'num_trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])
    
    # Convert to proper types
    df['timeOpen'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
    df['timeClose'] = pd.to_datetime(df['close_time'], unit='ms', utc=True)
    df['open'] = df['open'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['close'] = df['close'].astype(float)
    df['volume'] = df['volume'].astype(float)
    df['numberOfTrades'] = df['num_trades'].astype(int)
    
    # Select only required columns
    df = df[['timeOpen', 'timeClose', 'open', 'high', 'low', 'close', 'volume', 'numberOfTrades']]
    
    return df


def paginate_6_months(symbol: str, interval: str) -> pd.DataFrame:
    """
    Fetch 6 months of historical klines data by paginating through time ranges.
    
    Args:
        symbol: Trading pair (e.g., "BTCUSDT")
        interval: Kline interval (e.g., "1m")
    
    Returns:
        DataFrame with all historical data
    """
    # Calculate time range (6 months ago to now)
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=180)  # Approximately 6 months
    
    logger.info(f"Fetching data from {start_time} to {end_time}")
    
    # Convert to milliseconds
    current_start_ms = int(start_time.timestamp() * 1000)
    end_ms = int(end_time.timestamp() * 1000)
    
    all_klines = []
    
    # Pagination loop
    while current_start_ms < end_ms:
        # Fetch up to 1000 klines (Binance limit)
        # For 1m interval: 1000 minutes = ~16.67 hours
        klines = fetch_klines(symbol, interval, current_start_ms, end_ms)
        
        if not klines:
            logger.warning("No more data returned, stopping pagination")
            break
        
        all_klines.extend(klines)
        
        # Update start time to the close time of the last kline + 1ms
        last_close_time = klines[-1][6]  # Close time is at index 6
        current_start_ms = last_close_time + 1
        
        # Small delay to respect rate limits (weight: 1 per request)
        time.sleep(0.1)
        
        # Log progress
        progress_pct = ((current_start_ms - int(start_time.timestamp() * 1000)) / 
                       (end_ms - int(start_time.timestamp() * 1000))) * 100
        logger.info(f"Progress: {progress_pct:.1f}%")
    
    logger.info(f"Total klines fetched: {len(all_klines)}")
    
    # Normalize all data
    df = normalize_klines(all_klines)
    
    return df


def save_csv(df: pd.DataFrame, path: str) -> None:
    """
    Save DataFrame to CSV with data integrity checks.
    
    Args:
        df: DataFrame to save
        path: Output CSV file path
    """
    logger.info("Processing data for integrity...")
    
    # Remove duplicates by timeOpen
    initial_count = len(df)
    df = df.drop_duplicates(subset=['timeOpen'], keep='first')
    duplicates_removed = initial_count - len(df)
    if duplicates_removed > 0:
        logger.info(f"Removed {duplicates_removed} duplicate rows")
    
    # Sort by timeOpen
    df = df.sort_values('timeOpen').reset_index(drop=True)
    logger.info("Data sorted by timeOpen")
    
    # Validate gaps (1-minute intervals)
    gaps_found = 0
    for i in range(1, len(df)):
        expected_time = df.loc[i-1, 'timeOpen'] + timedelta(minutes=1)
        actual_time = df.loc[i, 'timeOpen']
        
        if actual_time != expected_time:
            gap_minutes = (actual_time - expected_time).total_seconds() / 60
            logger.warning(f"Gap detected: {gap_minutes:.0f} minutes missing after {df.loc[i-1, 'timeOpen']}")
            gaps_found += 1
    
    if gaps_found == 0:
        logger.info("No gaps detected - data is continuous")
    else:
        logger.warning(f"Total gaps found: {gaps_found}")
    
    # Save to CSV
    df.to_csv(path, index=False)
    logger.info(f"Data saved to {path}")
    logger.info(f"Total rows: {len(df)}")
    logger.info(f"Date range: {df['timeOpen'].min()} to {df['timeOpen'].max()}")


def main():
    """Main execution function."""
    logger.info("Starting Bitcoin historical data collection...")
    
    # Load Config
    config = {}
    if os.path.exists("config.yaml"):
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
    
    symbol = config.get('params', {}).get('symbol', 'BTCUSDT')
    output_path = config.get('paths', {}).get('historical_data', OUTPUT_CSV)
    
    try:
        # Fetch 6 months of BTC/USDT 1-minute data
        df = paginate_6_months(symbol=symbol, interval="1m")
        
        if df.empty:
            msg = "No data fetched from Binance API. Check internet connection or API availability."
            logger.error(msg)
            print(f"CRITICAL ERROR: {msg}", file=sys.stderr)
            return
        
        # Save to CSV with integrity checks
        save_csv(df, output_path)
        
        logger.info("Data collection completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during data collection: {e}", exc_info=True)
        print(f"CRITICAL FATAL ERROR: {str(e)}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
