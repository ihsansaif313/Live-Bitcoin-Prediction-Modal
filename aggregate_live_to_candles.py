"""
Bitcoin Live Trade Aggregation Script
Reads raw trades from btc_trades_live.csv and produces 1-minute candles.
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
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
INPUT_CSV = "btc_trades_live.csv"
OUTPUT_CSV = "btc_live_candles.csv"
REFRESH_INTERVAL = 0.5  # seconds

# Global state
running = True
last_processed_timestamp = None


def load_new_trades(last_ts: Optional[datetime]) -> pd.DataFrame:
    """
    Load only new trades since the last processed timestamp.
    
    Args:
        last_ts: Last processed timestamp (None to load all)
    
    Returns:
        DataFrame with new trades
    """
    try:
        if not os.path.exists(INPUT_CSV):
            logger.debug(f"{INPUT_CSV} does not exist yet")
            return pd.DataFrame()
        
        logger.debug(f"Reading {INPUT_CSV}...")
        
        # Read the CSV
        df = pd.read_csv(INPUT_CSV)
        
        logger.debug(f"Read {len(df)} rows from CSV")
        
        if df.empty:
            logger.debug("CSV is empty")
            return pd.DataFrame()
        
        # Convert time column to datetime with explicit format
        df['time'] = pd.to_datetime(df['time'], format='mixed', utc=True)
        
        logger.debug(f"Converted time column, last_ts={last_ts}")
        
        # Filter for new trades only
        if last_ts is not None:
            df = df[df['time'] > last_ts]
            logger.debug(f"Filtered to {len(df)} trades after {last_ts}")
        
        logger.debug(f"Loaded {len(df)} new trades")
        return df
        
    except Exception as e:
        logger.error(f"Error loading trades: {e}", exc_info=True)
        return pd.DataFrame()


def aggregate_minute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate trades into 1-minute candles.
    
    Args:
        df: DataFrame with raw trades
    
    Returns:
        DataFrame with 1-minute candles
    """
    if df.empty:
        return pd.DataFrame()
    
    try:
        # Floor timestamps to minute boundaries
        df['minute'] = df['time'].dt.floor('1min')
        
        # Get current time to exclude incomplete minute
        current_time = datetime.now(timezone.utc)
        current_minute = current_time.replace(second=0, microsecond=0)
        
        logger.debug(f"Current minute: {current_minute}")
        logger.debug(f"Total minutes in data: {df['minute'].nunique()}")
        
        # Only aggregate completed minutes
        df = df[df['minute'] < current_minute]
        
        if df.empty:
            logger.debug("No completed minutes to aggregate")
            return pd.DataFrame()
        
        logger.debug(f"Completed minutes to aggregate: {df['minute'].nunique()}")
        
        # Group by minute and aggregate
        candles = df.groupby('minute').agg(
            open=('price', 'first'),
            high=('price', 'max'),
            low=('price', 'min'),
            close=('price', 'last'),
            volume=('qty', 'sum'),
            numberOfTrades=('tradeId', 'count')
        ).reset_index()
        
        # Rename and add timeClose
        candles.rename(columns={'minute': 'timeOpen'}, inplace=True)
        candles['timeClose'] = candles['timeOpen'] + timedelta(seconds=59, milliseconds=999)
        
        # Reorder columns to match historical schema
        candles = candles[['timeOpen', 'timeClose', 'open', 'high', 'low', 'close', 'volume', 'numberOfTrades']]
        
        logger.info(f"Aggregated {len(candles)} candles")
        return candles
        
    except Exception as e:
        logger.error(f"Error aggregating candles: {e}")
        return pd.DataFrame()


def append_candles(df: pd.DataFrame, path: str) -> None:
    """
    Append candles to CSV with deduplication.
    
    Args:
        df: DataFrame with candles to append
        path: Output CSV file path
    """
    if df.empty:
        return
    
    try:
        # Check if output file exists
        if os.path.exists(path):
            # Load existing candles
            existing = pd.read_csv(path)
            existing['timeOpen'] = pd.to_datetime(existing['timeOpen'], utc=True)
            
            # Combine and deduplicate by timeOpen
            combined = pd.concat([existing, df], ignore_index=True)
            combined = combined.drop_duplicates(subset=['timeOpen'], keep='last')
            combined = combined.sort_values('timeOpen').reset_index(drop=True)
            
            # Calculate how many new candles were added
            new_count = len(combined) - len(existing)
            
            if new_count > 0:
                # Write back the entire file
                combined.to_csv(path, index=False)
                logger.info(f"Appended {new_count} new candles to {path}")
            else:
                logger.debug("No new candles to append")
        else:
            # First time - just write
            df.to_csv(path, index=False)
            logger.info(f"Created {path} with {len(df)} candles")
            
    except Exception as e:
        logger.error(f"Error appending candles: {e}")


def run_aggregation() -> None:
    """
    Main aggregation loop - refreshes every 0.5 seconds.
    """
    global running, last_processed_timestamp
    
    logger.info("Starting live trade aggregation...")
    logger.info(f"Input: {INPUT_CSV}")
    logger.info(f"Output: {OUTPUT_CSV}")
    logger.info(f"Refresh interval: {REFRESH_INTERVAL}s")
    logger.info("Press Ctrl+C to stop")
    
    while running:
        try:
            # Load new trades
            new_trades = load_new_trades(last_processed_timestamp)
            
            if not new_trades.empty:
                # Update last processed timestamp
                last_processed_timestamp = new_trades['time'].max()
                
                # Aggregate into candles
                candles = aggregate_minute(new_trades)
                
                # Append to output
                append_candles(candles, OUTPUT_CSV)
            
            # Wait before next refresh
            time.sleep(REFRESH_INTERVAL)
            
        except KeyboardInterrupt:
            logger.info("Received interrupt signal, shutting down...")
            running = False
            break
            
        except Exception as e:
            logger.error(f"Error in aggregation loop: {e}")
            time.sleep(REFRESH_INTERVAL)
    
    logger.info("Aggregation stopped")


def signal_handler(sig, frame):
    """Handle graceful shutdown on SIGINT/SIGTERM."""
    global running
    logger.info("Shutdown signal received...")
    running = False


def main():
    """Main execution function."""
    global running
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        run_aggregation()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        running = False
    finally:
        logger.info("Cleanup complete")


if __name__ == "__main__":
    main()
