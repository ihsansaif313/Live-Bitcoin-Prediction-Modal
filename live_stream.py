"""
Bitcoin Live Trade Streaming Script
Streams live BTC/USDT trades from Binance WebSocket and saves to CSV.
"""

import websocket
import json
import pandas as pd
import logging
import time
import threading
from datetime import datetime, timezone
from typing import List, Dict
import os
import signal
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
WEBSOCKET_URLS = [
    "wss://stream.binance.com:9443/ws/btcusdt@trade",
    "wss://stream.binance.us:9443/ws/btcusdt@trade" # Fallback for US-hosted servers
]
OUTPUT_CSV = "btc_trades_live.csv"
BATCH_INTERVAL = 2  # seconds
MAX_RECONNECT_DELAY = 60  # seconds
INITIAL_RECONNECT_DELAY = 1  # seconds

# Global state
trade_buffer = []
buffer_lock = threading.Lock()
running = True
trade_count = 0
last_log_time = time.time()


def on_message(ws, message: str) -> Dict:
    """
    Parse incoming WebSocket trade message.
    
    Args:
        ws: WebSocket connection
        message: Raw JSON message string
    
    Returns:
        Normalized trade dictionary
    """
    global trade_count, last_log_time
    
    try:
        msg = json.loads(message)
        
        # Binance trade stream format:
        # {
        #   "e": "trade",
        #   "E": event_time,
        #   "s": "BTCUSDT",
        #   "t": trade_id,
        #   "p": price,
        #   "q": quantity,
        #   "b": buyer_order_id,
        #   "a": seller_order_id,
        #   "T": trade_time,
        #   "m": is_buyer_maker,
        #   "M": ignore
        # }
        
        trade = {
            'tradeId': int(msg['t']),
            'price': float(msg['p']),
            'qty': float(msg['q']),
            'quoteQty': float(msg['p']) * float(msg['q']),  # price * qty
            'time': pd.to_datetime(msg['T'], unit='ms', utc=True),
            'isBuyerMaker': bool(msg['m'])
        }
        
        # Add to buffer
        with buffer_lock:
            trade_buffer.append(trade)
        
        trade_count += 1
        
        # Log stats every minute
        current_time = time.time()
        if current_time - last_log_time >= 60:
            logger.info(f"Trades received in last minute: {trade_count}")
            trade_count = 0
            last_log_time = current_time
        
        return trade
        
    except Exception as e:
        logger.error(f"Error parsing message: {e}")
        return {}


def on_error(ws, error):
    """Handle WebSocket errors."""
    logger.error(f"WebSocket error: {error}")


def on_close(ws, close_status_code, close_msg):
    """Handle WebSocket close."""
    logger.warning(f"WebSocket closed: {close_status_code} - {close_msg}")


def on_open(ws):
    """Handle WebSocket open."""
    logger.info("WebSocket connection established")


def append_trades_to_csv(rows: List[Dict], path: str) -> None:
    """
    Append trades to CSV file in batch.
    
    Args:
        rows: List of trade dictionaries
        path: Output CSV file path
    """
    if not rows:
        return
    
    try:
        df = pd.DataFrame(rows)
        
        # Check if file exists to determine if we need headers
        file_exists = os.path.isfile(path)
        
        # Append to CSV
        df.to_csv(
            path,
            mode='a',
            header=not file_exists,
            index=False
        )
        
        logger.debug(f"Appended {len(rows)} trades to {path}")
        
    except Exception as e:
        logger.error(f"Error appending to CSV: {e}")
        print(f"CRITICAL ERROR WRITING CSV: {e}", file=sys.stderr)


def batch_writer():
    """
    Background thread that periodically flushes trade buffer to CSV.
    """
    global running
    
    while running:
        time.sleep(BATCH_INTERVAL)
        
        # Get trades from buffer
        with buffer_lock:
            if trade_buffer:
                trades_to_write = trade_buffer.copy()
                trade_buffer.clear()
            else:
                trades_to_write = []
        
        # Write to CSV
        if trades_to_write:
            append_trades_to_csv(trades_to_write, OUTPUT_CSV)


def run_stream() -> None:
    """
    Run the WebSocket stream with auto-reconnect and exponential backoff.
    """
    global running
    
    reconnect_delay = INITIAL_RECONNECT_DELAY
    reconnect_count = 0
    
    # Start batch writer thread
    writer_thread = threading.Thread(target=batch_writer, daemon=True)
    writer_thread.start()
    logger.info("Batch writer thread started")
    
    while running:
        # Rotate between global and US streams
        current_url = WEBSOCKET_URLS[reconnect_count % len(WEBSOCKET_URLS)]
        try:
            logger.info(f"Connecting to {current_url}...")
            
            import ssl
            import certifi
            
            # Create WebSocket connection with SSL context
            ws = websocket.WebSocketApp(
                current_url,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
                on_open=on_open
            )
            
            # Run WebSocket (blocking) with SSL options
            ws.run_forever(sslopt={"ca_certs": certifi.where()}, ping_interval=30, ping_timeout=10)
            
            # If we get here, connection was closed
            if running:
                reconnect_count += 1
                logger.warning(f"Connection lost or Blocked at {current_url}. Trying next in {reconnect_delay}s...")
                time.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, MAX_RECONNECT_DELAY)
            
        except KeyboardInterrupt:
            logger.info("Received interrupt signal, shutting down...")
            running = False
            break
            
        except Exception as e:
            logger.error(f"Error in stream: {e}")
            if running:
                reconnect_count += 1
                logger.warning(f"Reconnecting in {reconnect_delay}s (attempt {reconnect_count})...")
                time.sleep(reconnect_delay)
                
                # Exponential backoff
                reconnect_delay = min(reconnect_delay * 2, MAX_RECONNECT_DELAY)
    
    # Flush remaining trades
    logger.info("Flushing remaining trades to CSV...")
    with buffer_lock:
        if trade_buffer:
            append_trades_to_csv(trade_buffer, OUTPUT_CSV)
            trade_buffer.clear()
    
    logger.info("Stream stopped")


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
    
    logger.info("Starting Bitcoin live trade streaming...")
    logger.info(f"Output file: {OUTPUT_CSV}")
    logger.info(f"Batch interval: {BATCH_INTERVAL}s")
    logger.info("Press Ctrl+C to stop")
    
    # Ensure flush immediately
    if not os.path.exists(OUTPUT_CSV):
        try:
             with open(OUTPUT_CSV, 'w') as f:
                 f.write("tradeId,price,qty,quoteQty,time,isBuyerMaker\n")
             logger.info(f"Initialized {OUTPUT_CSV}")
        except Exception as e:
             logger.error(f"Failed to initialize CSV: {e}")

    try:
        run_stream()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        running = False
    finally:
        logger.info("Cleanup complete")


if __name__ == "__main__":
    main()
