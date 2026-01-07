"""
Binance Order Book Depth Collector
Connects to Binance WebSocket to stream top-20 order book depth,
computes high-frequency features, and aggregates them per minute.
"""

import websocket
import json
import time
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timezone

import yaml

# --- Configuration ---
def load_config():
    try:
        with open('config.yaml', 'r') as f:
            return yaml.safe_load(f)
    except:
        return {}

config = load_config()
SYMBOL = config.get('params', {}).get('symbol', 'BTCUSDT').lower()
WS_URL = f"wss://stream.binance.com:9443/ws/{SYMBOL}@depth20@100ms"
LOG_FILE = "logs/orderbook_depth.log"
CSV_FILE = "orderbook_depth.csv"
BPS_THRESHOLD = 0.0010  # 10 basis points (0.10%)

# --- Logging ---
if not os.path.exists("logs"):
    os.makedirs("logs")

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)
# Set websocket logger to error only to avoid bloat
logging.getLogger('websocket').setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

# --- Global Buffer ---
# Buffer to hold snapshot features for the current minute
snapshot_buffer = []


def compute_depth_features(data):
    """
    Compute features from a depth snapshot.
    data format: {'bids': [[price, qty], ...], 'asks': [[price, qty], ...]}
    """
    try:
        bids = np.array(data['bids'], dtype=float)
        asks = np.array(data['asks'], dtype=float)

        if len(bids) == 0 or len(asks) == 0:
            return None

        best_bid = bids[0, 0]
        best_ask = asks[0, 0]
        
        # 1. Price & Spread
        mid_price = (best_bid + best_ask) / 2
        spread = best_ask - best_bid
        
        # 2. Total Depth (Top 20)
        bid_depth_20 = np.sum(bids[:, 1])
        ask_depth_20 = np.sum(asks[:, 1])
        
        # 3. Imbalance
        total_depth = bid_depth_20 + ask_depth_20
        imbalance = (bid_depth_20 - ask_depth_20) / total_depth if total_depth > 0 else 0

        # 4. Largest Wall Distance
        max_bid_idx = np.argmax(bids[:, 1])
        max_ask_idx = np.argmax(asks[:, 1])
        
        max_bid_qty = bids[max_bid_idx, 1]
        max_ask_qty = asks[max_ask_idx, 1]
        
        if max_bid_qty > max_ask_qty:
            largest_wall_dist = mid_price - bids[max_bid_idx, 0]
        else:
            largest_wall_dist = asks[max_ask_idx, 0] - mid_price

        # 5. Cumulative Depth within X bps
        lower_bound = mid_price * (1 - BPS_THRESHOLD)
        upper_bound = mid_price * (1 + BPS_THRESHOLD)
        
        valid_bids = bids[bids[:, 0] >= lower_bound]
        valid_asks = asks[asks[:, 0] <= upper_bound]
        
        cum_depth_xbps = np.sum(valid_bids[:, 1]) + np.sum(valid_asks[:, 1])

        return {
            'timestamp': time.time(), # Processing time
            'mid_price': mid_price,
            'spread': spread,
            'bid_depth_20': bid_depth_20,
            'ask_depth_20': ask_depth_20,
            'imbalance': imbalance,
            'largest_wall_dist': largest_wall_dist,
            'cum_depth_10bps': cum_depth_xbps
        }

    except Exception as e:
        logger.error(f"Feature calc error: {e}")
        return None


def aggregate_depth_minute(buffer):
    """
    Aggregate minute buffer into a single row.
    """
    if not buffer:
        return None
    
    df = pd.DataFrame(buffer)
    agg = df.mean().to_dict()
    
    last_ts = df['timestamp'].iloc[-1]
    dt = datetime.fromtimestamp(last_ts, timezone.utc)
    time_open = dt.replace(second=0, microsecond=0)
    
    agg['timeOpen'] = time_open
    agg.pop('timestamp', None)
    
    return agg


def append_to_csv(row_dict):
    """
    Append aggregated row to CSV.
    """
    try:
        df = pd.DataFrame([row_dict])
        cols = ['timeOpen', 'mid_price', 'spread', 'bid_depth_20', 'ask_depth_20', 
                'imbalance', 'largest_wall_dist', 'cum_depth_10bps']
        
        for c in cols:
            if c not in df.columns:
                df[c] = 0.0
                
        df = df[cols]
        header = not os.path.exists(CSV_FILE)
        df.to_csv(CSV_FILE, mode='a', header=header, index=False)
        logger.info(f"Aggregated depth for {row_dict['timeOpen']}")
        
    except Exception as e:
        logger.error(f"CSV write error: {e}")

def init_csv():
    """Ensure CSV file exists with proper headers."""
    if not os.path.exists(CSV_FILE):
        cols = ['timeOpen', 'mid_price', 'spread', 'bid_depth_20', 'ask_depth_20', 
                'imbalance', 'largest_wall_dist', 'cum_depth_10bps']
        pd.DataFrame(columns=cols).to_csv(CSV_FILE, index=False)
        logger.info(f"Initialized {CSV_FILE}")


def on_message(ws, message):
    global snapshot_buffer
    
    try:
        data = json.loads(message)
        features = compute_depth_features(data)
        
        if features:
            current_ts = features['timestamp']
            current_min = int(current_ts // 60)
            
            # 1. Handle Aggregation Rollover
            if snapshot_buffer:
                last_ts = snapshot_buffer[-1]['timestamp']
                last_min = int(last_ts // 60)
                
                if current_min > last_min:
                    agg_row = aggregate_depth_minute(snapshot_buffer)
                    if agg_row:
                        append_to_csv(agg_row)
                    snapshot_buffer = []
            
            snapshot_buffer.append(features)

            # 2. Save Real-time Snapshot for Dashboard
            snapshot = {
                'timestamp': current_ts,
                'bids': data['bids'][:15], # Top 15 levels
                'asks': data['asks'][:15], # Top 15 levels
                'mid_price': features['mid_price']
            }
            try:
                # Use a temporary file for atomic write to avoid dashboard read errors
                with open("latest_orderbook.json.tmp", "w") as f:
                    json.dump(snapshot, f)
                os.replace("latest_orderbook.json.tmp", "latest_orderbook.json")
            except: pass
            
    except Exception as e:
        logger.error(f"Error processing message: {e}")

def on_error(ws, error):
    logger.error(f"WebSocket Error: {error}")

def on_close(ws, close_status_code, close_msg):
    logger.info("WebSocket Closed")

def on_open(ws):
    logger.info(f"Connected to {WS_URL}")

def collect_depth_stream():
    while True:
        try:
            ws = websocket.WebSocketApp(
                WS_URL,
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close
            )
            ws.run_forever()
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            time.sleep(5)

if __name__ == "__main__":
    logger.info("Starting Order Book Depth Collector...")
    init_csv()
    collect_depth_stream()
