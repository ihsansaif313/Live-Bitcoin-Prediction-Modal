import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import os
import json
import threading
import sys
import subprocess
import yaml
import pickle
import numpy as np
from datetime import datetime, timezone, timedelta
import prediction_tracker as tracker

# Robust Streamlit Internal Imports
try:
    from streamlit.runtime.scriptrunner import add_script_run_context
except ImportError:
    try:
        from streamlit.runtime.scriptrunner_utils.script_run_context import add_script_run_context
    except ImportError:
        try:
            from streamlit.runtime.scriptrunner.script_run_context import add_script_run_context
        except ImportError:
            def add_script_run_context(thread): pass # Fallback

# Lazy Torch Imports
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# --- 1. Configuration & Setup ---
st.set_page_config(
    page_title="BTC/USDT | Binance Pro",
    page_icon="💸",
    layout="wide"
)

# Constants
LIVE_CANDLES_CSV = "btc_live_candles.csv"
LIVE_TRADES_CSV = "btc_trades_live.csv"
MODELS_DIR = "models"
SCALERS_FILE = "scalers.pkl"

# --- 1.5. PyTorch Model Definitions ---
if TORCH_AVAILABLE:
    class LSTMModel(nn.Module):
        def __init__(self, input_dim, hidden_dim=128, num_layers=3, output_dim=1, task='regression'):
            super(LSTMModel, self).__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.3)
            self.fc = nn.Linear(hidden_dim, output_dim)
            self.task = task

        def forward(self, x):
            out, _ = self.lstm(x)
            out = self.fc(out[:, -1, :])
            if self.task == 'classification':
                out = torch.sigmoid(out)
            return out

    class TransformerModel(nn.Module):
        def __init__(self, input_dim, d_model=128, nhead=8, num_layers=3, output_dim=1, task='regression'):
            super(TransformerModel, self).__init__()
            self.embedding = nn.Linear(input_dim, d_model)
            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=256, batch_first=True)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.fc = nn.Linear(d_model, output_dim)
            self.task = task

        def forward(self, x):
            x = self.embedding(x)
            x = self.transformer_encoder(x)
            x = self.fc(x[:, -1, :])
            if self.task == 'classification':
                x = torch.sigmoid(x)
            return x
else:
    LSTMModel = None
    TransformerModel = None

# --- 2. Orchestration & Setup Logic ---
def load_config():
    if os.path.exists("config.yaml"):
        with open("config.yaml", "r") as f:
            return yaml.safe_load(f)
    return {}

def get_python_executable():
    """Get the path to the python executable, preferring venv if it exists."""
    if os.name == 'nt': # Windows
        venv_python = os.path.join(os.getcwd(), "venv", "Scripts", "python.exe")
    else: # Unix/Linux
        venv_python = os.path.join(os.getcwd(), "venv", "bin", "python")
    
    if os.path.exists(venv_python):
        return venv_python
    return sys.executable

def run_step(script_name: str, description: str):
    log_file = "logs/setup.log"
    if not os.path.exists("logs"): os.makedirs("logs")
    python_exe = get_python_executable()
    try:
        with open(log_file, "a") as f:
            f.write(f"\n>>> Starting: {description} ({script_name}) at {time.ctime()}\n")
            f.flush()
            subprocess.check_call([python_exe, script_name], stdout=f, stderr=subprocess.STDOUT)
            f.write(f"\n<<< Completed: {description} successfully.\n")
    except Exception as e:
        with open(log_file, "a") as f:
            f.write(f"\nERROR in {description}: {str(e)}\n")
        raise RuntimeError(f"Step {description} failed: {e}")

def update_status(status, progress, message, detail=""):
    status_file = "logs/setup_status.json"
    if not os.path.exists("logs"): os.makedirs("logs")
    try:
        with open(status_file, "w") as f:
            json.dump({"status": status, "progress": progress, "message": message, "detail": detail}, f)
    except: pass

def cleanup_data(config):
    paths = config.get('paths', {})
    files = [
        paths.get('historical_data', 'btc_historical.csv'),
        paths.get('historical_data_clean', 'btc_historical_clean.csv'),
        paths.get('dataset', 'btc_dataset.csv'),
        paths.get('live_candles', 'btc_live_candles.csv'),
        "btc_trades_live.csv", "btc_features.csv", "btc_features_normalized.csv",
        "sentiment_events.csv", "sentiment_minute.csv", "orderbook_depth.csv", "macro_factors.csv",
        "prediction_history.csv", "signals_v2.csv", "scalers.pkl", "logs/setup.log", "logs/data_cleaning.log",
        "logs/continuous_learning.log", "logs/orderbook_depth.log", "logs/macro_factors.log"
    ]
    
    # Force Kill background services to release file locks
    try:
        current_pid = os.getpid()
        scripts = [
            "live_stream.py", "orderbook_depth.py", "macro_factors.py", 
            "aggregate_live_to_candles.py", "continuous_learning.py", 
            "sentiment_ingest.py", "train_models.py", "build_dataset.py", "historical_data.py"
        ]
        print(f"Stopping background services (Excluding current PID {current_pid})...")
        for script in scripts:
            try:
                # 1. Try PowerShell with ForEach-Object for reliability
                ps_cmd = f"Get-CimInstance Win32_Process | Where-Object {{ ($_.CommandLine -like '*{script}*') -and ($_.ProcessId -ne {current_pid}) }} | ForEach-Object {{ Stop-Process -Id $_.ProcessId -Force }}"
                subprocess.run(["powershell", "-Command", ps_cmd], capture_output=True)
            except: pass
        
        # 2. Aggressive wait to allow Windows to release handles
        time.sleep(5.0) 
    except: pass
    
    for f in files:
        if os.path.exists(f): 
            for attempt in range(5): # Increased retries
                try: 
                    os.remove(f)
                    print(f"Deleted {f}")
                    break
                except Exception as e:
                    if attempt == 4: # Last attempt
                        print(f"Failed to delete {f}: {e}")
                        # FALLBACK: Try to truncate if we can't delete (keeps system running)
                        try:
                            with open(f, 'w') as f_truncate:
                                f_truncate.write("")
                            print(f"Truncated locked file: {f}")
                        except: pass
                    else:
                        time.sleep(0.5 * (attempt + 1)) # Exponential backoff
    
    models_dir = paths.get('models_dir', 'models')
    if os.path.exists(models_dir):
        try:
            import shutil
            shutil.rmtree(models_dir)
        except: pass

    # Clean all log files
    if os.path.exists("logs"):
        for f in os.listdir("logs"):
            if f.endswith(".log"):
                file_path = os.path.join("logs", f)
                for attempt in range(5):
                    try:
                        os.remove(file_path)
                        print(f"Deleted log: {f}")
                        break
                    except:
                        if attempt == 4:
                            try:
                                with open(file_path, 'w') as f_truncate:
                                    f_truncate.write("")
                                print(f"Truncated locked log: {f}")
                            except: pass
                        else:
                            time.sleep(0.5 * (attempt + 1))

@st.cache_resource
def get_server_session_state():
    return {
        "fresh_start_done": False, 
        "services_started": False,
        "setup_running": False,
        "setup_complete": False
    }

def run_setup_sequence(config, server_state):
    lock_file = "logs/setup.lock"
    if not os.path.exists("logs"): os.makedirs("logs")
    
    # 1. Singleton Lock: Prevent multiple setup threads from running simultaneously
    # Checked FIRST before any cleanup or file opening
    if os.path.exists(lock_file):
        try:
            with open(lock_file, 'r') as f:
                locked_pid = int(f.read().strip())
            
            # Check if process is still running (Windows compatible)
            is_running = False
            try:
                cmd = f'tasklist /FI "PID eq {locked_pid}" /NH'
                output = subprocess.check_output(cmd, shell=True).decode()
                if str(locked_pid) in output:
                    is_running = True
            except: pass

            if is_running and locked_pid != os.getpid():
                print(f"Setup already in progress (PID {locked_pid}). Skipping redundant setup thread.")
                return 
            else:
                # Stale lock (process dead)
                print(f"Removing stale lock from PID {locked_pid}")
                try: os.remove(lock_file)
                except: pass
        except Exception:
            try: os.remove(lock_file)
            except: pass

    try:
        with open(lock_file, "w") as f:
            f.write(str(os.getpid()))

        # 2. MARK AS RUNNING GLOBALLY (Double check)
        server_state["setup_running"] = True
        server_state["setup_complete"] = False

        # 3. CRITICAL: Run cleanup AFTER lock but BEFORE opening setup log files
        if config.get('params', {}).get('fresh_start', False):
            update_status("running", 0.01, "Initializing Fresh Start", "Cleaning up old data...")
            # Reset flags to ensure fresh services are started
            server_state["services_started"] = False
            cleanup_data(config)
            try: 
                tracker.init_tracker(overwrite=True)
                init_signals_v2(overwrite=True)
            except: pass

        log_file = "logs/setup.log"
        with open(log_file, "w") as f:
            f.write(f"--- Initialization Started: {time.ctime()} ---\n")

        # 4. Setup Logic
        python_exe = get_python_executable()

        # 2. Historical Data
        dataset_path = config.get('paths', {}).get('dataset', 'btc_dataset.csv')
        hist_path = config.get('paths', {}).get('historical_data', 'btc_historical.csv')
        symbol = config.get('params', {}).get('symbol', 'BTCUSDT')

        if not os.path.exists(dataset_path):
            lookback_days = config.get('params', {}).get('lookback_days', 365)
            update_status("running", 0.1, "Downloading Historical Data", f"Fetching {lookback_days} days of OHLC data...")
            
            def hist_progress(pct, candle):
                # Map 0-100% of download to 0.1-0.2 of total setup progress
                mapped_progress = 0.1 + (pct/100.0 * 0.1)
                update_status("running", mapped_progress, "Downloading Historical Data", f"Progress: {pct:.1f}% | Last: {candle}")

            try:
                from historical_data import collect_historical_data, save_csv
                df = collect_historical_data(symbol=symbol, interval="1m", lookback_days=lookback_days, progress_callback=hist_progress)
                save_csv(df, hist_path)
            except Exception as e:
                run_step("historical_data.py", "Download Historical Data")
            
            update_status("running", 0.2, "Cleaning Data", "Detecting outliers and removing noise...")
            try:
                from data_cleaner import clean_historical_data
                clean_historical_data(config.get('paths', {}).get('historical_data'), config.get('paths', {}).get('historical_data_clean'))
            except:
                run_step("data_cleaner.py", "Clean Data")
            
            update_status("running", 0.3, "Building Dataset", "Generating technical indicators...")
            run_step("build_dataset.py", "Build Initial Dataset")
        
        # 3. Training
        models_dir = config.get('paths', {}).get('models_dir', 'models')
        should_train = config.get('params', {}).get('fresh_start', False) or \
                       not os.path.exists(models_dir) or \
                       not any(f.endswith('.pth') or f.endswith('.pkl') for f in os.listdir(models_dir))
        
        if should_train:
            update_status("running", 0.4, "Training Models", "Initializing training process...")
            run_step("train_models.py", "Train Initial Models")
            
        # 6. Start Background Services (ONLY AFTER training is complete)
        if not server_state.get("services_started", False):
            if os.environ.get("ORCHESTRATOR_RUNNING") != "1":
                update_status("running", 0.9, "Starting Live Engines", "Launching background streaming and analysis services...")
                
                scripts = ["live_stream.py", "orderbook_depth.py", "macro_factors.py", 
                           "aggregate_live_to_candles.py", "continuous_learning.py", "sentiment_ingest.py"]

                for script in scripts:
                    log_name = script.replace(".py", ".log")
                    with open(f"logs/{log_name}", "a") as log_f:
                        subprocess.Popen([python_exe, script], stdout=log_f, stderr=subprocess.STDOUT, env=os.environ.copy())
                
                server_state["services_started"] = True
            else:
                update_status("running", 0.9, "Orchestrator Mode", "Waiting for services to sync...")
        
        update_status("complete", 1.0, "Setup Complete", "All models trained and services synchronized.")
        server_state["setup_complete"] = True

    except Exception as e:
        update_status("error", 0.0, "Setup Failed", str(e))
    finally:
        server_state["setup_running"] = False
        if os.path.exists(lock_file):
            try: os.remove(lock_file)
            except: pass

def start_background_services():
    config = load_config()
    is_fresh = config.get('params', {}).get('fresh_start', False)
    
    server_state = get_server_session_state()
    
    # If a setup is already running globally, don't start a second one
    if server_state.get("setup_running"):
        return None

    if is_fresh and server_state["fresh_start_done"]:
        is_fresh = False
        if 'params' in config:
            config['params']['fresh_start'] = False
    elif is_fresh:
        server_state["fresh_start_done"] = True
    
    if 'setup_thread_started' in st.session_state:
        return None
    
    if is_fresh:
        status = check_setup_status_simple()
        if status and status.get("status") == "complete":
            update_status("running", 0.01, "Initializing Fresh Start", "Preparing environment...")
    
    # SET FLAG BEFORE STARTING THREAD (PREVENTS RACE)
    server_state["setup_running"] = True
    
    setup_thread = threading.Thread(target=run_setup_sequence, args=(config, server_state))
    add_script_run_context(setup_thread) 
    setup_thread.start()
    st.session_state.setup_thread_started = True
    return setup_thread

def check_setup_status_simple():
    status_file = "logs/setup_status.json"
    if os.path.exists(status_file):
        try:
            with open(status_file, "r") as f:
                return json.load(f)
        except: return None
    return None

# --- 3. CSS & Styling ---
def inject_custom_css():
    st.html("""
        <style>
        .stApp { background-color: #161A25; color: #EAECEF; }
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}
        
        div[data-testid="stMetricValue"] {
            font-size: 28px !important;
            font-weight: 700 !important;
            font-family: 'IBM Plex Sans', sans-serif;
        }
        div[data-testid="stMetricLabel"] {
            font-size: 14px !important;
            color: #848E9C !important;
        }
        
        .binance-card {
            background-color: #1E2329;
            border-radius: 4px;
            padding: 15px;
            margin-bottom: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }
        
        .text-green { color: #0ECB81; }
        .text-red { color: #F6465D; }
        .text-primary { color: #EAECEF; }
        .text-secondary { color: #848E9C; }
        
        @media (max-width: 800px) {
            .hide-mobile { display: none; }
        }
        </style>
    """)

# --- 3. Background Data Orchestration (Zero-Latency) ---

GLOBAL_CACHE = {
    'price': 0.0, 'ticker': {}, 'df': pd.DataFrame(), 
    'trades': pd.DataFrame(), 'ob': {}, 'pred': None,
    'sentiment_events': pd.DataFrame(), 'sentiment_minute': pd.DataFrame(),
    'macro_factors': {}, 'accuracy': {}, 'signals': [], 'last_update': 0
}

GLOBAL_MODELS = {
    'Reg': None, 'Cls': None, 
    'LSTM_Reg': None, 'Trans_Reg': None, 'XGB_Reg': None,
    'scalers': None, 'last_load': 0
}

SIGNALS_V2_CSV = "signals_v2.csv"

def init_signals_v2(overwrite=False):
    if overwrite and os.path.exists(SIGNALS_V2_CSV):
        try: os.remove(SIGNALS_V2_CSV)
        except: pass
        
    if not os.path.exists(SIGNALS_V2_CSV):
        try:
            df = pd.DataFrame(columns=["timestamp", "price", "signal", "confidence", "reason", "spread", "imbalance", "sentiment", "macro"])
            df.to_csv(SIGNALS_V2_CSV, index=False)
        except: pass

def generate_signal_v2(price, pred_data, ob_data, macro_data, sent_data):
    """
    Implements Signal Engine V2 Rules.
    Rules:
    - UP: Confidence >= threshold AND spread < max AND imbalance > min AND no neg sentiment spike.
    - DOWN: Confidence <= threshold OR (neg sentiment spike AND macro risk_off).
    """
    config = load_config()
    engine_conf = config.get('signal_engine', {})
    conf_thresh = engine_conf.get('confidence_threshold', 0.65)
    max_spread = engine_conf.get('max_spread', 15.0)
    min_imb = engine_conf.get('min_imbalance', 0.15)
    
    p_up = pred_data.get('confidence', 0.5)
    direction = pred_data.get('direction', 'NEUTRAL')
    
    # Orderbook metrics
    bids = ob_data.get('bids', [])
    asks = ob_data.get('asks', [])
    spread = abs(float(asks[0][0]) - float(bids[0][0])) if bids and asks else 999.0
    bid_vol = sum([float(q) for p, q in bids[:10]]) if bids else 0
    ask_vol = sum([float(q) for p, q in asks[:10]]) if asks else 0
    imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol) if (bid_vol + ask_vol) > 0 else 0
    
    # Sentiment & Macro
    sent_spike = 0
    if not sent_data.empty:
        sent_spike = int(sent_data.iloc[-1].get('negative_spike_flag', 0))
    
    macro_regime = "NEUTRAL"
    spx_z = float(macro_data.get('z_spx', 0))
    dxy_z = float(macro_data.get('z_dxy', 0))
    if spx_z < -1.0 and dxy_z > 1.0: macro_regime = "RISK_OFF"
    
    signal = "NEUTRAL"
    reasons = []
    
    # DIRECTIONAL GUARD
    if direction == "NEUTRAL":
        return "WAIT (Neutral)", ["Market predicted as flat/noise"]

    # UP LOGIC
    if p_up >= conf_thresh and direction == "UP":
        if spread <= max_spread and imbalance >= min_imb and sent_spike == 0:
            signal = "BUY/UP"
            reasons.append("High Confidence + Liquidity Support")
        else:
            if spread > max_spread: 
                reasons.append("Spread too wide")
                signal = "WAIT (Liquidity)"
            elif imbalance < min_imb: 
                reasons.append("Low buy pressure")
                signal = "WAIT (Liquidity)"
            elif sent_spike: 
                reasons.append("Negative sentiment spike")
                signal = "WAIT (Sentiment)"
            else:
                signal = "WAIT (Weak UP)"
            
    # DOWN LOGIC
    elif (p_up >= conf_thresh or (sent_spike == 1 and macro_regime == "RISK_OFF")) and direction == "DOWN":
        signal = "SELL/DOWN"
        if p_up >= conf_thresh: reasons.append("High Down Confidence")
        if sent_spike and macro_regime == "RISK_OFF": reasons.append("Sentiment Spike + Macro Risk-Off")
        
    if signal != "NEUTRAL":
        # Log to file (with debouncing: only BUY/SELL or status change)
        try:
            # Check last signal in cache to avoid spamming "WAIT"
            last_signals = GLOBAL_CACHE.get('signals', [])
            should_log = False
            
            if signal in ["BUY/UP", "SELL/DOWN"]:
                should_log = True # Always log high-intent signals
            elif not last_signals or last_signals[0]['signal'] != signal:
                should_log = True # Log status changes (e.g. NEUTRAL -> WAIT)
            
            if should_log:
                row = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "price": price,
                    "signal": signal,
                    "confidence": p_up,
                    "reason": " & ".join(reasons),
                    "spread": spread,
                    "imbalance": imbalance,
                    "sentiment": sent_spike,
                    "macro": macro_regime
                }
                # Double check to avoid filling disk with "WAIT" logs
                # Only append if BUY/SELL or if its been 5 minutes since last log
                pd.DataFrame([row]).to_csv(SIGNALS_V2_CSV, mode='a', header=False, index=False)
                GLOBAL_CACHE['signals'] = ([row] + GLOBAL_CACHE['signals'])[:5]
        except: pass
    
    return signal, reasons

def generate_narrative(pred_data, ticker, ob_data, macro_data, sent_data):
    """
    Translates raw AI data into a professional 'Thought' narrative.
    """
    if not pred_data:
        import random
        statuses = [
            "📡 Establishing secure handshake with Binance WebSocket API...",
            "📊 Aggregating live order book depth and calculating spread liquidity...",
            "🧠 Loading ensemble models (LSTM, Transformer, XGBoost) into memory...",
            "⏲️ Synchronizing server time for high-precision latency correction...",
            "🔍 Scanning sentiment feeds for breaking news and volatility triggers..."
        ]
        return random.choice(statuses), "#848E9C"
    
    direction = pred_data.get('direction', 'NEUTRAL')
    conf = pred_data.get('confidence', 0.5)
    
    # Extract Macro
    spx_z = float(macro_data.get('z_spx', 0))
    dxy_z = float(macro_data.get('z_dxy', 0))
    
    # Extract OB
    imbalance = float(ob_data.get('imbalance', 0))
    
    # Extract Sentiment
    sent_spike = int(sent_data.iloc[-1].get('negative_spike_flag', 0)) if not sent_data.empty else 0
    
    narrative = ""
    color = "#848E9C" # Default Gray
    
    # 1. CRITICAL MACRO/SENTIMENT WARNINGS (Highest Priority)
    if sent_spike:
        narrative = "⚠️ NEGATIVE SENTIMENT SPIKE: AI detecting panic in news feeds; shifting to defensive stance."
        color = "#F6465D"
    elif spx_z < -1.5:
        narrative = f"📉 MACRO ALARM: S&P 500 crashing (z={spx_z:.1f}); high correlation with BTC sell-off."
        color = "#F6465D"
    elif dxy_z > 1.5:
        narrative = f"💵 DOLLAR SURGE: DXY Index spiking (z={dxy_z:.1f}); historically bearish for crypto assets."
        color = "#F6465D"
        
    # 2. DIRECTIONAL LOGIC
    elif direction == "UP" and conf > 0.65:
        if imbalance > 0.15:
            narrative = "🚀 BULLISH CONSENSUS: Strong architectural agreement + Large Whale Buy Wall detected."
        else:
            narrative = "📈 GROWTH MODE: Momentum building; models detecting consistent upward sequence patterns."
        color = "#0ECB81"
        
    elif direction == "DOWN" and conf > 0.65:
        narrative = "🆘 RISK WARNING: Multi-model consensus predicts downward trend; exit liquidity thinning."
        color = "#F6465D"
        
    elif direction == "NEUTRAL":
        config = load_config()
        thresh = config.get('params', {}).get('min_target_return', 0.001)
        narrative = f"⚖️ NOISE FILTER ACTIVE: Market moves < {thresh*100:.1f}%. AI waiting for high-intent signal."
        color = "#848E9C"
    
    # 3. LEAN STATUS (Lower Priority)
    else:
        if spx_z > 1.0:
            narrative = "🐂 MACRO SUPPORT: S&P 500 showing strength; lean bullish correlation in effect."
        elif imbalance < -0.15:
            narrative = "🐋 WHALE ACTIVITY: Large Sell Walls detected. Models leaning defensive."
        else:
            narrative = "🔍 ANALYZING: Sequence length 180m; internal ensemble processing market intent."
            
    return narrative, color

def data_orchestrator():
    """
    Optimized multi-threaded background engine.
    - Thread 1: High-frequency Ticker (0.2s)
    - Thread 2: Medium-frequency Data (2s)
    - Thread 3: Low-frequency AI Prediction (10s)
    """
    import requests
    
    def ticker_loop():
        global GLOBAL_CACHE
        while True:
            try:
                # Primary: Try Binance REST API with longer timeout
                try:
                    r = requests.get("https://api.binance.com/api/v3/ticker/24hr?symbol=BTCUSDT", timeout=5.0)
                    if r.status_code == 200:
                        data = r.json()
                        GLOBAL_CACHE['ticker'] = data
                        curr_p = float(data.get('lastPrice', 0))
                        if curr_p > 0: GLOBAL_CACHE['price'] = curr_p
                except Exception as e:
                    # Fallback: Try reading from live trades CSV populated by background service
                    if os.path.exists("btc_trades_live.csv"):
                        try:
                            df_t = pd.read_csv("btc_trades_live.csv", low_memory=False).tail(1)
                            if not df_t.empty:
                                curr_p = float(df_t.iloc[0]['price'])
                                GLOBAL_CACHE['price'] = curr_p
                                # Fake a ticker structure for UI compatibility
                                if 'ticker' not in GLOBAL_CACHE or not GLOBAL_CACHE['ticker']:
                                    GLOBAL_CACHE['ticker'] = {
                                        'lastPrice': str(curr_p),
                                        'priceChangePercent': '0.0' # Unknown from CSV
                                    }
                        except: pass
            except: pass
            time.sleep(2.0) # Slower poll to be nice to API and CPU

    def data_loop():
        global GLOBAL_CACHE
        while True:
            try:
                # Load CSVs only if needed or periodically
                if os.path.exists("btc_dataset.csv"):
                    try:
                        df = pd.read_csv("btc_dataset.csv", low_memory=False).tail(500)
                        df['timeOpen'] = pd.to_datetime(df['timeOpen'], utc=True)
                        
                        # Live Merge: Append any newer rows from btc_live_candles.csv
                        if os.path.exists(LIVE_CANDLES_CSV):
                            live_df = pd.read_csv(LIVE_CANDLES_CSV, low_memory=False)
                            live_df['timeOpen'] = pd.to_datetime(live_df['timeOpen'], utc=True)
                            
                            # Merge and deduplicate
                            df = pd.concat([df, live_df]).drop_duplicates('timeOpen', keep='last')
                            df = df.sort_values('timeOpen').tail(500)
                            
                        GLOBAL_CACHE['df'] = df
                    except: pass
                
                if os.path.exists(LIVE_TRADES_CSV):
                    try:
                        trades_df = pd.read_csv(LIVE_TRADES_CSV, low_memory=False).tail(100)
                        trades_df['time'] = pd.to_datetime(trades_df['time'], utc=True)
                        trades_df = trades_df.sort_values('time', ascending=False)
                        GLOBAL_CACHE['trades'] = trades_df
                    except: pass
                
                if os.path.exists("sentiment_events.csv"):
                    try:
                        sent_e = pd.read_csv("sentiment_events.csv", low_memory=False).tail(20)
                        sent_e['time'] = pd.to_datetime(sent_e['time'], utc=True)
                        GLOBAL_CACHE['sentiment_events'] = sent_e
                    except: pass

                if os.path.exists("sentiment_minute.csv"):
                    try:
                        sent_m = pd.read_csv("sentiment_minute.csv", low_memory=False).tail(60)
                        sent_m['timeOpen'] = pd.to_datetime(sent_m['timeOpen'], utc=True)
                        GLOBAL_CACHE['sentiment_minute'] = sent_m
                    except: pass
                
                if os.path.exists("macro_factors.csv"):
                    try:
                        m_df = pd.read_csv("macro_factors.csv", low_memory=False).tail(1)
                        if not m_df.empty:
                            GLOBAL_CACHE['macro_factors'] = m_df.iloc[0].to_dict()
                    except: pass
                
                if os.path.exists("latest_orderbook.json"):
                    try:
                        with open("latest_orderbook.json", "r") as f:
                            ob_data = json.load(f)
                        all_qs = [float(q) for p, q in ob_data.get('bids', [])] + [float(q) for p, q in ob_data.get('asks', [])]
                        if all_qs:
                            med = np.median(all_qs)
                            ob_data['whale_threshold'] = max(med * 5, 2.0)
                        GLOBAL_CACHE['ob'] = ob_data
                    except: pass
                    
                # Accuracy Metrics (every cycle)
                acc_metrics = tracker.get_accuracy_metrics()
                GLOBAL_CACHE['accuracy'] = acc_metrics
                
            except: pass
            time.sleep(2)

    def ai_loop():
        global GLOBAL_CACHE, GLOBAL_MODELS
        import logging
        
        while True:
            try:
                # 1. Model Maintenance (Reload every 15 min or if ensemble incomplete)
                now = time.time()
                needs_reload = False
                if not GLOBAL_MODELS.get('Reg'):
                    needs_reload = True
                elif (now - GLOBAL_MODELS['last_load'] > 900):
                    needs_reload = True
                # If we have base models but missing ensembles, retry every 60s during warmup
                elif not GLOBAL_MODELS.get('XGB_Reg') and (now - GLOBAL_MODELS['last_load'] > 60):
                    needs_reload = True
                
                if needs_reload:
                    m, s, success = load_models()
                    if success:
                        GLOBAL_MODELS.update(m)
                        GLOBAL_MODELS['scalers'] = s
                        GLOBAL_MODELS['last_load'] = now
                        logging.info(f"AI Loop: Successfully loaded {len(m)} models. (Ensemble check: {'OK' if m.get('XGB_Reg') else 'Ensemble Missing'})")
                
                # 2. Inference
                df = GLOBAL_CACHE['df']
                curr_p = GLOBAL_CACHE['price']
                
                # Check prerequisites and update status if missing
                missing = []
                if df.empty: missing.append("Historical Data")
                if curr_p <= 0: missing.append("Live Price")
                if not GLOBAL_MODELS['Reg']: missing.append("Trained Models")
                
                if not missing:
                    try:
                        # Use cached features if possible or optimized predict_live
                        # Passing 500 rows to ensure indicators have enough history for 180-window
                        pred = predict_live(df.tail(500), 
                                           {'Reg': GLOBAL_MODELS['Reg'], 'Cls': GLOBAL_MODELS['Cls'],
                                            'LSTM_Reg': GLOBAL_MODELS.get('LSTM_Reg'), 
                                            'Trans_Reg': GLOBAL_MODELS.get('Trans_Reg'),
                                            'XGB_Reg': GLOBAL_MODELS.get('XGB_Reg')}, 
                                           GLOBAL_MODELS['scalers'], 
                                           curr_p)
                        if pred:
                            GLOBAL_CACHE['pred'] = pred
                            GLOBAL_CACHE['last_error'] = None 
                            
                            # New: Log prediction for accuracy tracking
                            tracker.log_prediction(curr_p, pred['prediction'], pred['direction'])
                            
                            # 3. Signal Engine V2
                            sig, sig_reasons = generate_signal_v2(
                                curr_p, pred, GLOBAL_CACHE['ob'], 
                                GLOBAL_CACHE['macro_factors'], 
                                GLOBAL_CACHE['sentiment_minute']
                            )
                            
                            # 4. Generate Narrative (AI Thoughts)
                            narrative, narr_color = generate_narrative(
                                pred, curr_p, GLOBAL_CACHE['ob'], 
                                GLOBAL_CACHE['macro_factors'], 
                                GLOBAL_CACHE['sentiment_minute']
                            )
                            GLOBAL_CACHE['narrative'] = {'text': narrative, 'color': narr_color}
                        else:
                            # If pred is None, check for cached debug errors
                            err = GLOBAL_CACHE.get('last_error')
                            if err:
                                GLOBAL_CACHE['narrative'] = {'text': f"️⚠️ Engine stalled: {err}", 'color': '#F6465D'}
                            else:
                                GLOBAL_CACHE['narrative'] = {'text': "🔍 Analyzing: Processing feature vectors and ensemble consensus...", 'color': '#848E9C'}
                    except Exception as e:
                        logging.error(f"Inference Loop Error: {e}")
                        GLOBAL_CACHE['narrative'] = {'text': f"⚠️ Inference Error: {e}", 'color': '#F6465D'}
                        GLOBAL_CACHE['last_error'] = str(e)
                else:
                    # System is waiting for data, update status with specific details
                    wait_details = []
                    if "Historical Data" in missing: wait_details.append("Waiting for candles (min 180)")
                    if "Live Price" in missing: wait_details.append("Syncing Binance ticker")
                    if "Trained Models" in missing: wait_details.append("Loading AI weights")
                    
                    wait_msg = " | ".join(wait_details)
                    GLOBAL_CACHE['narrative'] = {'text': f"⏳ {wait_msg}", 'color': '#E0AA3E'}
                
                # 6. Outcomes update & UI Metric Sync
                if curr_p > 0:
                    tracker.update_outcomes(curr_p)
                    # Sync metrics from file to UI cache
                    GLOBAL_CACHE['accuracy'] = tracker.get_accuracy_metrics()
                    
            except: pass
            time.sleep(5)

    if 'orchestrator_threads_active' not in st.session_state:
        for target in [ticker_loop, data_loop, ai_loop]:
            t = threading.Thread(target=target, daemon=True)
            add_script_run_context(t)
            t.start()
        st.session_state.orchestrator_threads_active = True

# CSS Injected in main()

# --- 4. Data Logic ---

# Note: load_data_with_partial is deprecated in favor of data_orchestrator cache.
# We keep a dummy for backward compatibility if needed, but it's no longer used.
def load_data_with_partial():
    return GLOBAL_CACHE['df'], GLOBAL_CACHE['trades']

# --- 4.5 Model Loading & Prediction ---
def load_models():
    """Load trained models and scalers (supports .pkl, .h5, .keras, .pth)."""
    models = {}
    scalers = None
    success = False
    
    try:
        if os.path.exists(SCALERS_FILE):
            with open(SCALERS_FILE, 'rb') as f:
                scalers = pickle.load(f)
        
        def load_specific_model(type_name, filename_base):
            # 1. Try Keras (.keras then .h5)
            keras_path = os.path.join(MODELS_DIR, f"{filename_base}.keras")
            h5_path = os.path.join(MODELS_DIR, f"{filename_base}.h5")
            for dl_path in [keras_path, h5_path]:
                if os.path.exists(dl_path):
                    try:
                        import tensorflow as tf
                        model = tf.keras.models.load_model(dl_path, compile=False)
                        if model: return model
                    except: continue
            
            # 2. Try PyTorch (.pth)
            pth_path = os.path.join(MODELS_DIR, f"{filename_base}.pth")
            if os.path.exists(pth_path):
                try:
                    checkpoint = torch.load(pth_path, map_location=torch.device('cpu'), weights_only=True)
                    input_dim = checkpoint['input_dim']
                    task = checkpoint['task']
                    model = LSTMModel(input_dim, task=task) if checkpoint['model_type'] == 'LSTM' else TransformerModel(input_dim, task=task)
                    model.load_state_dict(checkpoint['state_dict'])
                    model.eval()
                    return model
                except: pass
            
            # 3. Try Pickle (.pkl)
            pkl_path = os.path.join(MODELS_DIR, f"{filename_base}.pkl")
            if os.path.exists(pkl_path):
                try:
                    with open(pkl_path, 'rb') as f: return pickle.load(f)
                except: pass
            return None

        models['Reg'] = load_specific_model("Regression", "btc_model_reg")
        models['Cls'] = load_specific_model("Classification", "btc_model_cls")
        
        # New: Ensemble Models
        models['LSTM_Reg'] = load_specific_model("Regression", "btc_lstm_reg")
        models['Trans_Reg'] = load_specific_model("Regression", "btc_trans_reg")
        models['XGB_Reg'] = load_specific_model("Regression", "btc_xgb_reg")
        
        if models['Reg'] and models['Cls'] and scalers: success = True
    except: pass
    return models, scalers, success

def predict_live(df, _models, _scalers, current_price):
    """Generate prediction for the next 15 minutes."""
    if df.empty or not _models or not _scalers: return None
    try:
        from build_dataset import make_features, merge_sentiment_features, load_sentiment_data
        from build_dataset import merge_orderbook_features, load_orderbook_data
        from build_dataset import merge_macro_features, load_macro_data
        
        # 1. Create technical features
        features_df = make_features(df.tail(200))
        if features_df.empty: return None
        
        # 2. Merge Multi-Modal Data from CACHE (not disk)
        sent_m_cache = GLOBAL_CACHE.get('sentiment_minute', pd.DataFrame())
        if not sent_m_cache.empty:
            features_df = merge_sentiment_features(features_df, sent_m_cache)
        else: # Fallback to disk if cache empty
            features_df = merge_sentiment_features(features_df, load_sentiment_data())
            
        ob_cache = GLOBAL_CACHE.get('ob', {})
        if ob_cache:
            ob_df = load_orderbook_data() 
            features_df = merge_orderbook_features(features_df, ob_df)
        else:
            features_df = merge_orderbook_features(features_df, load_orderbook_data())
            
        macro_cache = GLOBAL_CACHE.get('macro_factors', {})
        if macro_cache:
            macro_df = load_macro_data()
            features_df = merge_macro_features(features_df, macro_df)
        else:
            features_df = merge_macro_features(features_df, load_macro_data())
        
        # 3. Align and Finalize Feature Set
        # The models expect exactly the same columns in the same order as scalers.pkl
        scaler_features = sorted(list(_scalers.keys())) if isinstance(_scalers, dict) else []
        
        # If scalers are combined in a single object (unlikely here but for safety)
        if not scaler_features and hasattr(_scalers, 'feature_names_in_'):
            scaler_features = list(_scalers.feature_names_in_)
            
        if not scaler_features:
            # Fallback: Just use current features if scalers empty (should not happen)
            feature_cols = [c for c in features_df.columns if c != 'timeOpen']
        else:
            feature_cols = scaler_features
            # Fill missing columns with 0.0
            for col in feature_cols:
                if col not in features_df.columns:
                    features_df[col] = 0.0
        
        # Reorder to match training order exactly
        last_row = features_df.iloc[[-1]].copy()
        
        config = load_config()
        win_size = config.get('params', {}).get('window_size', 180)
        threshold = config.get('params', {}).get('min_target_return', 0.001)

        # Scale for Baselines (XGBoost)
        X_vec = []
        for col in feature_cols:
            val = last_row[col].values[0]
            if col in _scalers:
                try:
                    X_vec.append(_scalers[col].transform([[val]])[0][0])
                except:
                    X_vec.append(val)
            else:
                X_vec.append(val)
        X_vector = np.array(X_vec).reshape(1, -1)

        # Scale for Deep Learning (LSTM/Transformer)
        # We need normalized history of 'win_size'
        norm_tail = pd.DataFrame(index=features_df.index)
        for col in feature_cols:
            col_vals = features_df[col].values.reshape(-1, 1)
            if col in _scalers:
                try:
                    norm_tail[col] = _scalers[col].transform(col_vals).flatten()
                except:
                    norm_tail[col] = features_df[col]
            else:
                norm_tail[col] = features_df[col]
        
        # Final safety check for NaNs (especially in technical indicators at the start)
        norm_tail = norm_tail.fillna(0.0)
        
        # 1. Base Predictions
        pred_results = {}
        
        # LSTM
        if _models.get('LSTM_Reg'):
            if len(norm_tail) >= win_size:
                seq_data = norm_tail.iloc[-win_size:].values.astype(np.float32)
                X_seq = torch.from_numpy(seq_data).unsqueeze(0)
                try:
                    with torch.no_grad(): 
                        lstm_ret = float(_models['LSTM_Reg'](X_seq).numpy().flatten()[0])
                        pred_results['LSTM'] = lstm_ret
                except Exception as e:
                    logging.warning(f"LSTM inference error: {e}")

        # Transformer
        if _models.get('Trans_Reg'):
            if len(norm_tail) >= win_size:
                # Re-check seq if LSTM failed or skipped
                seq_data = norm_tail.iloc[-win_size:].values.astype(np.float32)
                X_seq = torch.from_numpy(seq_data).unsqueeze(0)
                try:
                    with torch.no_grad():
                        trans_ret = float(_models['Trans_Reg'](X_seq).numpy().flatten()[0])
                        pred_results['Trans'] = trans_ret
                except Exception as e:
                    logging.warning(f"Transformer inference error: {e}")
        
        # XGBoost
        if _models.get('XGB_Reg'):
             try:
                 xgb_ret = _models['XGB_Reg'].predict(X_vector)[0]
                 pred_results['XGB'] = xgb_ret
             except Exception as e:
                 logging.warning(f"XGBoost inference error: {e}")
        
        # Fallback/Legacy Reg Model
        if _models.get('Reg') and 'XGB' not in pred_results and 'LSTM' not in pred_results:
            try:
                # Reg model might be SKLearn or Torch
                if hasattr(_models['Reg'], 'predict'):
                    reg_ret = _models['Reg'].predict(X_vector)[0]
                else: # Assume Torch
                    X_seq = torch.from_numpy(norm_tail.iloc[-win_size:].values.astype(np.float32)).unsqueeze(0)
                    with torch.no_grad():
                        reg_ret = float(_models['Reg'](X_seq).numpy().flatten()[0])
                pred_results['BaseReg'] = reg_ret
            except: pass

        if not pred_results: 
            # Detailed error about why models failed
            fail_reason = "No models responded."
            if len(norm_tail) < win_size:
                fail_reason = f"Insufficient history: {len(norm_tail)}/{win_size} candles"
            else:
                # Check for NaNs in X_vector
                if np.isnan(X_vector).any():
                    fail_reason = "NaNs detected in features"
                else:
                    fail_reason = f"Models skipped (Feature Count: {len(feature_cols)})"
            
            GLOBAL_CACHE['last_error'] = fail_reason
            return None

        # 2. Consensus Logic (Robust Median-based)
        # Use median return to prevent outliers from flipping price against majority direction
        rets = list(pred_results.values())
        avg_ret = np.median(rets)
        
        # ⚠️ Safety Clamp: Keep predictions within realistic 15-min volatility (max +/- 1.5%)
        # This prevents the "BTC growing too much in 15 mins" artifact from outlier models
        max_v = config.get('params', {}).get('max_15m_volatility', 0.015)
        avg_ret = np.clip(avg_ret, -max_v, max_v)
        
        # Enforce consistency: Direction MUST match price target
        if avg_ret >= threshold:
            final_dir = "UP"
        elif avg_ret <= -threshold:
            final_dir = "DOWN"
        else:
            final_dir = "NEUTRAL"
            
        pred_price = float(current_price * (1 + avg_ret))
        
        # Track individual model directions for agreement stats
        dirs = ["UP" if r >= threshold else "DOWN" if r <= -threshold else "NEUTRAL" for r in rets]
        
        # Consensus Score: % of models agreeing with final_dir
        # If final_dir is UP, how many rets were actually UP?
        agreement = dirs.count(final_dir) / len(dirs) if len(dirs) > 0 else 1.0
        
        # Confidence: Combination of model confidence and consensus agreement
        # Use return magnitude vs threshold as a signal strength proxy
        strength = min(abs(avg_ret) / (threshold * 3), 1.0)
        final_confidence = (strength * 0.4) + (agreement * 0.6) # Weight agreement more heavily

        return {
            'prediction': pred_price,
            'direction': final_dir,
            'confidence': final_confidence,
            'time': last_row['timeOpen'].iloc[0],
            'ensemble': pred_results # For debugging/detail
        }
    except Exception as e:
        # Log error for debugging
        import logging
        err_msg = f"Prediction error: {e}"
        logging.error(err_msg)
        GLOBAL_CACHE['last_error'] = err_msg
        return None

# --- 5. UI Fragments ---

@st.fragment(run_every=0.5)
def render_header_ticker():
    cache = GLOBAL_CACHE
    price = cache.get('price', 0.0)
    data = cache.get('ticker', {})
    change_pct = float(data.get('priceChangePercent', 0.0))
    high = float(data.get('highPrice', 0.0))
    low = float(data.get('lowPrice', 0.0))
    vol = float(data.get('volume', 0.0))
    
    pred = cache.get('pred')
    pred_text = "Waiting..."
    pred_color = "text-secondary"
    confidence_html = ""
    
    if pred:
        p_up = pred.get('confidence', 0.5)
        direction = pred.get('direction', 'NEUTRAL')
        target = pred.get('prediction', price)
        ensemble = pred.get('ensemble', {})
        
        # Calculate agreement
        total_models = len(ensemble)
        if total_models > 0:
            dirs = ["UP" if r >= 0.001 else "DOWN" if r <= -0.001 else "NEUTRAL" for r in ensemble.values()]
            agreement_count = dirs.count(direction)
            models_list = ", ".join(ensemble.keys())
            ensemble_html = f'<div style="font-size: 9px; color: #848E9C; margin-top: 1px;">{agreement_count}/{total_models} Agree ({models_list})</div>'
        else:
            ensemble_html = ""

        config = load_config()
        conf_thresh = config.get('signal_engine', {}).get('confidence_threshold', 0.65)
        
        if p_up >= conf_thresh:
            if direction == "UP":
                direction_emoji = "UP 🟢"
                color = "text-green"
            elif direction == "DOWN":
                direction_emoji = "DOWN 🔴"
                color = "text-red"
            else: # NEUTRAL, but high confidence
                direction_emoji = "NEUTRAL ⚪"
                color = "text-secondary"
            pred_text = f"{direction_emoji} ${target:,.2f}"
            pred_color = color
            confidence_html = f'<div style="font-size: 11px; color: #848E9C; margin-top: 2px;">{p_up*100:.1f}% Confidence</div>' + ensemble_html
        elif p_up >= 0.55: # LEAN Status
            is_up = target > price
            direction_emoji = "LEAN UP 🟡" if is_up else "LEAN DOWN 🟡"
            pred_text = f"{direction_emoji} ${target:,.2f}"
            pred_color = "text-secondary"
            confidence_html = f'<div style="font-size: 11px; color: #848E9C; margin-top: 2px;">Lean {p_up*100:.0f}% Confidence</div>' + ensemble_html
        else:
            pred_text = "WAIT (Neutral)"
            pred_color = "text-secondary"
            # Show what we are waiting for / analyzing
            if p_up < 0.5:
                # If confidence is very low, it means models disagree or signal is noisy
                confidence_html = f'<div style="font-size: 11px; color: #848E9C; margin-top: 2px;">Noisy signal ({p_up*100:.0f}%)</div>' + ensemble_html
            else:
                confidence_html = f'<div style="font-size: 11px; color: #848E9C; margin-top: 2px;">Analyzing {p_up*100:.0f}% Signal...</div>' + ensemble_html
    else:
        # Check why pred is missing
        narr = cache.get('narrative', {}).get('text', '')
        if "Waiting" in narr:
            pred_text = "Syncing..."
            confidence_html = f'<div style="font-size: 10px; color: #E0AA3E; margin-top: 2px;">{narr.replace("⏳ ", "")}</div>'
        else:
            pred_text = "Calculating..."
            confidence_html = f'<div style="font-size: 10px; color: #848E9C; margin-top: 2px;">Processing ensemble...</div>'

    color_class = "text-green" if change_pct >= 0 else "text-red"
    
    narr = cache.get('narrative', {'text': 'System Initialized. Observing market variables...', 'color': '#848E9C'})
    
    st.html(f"""
<div class="binance-card" style="display: flex; justify-content: space-between; align-items: center; padding: 15px 30px; margin-bottom: 5px;">
    <div style="display: flex; align-items: center; gap: 20px;">
        <div>
            <div style="font-size: 11px; color: #848E9C; margin-bottom: 2px;">BTC / USDT</div>
            <div style="font-size: 18px; font-weight: 600; color: #EAECEF;">${price:,.2f} <span style="font-size: 12px; margin-left: 2px;" class="{color_class}">{change_pct:+.2f}%</span></div>
        </div>
        <div style="width: 1px; height: 35px; background: #2F3336;"></div>
        <div>
            <div style="font-size: 11px; color: #848E9C; margin-bottom: 2px;">24h High/Low</div>
            <div style="font-size: 13px; color: #EAECEF;">${high:,.0f} / ${low:,.0f}</div>
        </div>
        <div style="width: 1px; height: 35px; background: #2F3336;"></div>
        <div>
            <div style="font-size: 11px; color: #848E9C; margin-bottom: 2px;">24h Volume</div>
            <div style="font-size: 13px; color: #EAECEF;">{vol:,.0f} BTC</div>
        </div>
    </div>
    
    <div style="text-align: right; border-left: 1px solid #2B3139; padding-left: 30px;">
        <div style="font-size: 11px; color: #848E9C; margin-bottom: 2px;">AI Target Consensus</div>
        <div class="{pred_color}" style="font-size: 18px; font-weight: 600;">{pred_text}</div>
        {confidence_html}
    </div>
</div>
<div class="binance-card" style="padding: 10px 30px; margin-bottom: 20px; border-top: 1px solid #2F3336; background: rgba(30,32,38,0.4); display: flex; align-items: center; gap: 12px;">
    <div style="width: 8px; height: 8px; background: {narr['color']}; border-radius: 50%; box-shadow: 0 0 8px {narr['color']}CC; flex-shrink: 0;"></div>
    <div style="font-size: 13px; color: {narr['color']}; font-weight: 400; font-family: 'Inter', sans-serif;">
        {narr['text']}
    </div>
</div>
""")

@st.fragment(run_every=30)
def render_main_chart():
    """Embed official TradingView Advanced Chart Widget."""
    import streamlit.components.v1 as components
    
    # We wrap the widget in a styled div within the same iframe to ensure it's contained correctly
    chart_html = """
    <div style="background-color: #161A25; border: 1px solid #2B3139; border-radius: 4px; height: 600px; overflow: hidden; padding: 0;">
        <!-- TradingView Widget BEGIN -->
        <div class="tradingview-widget-container" style="height:600px;width:100%">
          <div class="tradingview-widget-container__widget" style="height:calc(100% - 32px);width:100%"></div>
          <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js" async>
          {
          "allow_symbol_change": true,
          "calendar": false,
          "details": false,
          "hide_side_toolbar": true,
          "hide_top_toolbar": false,
          "hide_legend": false,
          "hide_volume": false,
          "hotlist": false,
          "interval": "1",
          "locale": "en",
          "save_image": true,
          "style": "1",
          "symbol": "BINANCE:BTCUSDT",
          "theme": "dark",
          "timezone": "Etc/UTC",
          "backgroundColor": "#161A25",
          "gridColor": "rgba(242, 242, 242, 0.06)",
          "watchlist": [],
          "withdateranges": false,
          "compareSymbols": [],
          "studies": [],
          "autosize": true
        }
          </script>
        </div>
        <!-- TradingView Widget END -->
    </div>
    """
    components.html(chart_html, height=610)

@st.fragment(run_every=0.5)
def render_order_book():
    """Renders the top bids and asks from memory cache with Whale Walls."""
    cache = GLOBAL_CACHE
    ob_data = cache.get('ob', {})
    whale_thresh = ob_data.get('whale_threshold', 1000.0)
    
    # Calculate imbalance
    bids = ob_data.get('bids', [])
    asks = ob_data.get('asks', [])
    bid_vol = sum([float(q) for p, q in bids[:10]]) if bids else 0
    ask_vol = sum([float(q) for p, q in asks[:10]]) if asks else 0
    total_vol = bid_vol + ask_vol
    imbalance = (bid_vol / total_vol) if total_vol > 0 else 0.5
    
    content_html = '<div class="binance-card" style="height: 440px; overflow-y: hidden; display: flex; flex-direction: column; margin-bottom: 20px;">'
    content_html += '<div style="display: flex; justify-content: space-between; margin-bottom: 5px;">'
    content_html += '  <span class="text-secondary" style="font-weight:bold; font-size:12px;">Order Book</span>'
    content_html += f'  <span class="text-secondary" style="font-size:10px;">Walls > {whale_thresh:.1f} BTC</span>'
    content_html += '</div>'
    
    # Imbalance Bar
    content_html += f"""
<div style="width: 100%; height: 6px; background: #2B3139; border-radius: 3px; margin-bottom: 12px; display: flex; overflow: hidden;">
    <div style="width: {imbalance*100}%; background: #0ECB81; height: 100%;"></div>
    <div style="width: {(1-imbalance)*100}%; background: #F6465D; height: 100%;"></div>
</div>
"""
    
    # Header
    content_html += '<div style="display: flex; justify-content: space-between; font-size: 11px; color: #848E9C; margin-bottom: 5px;">'
    content_html += '<span>Price(USDT)</span><span>Amount(BTC)</span>'
    content_html += '</div>'

    if ob_data:
        # ASKS (Red)
        asks_v = asks[:7]
        asks_v.reverse()
        for p, q in asks_v:
            p, q = float(p), float(q)
            wall_tag = '<span style="color: #F6465D; font-size: 9px; margin-left: 5px;">WALL</span>' if q >= whale_thresh else ''
            content_html += f"""
<div style="display: flex; justify-content: space-between; font-family: monospace; font-size: 12px; margin-bottom: 2px;">
    <span style="color: #F6465D;">{p:,.2f}</span>
    <span class="text-primary">{q:.4f}{wall_tag}</span>
</div>
"""
        # MID PRICE
        mid = ob_data.get('mid_price', 0)
        content_html += f"""
<div style="padding: 8px 0; border-top: 1px solid #2B3139; border-bottom: 1px solid #2B3139; margin: 5px 0; text-align: center;">
    <h3 style="margin: 0; color: #EAECEF; font-size: 16px;">{mid:,.2f}</h3>
</div>
"""
        # BIDS (Green)
        bids_v = bids[:7]
        for p, q in bids_v:
            p, q = float(p), float(q)
            wall_tag = '<span style="color: #0ECB81; font-size: 9px; margin-left: 5px;">WALL</span>' if q >= whale_thresh else ''
            content_html += f"""
<div style="display: flex; justify-content: space-between; font-family: monospace; font-size: 12px; margin-bottom: 2px;">
    <span style="color: #0ECB81;">{p:,.2f}</span>
    <span class="text-primary">{q:.4f}{wall_tag}</span>
</div>
"""
    else:
        content_html += '<div class="text-secondary" style="text-align:center; padding: 20px;">Preparing Order Book...</div>'
    
    content_html += '</div>'
    st.html(content_html)

@st.fragment(run_every=5)
def render_macro_panel():
    """Renders Macro panel with SPX/DXY z-scores and regime flags."""
    cache = GLOBAL_CACHE
    macro = cache.get('macro_factors', {})
    
    if not macro:
        st.html('<div class="binance-card" style="text-align:center; color:#848E9C; font-size:12px;">Macro Factors Loading...</div>')
        return
        
    spx_z = float(macro.get('z_spx', 0))
    dxy_z = float(macro.get('z_dxy', 0))
    regime = "NEUTRAL"
    regime_color = "#EAECEF"
    
    # Simple Regime Logic
    if spx_z > 1.0 and dxy_z < -1.0:
        regime = "RISK-ON 🚀"
        regime_color = "#0ECB81"
    elif spx_z < -1.0 and dxy_z > 1.0:
        regime = "RISK-OFF 🛡️"
        regime_color = "#F6465D"
        
    content_html = f'<div class="binance-card">'
    content_html += '<div class="text-secondary" style="font-size:12px; font-weight:bold; margin-bottom:10px;">Macro Correlations</div>'
    
    content_html += f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">'
    content_html += f'  <div style="font-size: 18px; font-weight: bold; color: {regime_color};">{regime}</div>'
    content_html += '  <div class="text-secondary" style="font-size: 10px;">Z-Score (20d)</div>'
    content_html += '</div>'
    
    # Progress bars for Z-scores (-3 to +3 range)
    def z_to_pct(z): return max(0, min(100, (z + 3) / 6 * 100))
    
    spx_pct = z_to_pct(spx_z)
    dxy_pct = z_to_pct(dxy_z)
    
    content_html += f"""
<div style="margin-bottom: 10px;">
    <div style="display: flex; justify-content: space-between; font-size: 11px; margin-bottom: 3px;">
        <span class="text-primary">S&P 500</span>
        <span class="{"text-green" if spx_z > 0 else "text-red"}">{spx_z:+.2f}</span>
    </div>
    <div style="width: 100%; height: 4px; background: #2B3139; border-radius: 2px;">
        <div style="width: {spx_pct}%; background: #0ECB81; height: 100%; border-radius: 2px;"></div>
    </div>
</div>

<div style="margin-bottom: 5px;">
    <div style="display: flex; justify-content: space-between; font-size: 11px; margin-bottom: 3px;">
        <span class="text-primary">DXY (USD)</span>
        <span class="{"text-red" if dxy_z > 0 else "text-green"}">{dxy_z:+.2f}</span>
    </div>
    <div style="width: 100%; height: 4px; background: #2B3139; border-radius: 2px;">
        <div style="width: {dxy_pct}%; background: #F6465D; height: 100%; border-radius: 2px;"></div>
    </div>
</div>
"""
    content_html += '</div>'
    st.html(content_html)

@st.fragment(run_every=0.5)
def render_signals_panel():
    """Renders the Signal Engine V2 panel."""
    cache = GLOBAL_CACHE
    signals = cache.get('signals', [])
    
    content_html = '<div class="binance-card" style="margin-bottom: 20px;">'
    content_html += '<div class="text-secondary" style="font-size:12px; font-weight:bold; margin-bottom:10px;">Signal Engine V2</div>'
    
    if not signals:
        content_html += '<div style="text-align:center; padding:15px; color:#848E9C; font-size:12px;">Searching for entry signals...</div>'
    else:
        latest = signals[0]
        sig_color = "#0ECB81" if "BUY" in latest['signal'] else ("#F6465D" if "SELL" in latest['signal'] else "#848E9C")
        
        content_html += f"""
<div style="background: rgba(30, 35, 41, 0.5); padding: 12px; border-left: 4px solid {sig_color}; border-radius: 4px; margin-bottom: 15px;">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <span style="font-size: 20px; font-weight: bold; color: {sig_color};">{latest['signal']}</span>
        <span class="text-secondary" style="font-size: 11px;">{latest['timestamp'][11:19]}</span>
    </div>
    <div style="font-size: 12px; color: #EAECEF; margin-top: 5px; font-style: italic;">{latest['reason']}</div>
</div>
"""
        # History table (last 3)
        content_html += '<div style="font-size: 10px; color: #848E9C; margin-bottom: 5px; border-top: 1px solid #2B3139; padding-top: 10px;">Recent Signals</div>'
        for s in signals[1:4]:
            c = "#0ECB81" if "BUY" in s['signal'] else ("#F6465D" if "SELL" in s['signal'] else "#848E9C")
            content_html += f"""
<div style="display: flex; justify-content: space-between; font-size: 11px; margin-bottom: 4px;">
    <span style="color: {c}; font-weight: bold;">{s['signal'].split('/')[0]}</span>
    <span class="text-secondary">{s['timestamp'][11:16]}</span>
</div>
"""
            
    content_html += '</div>'
    st.html(content_html)

@st.fragment(run_every=0.5)
def render_recent_trades():
    """Renders latest trades from memory cache."""
    cache = GLOBAL_CACHE
    trades = cache.get('trades', pd.DataFrame())
    
    content_html = '<div class="binance-card" style="height: 220px; overflow-y: hidden;">'
    content_html += '<div class="text-secondary" style="font-size:12px; margin-bottom: 10px; font-weight:bold;">Recent Trades</div>'
    
    if not trades.empty:
        # Header
        content_html += '<div style="display: flex; justify-content: space-between; font-size: 10px; color: #848E9C; margin-bottom: 5px;">'
        content_html += '<span>Price</span><span>Qty</span><span>Time</span>'
        content_html += '</div>'
        
        for _, row in trades.head(8).iterrows():
            p_col = "#F6465D" if row['isBuyerMaker'] else "#0ECB81"
            try:
                t_val = row['time']
                if isinstance(t_val, str):
                    t_val = pd.to_datetime(t_val)
                time_str = t_val.strftime('%H:%M:%S')
            except:
                time_str = str(row.get('time', ''))[-8:] # Fallback for malformed data
                
            p_val = float(row.get('price', 0))
            q_val = float(row.get('qty', 0))
                
            content_html += f"""
<div style="display: flex; justify-content: space-between; font-family: monospace; font-size: 11px; margin-bottom: 2px;">
    <span style="color: {p_col};">{p_val:,.2f}</span>
    <span class="text-primary">{q_val:.4f}</span>
    <span class="text-secondary">{time_str}</span>
</div>
"""
    else:
        content_html += '<div class="text-secondary" style="text-align:center; padding: 10px;">Waiting for Trades...</div>'
    
    content_html += '</div>'
    st.html(content_html)

@st.fragment(run_every=5)
def render_sentiment_panel():
    """Renders beautiful sentiment analysis panel with news events and metrics."""
    cache = GLOBAL_CACHE
    sent_events = cache.get('sentiment_events', pd.DataFrame())
    sent_minute = cache.get('sentiment_minute', pd.DataFrame())
    
    # Calculate sentiment metrics
    avg_sentiment = 0.0
    sentiment_trend = "NEUTRAL"
    sentiment_color = "#848E9C"
    events_count = 0
    negative_spikes = 0
    
    if not sent_minute.empty:
        avg_sentiment = sent_minute['sentiment_mean'].mean()
        events_count = int(sent_minute['events_count'].sum())
        negative_spikes = int(sent_minute['negative_spike_flag'].sum())
        
        if avg_sentiment > 0.2:
            sentiment_trend = "BULLISH 🟢"
            sentiment_color = "#0ECB81"
        elif avg_sentiment < -0.2:
            sentiment_trend = "BEARISH 🔴"
            sentiment_color = "#F6465D"
        else:
            sentiment_trend = "NEUTRAL ⚪"
            sentiment_color = "#F0B90B"
    
    # Build HTML
    content_html = '<div class="binance-card" style="margin-bottom: 20px;">'
    
    # Header with gradient
    content_html += f'<div style="background: linear-gradient(135deg, #1E2329 0%, #2B3139 100%); padding: 15px; border-radius: 4px; margin-bottom: 15px;">'
    content_html += f'<div style="display: flex; justify-content: space-between; align-items: center;">'
    content_html += f'<div>'
    content_html += f'<div class="text-secondary" style="font-size: 12px; margin-bottom: 5px;">MARKET SENTIMENT</div>'
    content_html += f'<div style="font-size: 24px; font-weight: bold; color: {sentiment_color};">{sentiment_trend}  {avg_sentiment:.2f}</div>'
    content_html += f'</div>'
    content_html += f'<div style="text-align: right;">'
    content_html += f'<div class="text-secondary" style="font-size: 11px;">Last Hour</div>'
    content_html += f'<div class="text-primary" style="font-size: 16px; font-weight: bold;">{events_count} Events</div>'
    content_html += f'</div>'
    content_html += f'</div>'
    content_html += f'</div>'
    
    # Sentiment Metrics Grid
    if not sent_minute.empty:
        recent_sentiment = sent_minute.iloc[0]['sentiment_mean'] if len(sent_minute) > 0 else 0
        recent_neg = sent_minute.iloc[0]['sentiment_neg_mean'] if len(sent_minute) > 0 else 0
        recent_relevance = sent_minute.iloc[0]['relevance_score'] if len(sent_minute) > 0 else 0
        
        content_html += f'<div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; margin-bottom: 15px;">'
        
        content_html += f'<div style="background: #0B0E11; padding: 10px; border-radius: 4px; border-left: 3px solid #0ECB81;">'
        content_html += f'<div class="text-secondary" style="font-size: 10px;">CURRENT</div>'
        content_html += f'<div class="text-primary" style="font-size: 16px; font-weight: bold;">{recent_sentiment:.2f}</div>'
        content_html += f'</div>'
        
        content_html += f'<div style="background: #0B0E11; padding: 10px; border-radius: 4px; border-left: 3px solid #F6465D;">'
        content_html += f'<div class="text-secondary" style="font-size: 10px;">NEGATIVE</div>'
        content_html += f'<div class="text-primary" style="font-size: 16px; font-weight: bold;">{recent_neg:.2f}</div>'
        content_html += f'</div>'
        
        content_html += f'<div style="background: #0B0E11; padding: 10px; border-radius: 4px; border-left: 3px solid #F0B90B;">'
        content_html += f'<div class="text-secondary" style="font-size: 10px;">RELEVANCE</div>'
        content_html += f'<div class="text-primary" style="font-size: 16px; font-weight: bold;">{recent_relevance:.2f}</div>'
        content_html += f'</div>'
        
        content_html += f'</div>'

    # Alert for negative spikes
    if negative_spikes > 0:
        content_html += f'<div style="background: rgba(246, 70, 93, 0.1); border: 1px solid #F6465D; border-radius: 4px; padding: 10px; margin-bottom: 15px;">'
        content_html += f'<div style="color: #F6465D; font-size: 12px; font-weight: bold;">⚠️ {negative_spikes} Negative Spike(s) Detected</div>'
        content_html += f'<div class="text-secondary" style="font-size: 10px; margin-top: 3px;">High negative sentiment may indicate market fear</div>'
        content_html += f'</div>'
        
    # Recent News Events Section
    content_html += '<div class="text-secondary" style="font-size: 12px; font-weight: bold; margin-bottom: 10px; border-top: 1px solid #2B3139; padding-top: 15px;">📰 Recent News</div>'
    
    if not sent_events.empty:
        for idx, row in sent_events.head(5).iterrows():
            sentiment_val = row.get('sentiment_compound', 0)
            sent_color = "#0ECB81" if sentiment_val > 0 else "#F6465D" if sentiment_val < 0 else "#848E9C"
            sent_icon = "🟢" if sentiment_val > 0 else "🔴" if sentiment_val < 0 else "⚪"
            
            text = str(row.get('text', ''))[:120] + "..." if len(str(row.get('text', ''))) > 120 else str(row.get('text', ''))
            source = row.get('source', 'Unknown')
            
            try:
                t_val = row['time']
                if isinstance(t_val, str):
                    t_val = pd.to_datetime(t_val)
                time_str = t_val.strftime('%H:%M')
            except:
                time_str = str(row.get('time', ''))[:5] # Fallback for malformed data
            
            content_html += f'<div style="background: #0B0E11; padding: 10px; border-radius: 4px; margin-bottom: 8px; border-left: 3px solid {sent_color};">'
            content_html += f'<div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 5px;">'
            content_html += f'<span style="color: {sent_color}; font-size: 14px; font-weight: bold;">{sent_icon} {sentiment_val:+.2f}</span>'
            content_html += f'<span class="text-secondary" style="font-size: 10px;">{source} • {time_str}</span>'
            content_html += '</div>'
            content_html += f'<div class="text-primary" style="font-size: 11px; line-height: 1.4;">{text}</div>'
            content_html += '</div>'
    else:
        content_html += '<div class="text-secondary" style="text-align: center; padding: 20px; font-size: 11px;">Collecting sentiment data...</div>'

    content_html += '</div>'
    st.html(content_html)

def render_accuracy_badge(metrics):
    if not metrics or metrics.get('total_predictions', 0) == 0:
        return ""
    
    acc = metrics.get('overall_accuracy', 0)
    color = "#0ECB81" if acc >= 60 else "#F0B90B" if acc >= 50 else "#F6465D"
    
    badge_html = f'<div style="margin-top: 5px; font-size: 11px; background: rgba(255,255,255,0.05); padding: 2px 8px; border-radius: 10px; display: inline-block;">'
    badge_html += f'Signal Accuracy: <span style="color: {color}; font-weight: bold;">{acc}%</span>'
    badge_html += '</div>'
    return badge_html

@st.fragment(run_every=5)
def render_accuracy_panel():
    """Detailed accuracy stats panel."""
    cache = GLOBAL_CACHE
    metrics = cache.get('accuracy', {})
    sent_events = cache.get('sentiment_events', pd.DataFrame())
    
    if not metrics or metrics.get('total_predictions', 0) == 0:
        html = '<div class="binance-card" style="margin-bottom: 10px; text-align: center; padding: 20px;">'
        html += '<div class="text-secondary">Collecting prediction data...</div>'
        html += '<div style="font-size: 10px; color: #5E6673; margin-top: 5px;">Results appear after 15 mins</div>'
        html += '</div>'
        st.html(html)
        return

    acc = metrics.get('overall_accuracy', 0)
    total = metrics.get('total_predictions', 0)
    history = metrics.get('recent_history', [])
    
    color = "#0ECB81" if acc >= 60 else "#F0B90B" if acc >= 50 else "#F6465D"
    bg_color = color + "20"
    
    content_html = f'<div class="binance-card" style="margin-bottom: 10px;">'
    content_html += f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">'
    content_html += f'<span class="text-secondary" style="font-weight:bold; font-size:12px;">Signal Engine Performance</span>'
    content_html += f'<span style="font-size:10px; background: {bg_color}; color: {color}; padding: 2px 6px; border-radius: 4px;">High Confidence Only</span>'
    content_html += f'</div>'
    
    content_html += f'<div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 20px;">'
    content_html += f'<div>'
    content_html += f'<div style="font-size: 28px; font-weight: bold; color: {color};">{acc}%</div>'
    content_html += f'<div class="text-secondary" style="font-size: 11px;">Validated Signal Accuracy</div>'
    content_html += f'</div>'
    content_html += f'<div style="text-align: right;">'
    content_html += f'<div class="text-primary" style="font-size: 16px; font-weight: bold;">{total}</div>'
    content_html += f'<div class="text-secondary" style="font-size: 11px;">Samples</div>'
    content_html += f'</div>'
    content_html += f'</div>'
    
    content_html += f'<div class="text-secondary" style="font-size: 10px; margin-bottom: 5px;">RECENT OUTCOMES</div>'
    content_html += f'<div style="display: flex; gap: 4px; overflow-x: auto; padding-bottom: 5px;">'
    
    for h in history:
        res_color = "#0ECB81" if h['correct'] else "#F6465D"
        icon = "✓" if h['correct'] else "✗"
        try:
            t_obj = datetime.fromisoformat(h['time'])
            t_str = t_obj.strftime("%H:%M")
        except: t_str = ".."
            
        res_bg_color = res_color + "20"
        content_html += f'<div style="background: {res_bg_color}; border: 1px solid {res_color}; min-width: 45px; padding: 5px; border-radius: 4px; text-align: center;">'
        content_html += f'<div style="color: {res_color}; font-weight: bold; font-size: 14px;">{icon}</div>'
        content_html += f'<div style="font-size: 9px; color: #848E9C;">{t_str}</div>'
        content_html += f'<div style="font-size: 8px; color: {res_color};">{h["predicted"]}</div>'
        content_html += '</div>'
        
    content_html += '</div>'
    content_html += '</div>'
    st.html(content_html)

def main():
    inject_custom_css()
    
    # 0. Clean stale tracker locks
    if os.path.exists("prediction.lock"):
        try: shutil.rmtree("prediction.lock")
        except: pass
        
    # Environment Check
    if not TORCH_AVAILABLE:
        st.warning("⚠️ **Warning: PyTorch (torch) is not installed in the current environment.**")
        python_exe = get_python_executable()
        if "venv" in python_exe.lower():
            st.info(f"💡 A virtual environment was detected. Please run the dashboard using:\n\n`{python_exe} -m streamlit run binance_dashboard.py`")
        else:
            st.info("💡 Please install the requirements: `pip install -r requirements.txt`")

    config = load_config()
    is_fresh = config.get('params', {}).get('fresh_start', False)
    
    # 2. Setup Check
    _ = start_background_services()
    status = check_setup_status_simple()
    
    if is_fresh and (not status or status.get("status") != "complete"):
        s_prog = status.get('progress', 0) if status else 0.0
        s_msg = status.get('message', 'Initializing Fresh Start...') if status else 'Preparing environment...'
        s_det = status.get('detail', '') if status else ''
        
        log_lines = []
        if os.path.exists("logs/setup.log"):
            try:
                with open("logs/setup.log", "r") as f:
                    log_lines = f.readlines()[-15:]
            except: pass
        
        safe_logs = [l.strip().replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;") for l in log_lines]
        log_html = "<br>".join(safe_logs)
        
        st.html(f"""
            <div style="display:flex; justify-content:center; align-items:center; height:100vh; flex-direction:column; background-color:#161A25;">
                <h1 style="color:#F0B90B; font-family: 'IBM Plex Sans', sans-serif;">Binance Pro AI Initialization</h1>
                <div style="width: 60%; background-color: #2B3139; height: 8px; border-radius: 4px; overflow:hidden; margin-bottom: 20px;">
                    <div style="width: {int(s_prog*100)}%; background-color: #F0B90B; height: 100%; transition: width 0.5s;"></div>
                </div>
                <div style="color:#EAECEF; font-size: 18px; margin-bottom: 5px;">{s_msg}</div>
                <div style="color:#848E9C; font-size:14px; margin-bottom: 30px;">{s_det}</div>
                <div style="width: 70%; background-color: #0B0E11; border: 1px solid #2B3139; border-radius: 6px; padding: 15px; font-family: 'Courier New', monospace; font-size: 12px; color: #0ECB81; overflow-y: hidden; height: 300px; display: flex; flex-direction: column-reverse;">
                   <div>{log_html}</div>
                   <div style="color: #848E9C; border-bottom: 1px solid #2B3139; margin-bottom: 10px; padding-bottom: 5px;">> System Logs (Live Stream & Training)</div>
                </div>
                <div style="margin-top: 20px; color: #F6465D; font-size: 12px;">
                    * Live Data Collector is running in background.<br>
                    * Please do not close this window until training completes.
                </div>
            </div>
        """)
        time.sleep(2)
        st.rerun()
        return

    if status and status.get("status") == "error":
        st.error(f"Setup Failed: {status.get('detail')}")
        return

    # 3. Start Engines only after setup is ready (or if not in fresh start)
    data_orchestrator() 
    
    render_header_ticker()
    
    # Main layout with side panels
    col_chart, col_side = st.columns([0.75, 0.25])
    
    with col_chart:
        render_main_chart()
        render_accuracy_panel()
        render_sentiment_panel()
    
    with col_side:
        render_signals_panel()
        render_macro_panel()
        render_order_book()
        render_recent_trades()


if __name__ == "__main__":
    main()
