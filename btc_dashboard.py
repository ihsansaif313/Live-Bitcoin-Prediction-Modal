"""
Bitcoin Prediction Dashboard
Real-time visualization of candles, trades, and AI predictions.
Run with: streamlit run btc_dashboard.py
"""

import os
import sys
import time
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import subprocess
import yaml
import json
import threading
from typing import List, Dict

# Import feature generation logic
try:
    from build_dataset import make_features
except ImportError:
    st.error("Could not import build_dataset.py. Make sure it is in the same directory.")

# Page Config moved to main()

# Constants
DATASET_CSV = "btc_dataset.csv"
LIVE_CANDLES_CSV = "btc_live_candles.csv"
LIVE_TRADES_CSV = "btc_trades_live.csv"
MODELS_DIR = "models"
SCALERS_FILE = "scalers.pkl"

# --- Data Loading & Orchestration ---

def load_config():
    """Load configuration from config.yaml."""
    if os.path.exists("config.yaml"):
        with open("config.yaml", "r") as f:
            return yaml.safe_load(f)
    return {}

def run_step(script_name: str, description: str):
    """Run a script synchronously and wait for completion, logging to a file."""
    log_file = "logs/setup.log"
    if not os.path.exists("logs"): os.makedirs("logs")
    
    try:
        with open(log_file, "a") as f:
            f.write(f"\n>>> Starting: {description} ({script_name}) at {time.ctime()}\n")
            f.flush()
            # Run the process and capture all output to the log file
            subprocess.check_call([sys.executable, script_name], stdout=f, stderr=subprocess.STDOUT)
            f.write(f"\n<<< Completed: {description} successfully.\n")
    except Exception as e:
        with open(log_file, "a") as f:
            f.write(f"\nERROR in {description}: {str(e)}\n")
        raise RuntimeError(f"Step {description} failed: {e}")

def update_status(status, progress, message, detail=""):
    """Write status to JSON file for dashboard to read."""
    status_file = "logs/setup_status.json"
    if not os.path.exists("logs"): os.makedirs("logs")
    try:
        with open(status_file, "w") as f:
            json.dump({
                "status": status,
                "progress": progress,
                "message": message,
                "detail": detail
            }, f)
    except: pass

def cleanup_data(config: Dict):
    """Delete all data and models for a fresh start."""
    # Files to delete
    paths = config.get('paths', {})
    files = [
        paths.get('historical_data', 'btc_historical.csv'),
        paths.get('historical_data_clean', 'btc_historical_clean.csv'),
        paths.get('dataset', 'btc_dataset.csv'),
        paths.get('live_candles', 'btc_live_candles.csv'),
        "btc_trades_live.csv",
        "btc_features.csv",
        "btc_features_normalized.csv",
        "scalers.pkl",
        "logs/setup.log",
        "logs/data_cleaning.log",
        "logs/continuous_learning.log",
        os.path.join("reports", "metrics.csv"),
        os.path.join("reports", "evaluation_metrics.csv"),
        os.path.join("reports", "backtest_summary.csv"),
        os.path.join("reports", "actual_vs_predicted.png"),
        os.path.join("reports", "backtest_equity.png"),
        os.path.join("reports", "confusion_matrix.png"),
        os.path.join("reports", "metrics.json")
    ]
    
    for f in files:
        if os.path.exists(f):
            try: os.remove(f)
            except: pass

    # Models directory
    models_dir = paths.get('models_dir', 'models')
    if os.path.exists(models_dir):
        try:
            import shutil
            shutil.rmtree(models_dir)
        except: pass

def run_setup_sequence(config: Dict):
    """Run all setup steps in order, updating status."""
    log_file = "logs/setup.log"
    if not os.path.exists("logs"): os.makedirs("logs")
    # Clear setup log at the start of a new run
    with open(log_file, "w") as f:
        f.write(f"--- Initialization Started: {time.ctime()} ---\n")

    try:
        # 0. Cleanup for Fresh Start
        if config.get('params', {}).get('fresh_start', False):
            update_status("running", 0.05, "Cleaning up old data", "Deleting previous CSVs and Models...")
            cleanup_data(config)
        
        # 1. Start Live Streaming & Aggregation IMMEDIATELY after cleanup
        # This captures trades "now" while history is downloading "from the past"
        # to ensure no gap in data.
        update_status("running", 0.08, "Initializing Live Data Stream", "Connecting to Binance WebSockets...")
        creationflags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
        subprocess.Popen([sys.executable, "live_stream.py"], creationflags=creationflags, env=os.environ.copy())
        subprocess.Popen([sys.executable, "aggregate_live_to_candles.py"], creationflags=creationflags, env=os.environ.copy())

        # 2. Historical Data
        dataset_path = config.get('paths', {}).get('dataset', 'btc_dataset.csv')
        if not os.path.exists(dataset_path):
            update_status("running", 0.1, "Downloading Historical Data", "Fetching 6 months of OHLC data...")
            run_step("historical_data.py", "Download Historical Data")
            
            update_status("running", 0.2, "Cleaning Data", "Detecting outliers and removing noise...")
            try:
                from data_cleaner import clean_historical_data
                clean_historical_data(config['paths']['historical_data'], config['paths']['historical_data_clean'])
            except:
                run_step("data_cleaner.py", "Clean Data")
            
            update_status("running", 0.3, "Building Dataset", "Generating technical indicators...")
            run_step("build_dataset.py", "Build Initial Dataset")
        
        # 2. Model Training
        models_dir = config.get('paths', {}).get('models_dir', 'models')
        reg_keras = os.path.join(models_dir, "btc_model_reg.keras")
        if not os.path.exists(reg_keras):
            update_status("running", 0.4, "Training Models", "Initializing training process...")
            run_step("train_models.py", "Train Initial Models")
            
        update_status("complete", 1.0, "Setup Complete", "Launching live services...")
        
        # Start Continuous Learning ONLY after setup is finished
        creationflags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
        subprocess.Popen([sys.executable, "continuous_learning.py"], creationflags=creationflags, env=os.environ.copy())
        
    except Exception as e:
        update_status("error", 0.0, "Setup Failed", str(e))

@st.cache_resource
def start_background_services():
    """Start the setup thread. Other services are started by the setup sequence in order."""
    config = load_config()
    
    # Run setup if needed
    setup_thread = threading.Thread(target=run_setup_sequence, args=(config,))
    setup_thread.start()
    
    # We return the thread so st.cache_resource tracks it (though internal services manage themselves)
    return setup_thread

@st.cache_data(ttl=2)  # Cache for 2 seconds to allow near real-time updates
def load_data():
    """Load historical and live candle data."""
    # 1. Historical Dataset
    hist_df = pd.DataFrame()
    if os.path.exists(DATASET_CSV):
        try:
            hist_df = pd.read_csv(DATASET_CSV)
            hist_df['timeOpen'] = pd.to_datetime(hist_df['timeOpen'], utc=True)
        except Exception:
            hist_df = pd.DataFrame()

    # 2. Live Candles (Recent)
    live_df = pd.DataFrame()
    if os.path.exists(LIVE_CANDLES_CSV):
        try:
            live_df = pd.read_csv(LIVE_CANDLES_CSV)
            live_df['timeOpen'] = pd.to_datetime(live_df['timeOpen'], utc=True, errors='coerce')
            live_df = live_df.dropna(subset=['timeOpen'])
        except Exception:
            live_df = pd.DataFrame()

    # Merge: Prefer Dataset, append new Live candles that aren't in Dataset
    if not hist_df.empty and not live_df.empty:
        last_hist_time = hist_df['timeOpen'].max()
        new_live = live_df[live_df['timeOpen'] > last_hist_time]
        combined_df = pd.concat([hist_df, new_live]).reset_index(drop=True)
    elif not hist_df.empty:
        combined_df = hist_df
    else:
        combined_df = live_df
        
    # Sort and Deduplicate
    if not combined_df.empty:
        combined_df = combined_df.sort_values('timeOpen').drop_duplicates('timeOpen').reset_index(drop=True)

    # 3. Recent Trades (Optimized for speed)
    trades_df = pd.DataFrame()
    if os.path.exists(LIVE_TRADES_CSV):
        try:
            # Efficiently read only the tail of the file to maintain performance
            # Using tail -n 200 (approx)
            with open(LIVE_TRADES_CSV, 'rb') as f:
                f.seek(0, os.SEEK_END)
                buffer_size = 1024 * 10 # 10KB is enough for ~100 trades
                if f.tell() > buffer_size:
                    f.seek(-buffer_size, os.SEEK_END)
                
                # Read, skip first partial line
                lines = f.read().decode('utf-8', errors='ignore').splitlines()
                if len(lines) > 1:
                    # Parse last N lines
                    data = [l.split(',') for l in lines[-100:]]
                    temp_df = pd.DataFrame(data, columns=['tradeId', 'price', 'qty', 'quoteQty', 'time', 'isBuyerMaker'])
                    
                    # Convert types
                    temp_df['price'] = pd.to_numeric(temp_df['price'], errors='coerce')
                    temp_df['time'] = pd.to_datetime(temp_df['time'], utc=True, errors='coerce')
                    temp_df = temp_df.dropna(subset=['time', 'price'])
                    
                    if not temp_df.empty:
                        temp_df['symbol'] = 'BTCUSDT'
                        trades_df = temp_df.sort_values('time', ascending=False)
        except Exception as e:
            # print(f"Trade read error: {e}")
            trades_df = pd.DataFrame()

    return combined_df, trades_df

@st.cache_resource
def load_models():
    """Load trained models and scalers (supports .pkl and .h5)."""
    models = {}
    scalers = None
    
    try:
        # Load Scalers
        if os.path.exists(SCALERS_FILE):
            with open(SCALERS_FILE, 'rb') as f:
                scalers = pickle.load(f)
        
        # Helper to load model
        def load_specific_model(type_name, filename_base):
            # 0. Try .keras (Modern Keras 3) first - most stable for current environment
            keras_path = os.path.join(MODELS_DIR, f"{filename_base}.keras")
            h5_path = os.path.join(MODELS_DIR, f"{filename_base}.h5")
            
            # Paths to try for deep learning
            dl_paths = [keras_path, h5_path]
            
            for dl_path in dl_paths:
                if os.path.exists(dl_path):
                    try:
                        import tensorflow as tf
                        
                        # Get custom objects
                        custom_objects = {}
                        try:
                            from tensorflow.keras.layers import LayerNormalization, MultiHeadAttention
                            custom_objects = {'LayerNormalization': LayerNormalization, 'MultiHeadAttention': MultiHeadAttention}
                        except:
                            pass

                        # Sequence of attempts
                        loading_strategies = [
                            lambda p: tf.keras.models.load_model(filepath=p, custom_objects=custom_objects, compile=False),
                            lambda p: tf.keras.models.load_model(filepath=p, custom_objects=custom_objects),
                            lambda p: tf.keras.models.load_model(filepath=p, compile=False),
                            lambda p: tf.keras.models.load_model(p),
                        ]

                        for loader in loading_strategies:
                            try:
                                model = loader(dl_path)
                                if model: return model
                            except:
                                continue
                    except Exception:
                        pass
            
            # 2. Try .pkl (Standard Best Model)
            pkl_path = os.path.join(MODELS_DIR, f"{filename_base}.pkl")
            if os.path.exists(pkl_path):
                try:
                    with open(pkl_path, 'rb') as f:
                        return pickle.load(f)
                except Exception:
                    pass

            # 3. Tertiary Fallback: Explicit baseline backup
            baseline_path = os.path.join(MODELS_DIR, f"{filename_base}_baseline.pkl")
            if os.path.exists(baseline_path):
                try:
                    with open(baseline_path, 'rb') as f:
                        return pickle.load(f)
                except Exception:
                    pass
            
            return None

        models['Reg'] = load_specific_model("Regression", "btc_model_reg")
        models['Cls'] = load_specific_model("Classification", "btc_model_cls")

        if not models['Reg'] or not models['Cls']:
             st.warning("Could not load one or more models. Waiting for training to complete...")

    except Exception as e:
        st.warning(f"Error loading resources: {e}")
        
    return models, scalers

@st.cache_data(ttl=1)
def predict_live(df, _models, _scalers, current_price):
    """Generate prediction for the next 15 minutes."""
    if df.empty or not _models or not _scalers:
        return None

    # We need enough data to generate features (lag, rolling, etc.)
    # Build dataset logic requires specific columns
    # We'll take the tail of the dataframe to save compute
    tail_size = 200 # Sufficient for lags/rolling
    df_tail = df.iloc[-tail_size:].copy() if len(df) > tail_size else df.copy()
    
    # Generate Features
    # Note: make_features expects a specific structure
    features_df = make_features(df_tail)
    
    if features_df.empty:
        return None
        
    # Normalize
    last_row = features_df.iloc[[-1]].copy()
    feature_cols = [c for c in features_df.columns if c != 'timeOpen']
    
    # Apply scaling to the single row
    # (Inefficient to re-scale everything, but robust for display)
    # Ideally we'd scale just the vector, but scaler is fitted on DF structure usually? 
    # Scalers in typical sklearn are column-wise. We stored a dict of {col: scaler}.
    
    X_input = []
    valid_cols = []
    
    # This matching logic must strictly follow continuous_learning.py / build_dataset.py
    # Re-use scaler dict
    feature_input = {}
    
    for col in feature_cols:
        val = last_row[col].values[0]
        if col in _scalers:
            # Scaler expects 2D array
            val_scaled = _scalers[col].transform([[val]])[0][0]
            feature_input[col] = val_scaled
        else:
             feature_input[col] = val # Should not happen if build_dataset is consistent
    
    # Check model expected features provided by sklearn
    # With sklearn pipeline/models, input must be array of correct shape
    # We assume 'feature_cols' order matches training. 
    # Since we use the SAME `make_features` function, order of columns in DF *should* be stable
    # BUT dict iteration is insertion ordered in modern python.
    # Reliable way: models['Reg'].feature_names_in_ if available, else reliance on pipeline solidity.
    
    # Let's trust the DataFrame order from make_features matches training time
    X_vector = np.array([feature_input[c] for c in feature_cols]).reshape(1, -1)
    
    # Predict
    # Handle Input Shapes:
    # Sklearn expects (n_samples, n_features) -> (1, 30)
    # Keras (LSTM) expects (n_samples, window_size, n_features) -> (1, 60, 30)
    
    pred_price = None
    pred_dir = None
    
    try:
        # Check Regression Model Type
        reg_model = _models['Reg']
        if hasattr(reg_model, 'predict'):
            # Check if it's Keras (has 'input_shape' or similar, or just try/except)
            is_keras_reg = hasattr(reg_model, 'inputs') 
            
            if is_keras_reg:
                # We need a sequence, but we only have 1 row here from `make_features`
                # Realistically, `make_features` calculates row-based features (RSI, MA).
                # LSTM needs a sequence of *these rows*.
                # If we only pass 1 row to LSTM trained on 60, it will crash.
                
                # FALLBACK: If we loaded a Keras model, we need the last WINDOW_SIZE rows of *features*
                # Re-calculate features for the whole tail
                full_features_df = make_features(df_tail)
                
                # Normalize whole tail
                norm_tail = pd.DataFrame(index=full_features_df.index)
                for col in feature_cols:
                    if col in _scalers:
                        col_vals = full_features_df[col].values.reshape(-1, 1)
                        norm_tail[col] = _scalers[col].transform(col_vals).flatten()
                    else:
                        norm_tail[col] = full_features_df[col]
                
                # Get last 60
                window_size = 60
                if len(norm_tail) >= window_size:
                    X_seq = norm_tail.iloc[-window_size:].values.reshape(1, window_size, len(feature_cols))
                    predicted_return = float(reg_model.predict(X_seq, verbose=0)[0][0])
                    # Reconstruct price: Relative to current_price
                    pred_price = float(current_price * (1 + predicted_return))
                else:
                    return None
            else:
                # Sklearn (Baseline)
                X_vector = np.array([feature_input[c] for c in feature_cols]).reshape(1, -1)
                predicted_return = reg_model.predict(X_vector)[0]
                pred_price = float(current_price * (1 + predicted_return))

        # Check Classification Model Type
        cls_model = _models['Cls']
        if hasattr(cls_model, 'predict'):
            is_keras_cls = hasattr(cls_model, 'inputs')
            
            if is_keras_cls:
                # Same logic as above
                full_features_df = make_features(df_tail)
                norm_tail = pd.DataFrame(index=full_features_df.index)
                for col in feature_cols:
                    if col in _scalers:
                        col_vals = full_features_df[col].values.reshape(-1, 1)
                        norm_tail[col] = _scalers[col].transform(col_vals).flatten()
                    else:
                        norm_tail[col] = full_features_df[col]
                
                window_size = 60
                if len(norm_tail) >= window_size:
                    X_seq = norm_tail.iloc[-window_size:].values.reshape(1, window_size, len(feature_cols))
                    prob = cls_model.predict(X_seq, verbose=0)[0][0]
                    pred_dir = 1 if prob > 0.5 else 0
                else:
                    return None
            else:
                # Sklearn
                X_vector = np.array([feature_input[c] for c in feature_cols]).reshape(1, -1)
                pred_dir = cls_model.predict(X_vector)[0]
                
    except Exception as e:
        # print(f"Prediction Error: {e}")
        return None
    
    if pred_price is None or pred_dir is None:
        return None

    return {
        'next_close': pred_price,
        'direction': "UP 🟢" if pred_price > current_price else "DOWN 🔴",
        'current_close': current_price,
        'time': df_tail.iloc[-1]['timeOpen']
    }

# --- Main UI ---
import json

def check_setup_status():
    """Check if the backend setup is still running."""
    status_file = "logs/setup_status.json"
    if os.path.exists(status_file):
        try:
            with open(status_file, "r") as f:
                status = json.load(f)
            return status
        except:
            return None
    return None

def main():
    st.set_page_config(
        page_title="BTC Live AI Dashboard",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize Setup & Services
    _ = start_background_services()
    
    # Check setup status
    status = check_setup_status()
    if status and status.get("status") == "running":
        st.title("🚀 Initializing Bitcoin AI System")
        st.markdown(f"### {status.get('message', 'Loading...')}")
        st.progress(status.get("progress", 0.0))
        
        # Live Log Console
        st.subheader("System Logs (Live)")
        log_file = "logs/setup.log"
        if os.path.exists(log_file):
            try:
                with open(log_file, "r") as f:
                    log_lines = f.readlines()
                    # Show the last 20 lines to keep it readable
                    st.code("".join(log_lines[-20:]), language="text")
            except:
                st.info("Loading logs...")
        
        st.divider()
        st.warning("🔄 **Daily Automatic Retraining Enabled**")
        st.info("The system is configured to perform a fresh run every time the app starts. This ensures your models are always trained on the most recent 6 months of data.")
        st.info("ℹ️ **Note for Streamlit Cloud:** Training on Cloud CPUs can take 5-10 minutes. Please keep this tab open.")
        
        with st.expander("Cloud Deployment Tips"):
            st.write("""
            - **Ephemeral Storage**: Streamlit Cloud resets every 24h. Any models trained on the cloud will be lost on reset.
            - **Recommendation**: Train models locally on your GPU, then push the files in the `models/` folder to GitHub.
            - **Persistent Option**: For long-term cloud persistence, use AWS S3 or a Database to store models.
            """)
        
        st.info("The system is currently training. Dashboard will unlock automatically when finished.")
        time.sleep(2)
        st.rerun()
        return
        
    if status and status.get("status") == "error":
        st.error("Setup Failed!")
        st.error(status.get("detail"))
        return

    st.title("BTC/USDT AI Prediction Dashboard 🚀")
    st.sidebar.header("Controls")
    refresh_rate = st.sidebar.slider("Refresh Rate (sec)", 0.5, 60.0, 2.0)
    lookback = st.sidebar.selectbox("Lookback Window", ["1 Hour", "6 Hours", "24 Hours", "7 Days"], index=1)
    timezone_opt = st.sidebar.selectbox("Timezone Display", ["Local", "UTC", "Asia/Karachi", "US/Eastern"], index=0)
    
    # Load Data
    df, trades = load_data()
    models, scalers = load_models()
    
    if df.empty:
        st.info("Waiting for data... Ensure live_stream.py is running.")
        time.sleep(refresh_rate)
        st.rerun()
        return

    # Filter by lookback
    cutoff = pd.Timestamp.now(tz='UTC')
    if lookback == "1 Hour": cutoff -= pd.Timedelta(hours=1)
    elif lookback == "6 Hours": cutoff -= pd.Timedelta(hours=6)
    elif lookback == "24 Hours": cutoff -= pd.Timedelta(hours=24)
    elif lookback == "7 Days": cutoff -= pd.Timedelta(days=7)
    
    df_view = df[df['timeOpen'] > cutoff].copy()
    
    # Apply Timezone
    if timezone_opt != "UTC" and not df_view.empty:
        try:
            if timezone_opt == "Local":
                # Strip timezone to show as local/naive for the user's browser
                df_view['timeOpen'] = df_view['timeOpen'].dt.tz_convert(None)
            else:
                df_view['timeOpen'] = df_view['timeOpen'].dt.tz_convert(timezone_opt)
        except Exception:
            pass

    # Calculate Metrics
    # IMPORTANT: Use THE LATEST TRADE for real-time price, not the last candle
    current_price = trades['price'].iloc[0] if not trades.empty else df.iloc[-1]['close']
    last_price = df.iloc[-1]['close'] if not df.empty else current_price # Previous candle close
    price_delta = current_price - last_price
    volume_24h = df[df['timeOpen'] > (df['timeOpen'].max() - pd.Timedelta(hours=24))]['volume'].sum()

    # Top Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Price", f"${current_price:,.2f}", f"{price_delta:,.2f}")
    col3.metric("24h Volume", f"{volume_24h:,.0f}")
    
    # Prediction Section
    pred = predict_live(df, models, scalers, current_price)
    if pred:
        col2.metric("Predicted 15-Min Close", f"${pred['next_close']:,.2f}", 
                    delta=f"{pred['next_close'] - current_price:,.2f}")
        col4.metric("Predicted 15-Min Direction", pred['direction'])

    # Charts
    st.subheader("Market Overview")
    
    # Candlestick + MA
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, subplot_titles=('Price & Prediction', 'Volume'), 
                        row_width=[0.2, 0.7])

    # Candle
    fig.add_trace(go.Candlestick(x=df_view['timeOpen'],
                open=df_view['open'], high=df_view['high'],
                low=df_view['low'], close=df_view['close'], name='OHLC'), row=1, col=1)

    # MA (Simple calculation for viewing)
    ma50 = df_view['close'].rolling(50).mean()
    fig.add_trace(go.Scatter(x=df_view['timeOpen'], y=ma50, line=dict(color='orange', width=1), name='MA50'), row=1, col=1)

    # Prediction Marker (if available)
    if pred:
        # Plot predicted point 15 minutes ahead
        next_time = pred['time'] + pd.Timedelta(minutes=15)
        fig.add_trace(go.Scatter(
            x=[next_time], y=[pred['next_close']],
            mode='markers', marker=dict(color='purple', size=12, symbol='star'),
            name='15-Min AI Prediction'
        ), row=1, col=1)

    # Volume
    colors = ['green' if row['open'] - row['close'] >= 0 else 'red' for index, row in df_view.iterrows()]
    fig.add_trace(go.Bar(x=df_view['timeOpen'], y=df_view['volume'], marker_color=colors, name='Volume'), row=2, col=1)

    fig.update_layout(height=600, xaxis_rangeslider_visible=False, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    # Recent Trades
    st.subheader("Live Trades Feed")
    if not trades.empty:
        # Style trades
        def highlight_buy_sell(val):
            color = 'green' if val else 'red'
            return f'color: {color}'

        # Format
        trades_display = trades.copy()
        trades_display['time'] = trades_display['time'].dt.strftime('%H:%M:%S')
        trades_display['Side'] = trades_display['isBuyerMaker'].map({True: 'SELL', False: 'BUY'}) # Maker buy = Taker sell? 
        # Binance: isBuyerMaker=True means maker was buyer -> Taker was seller -> SELL trade
        
        st.dataframe(
            trades_display[['time', 'symbol', 'price', 'qty', 'Side']],
            use_container_width=True,
            height=300
        )
    else:
        st.write("No recent trades loaded.")

    # Auto-refresh
    time.sleep(refresh_rate)
    st.rerun()

if __name__ == "__main__":
    main()
