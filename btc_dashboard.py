"""
Bitcoin Prediction Dashboard
Real-time visualization of candles, trades, and AI predictions.
Run with: streamlit run btc_dashboard.py
"""

import os
import time
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

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

# --- Data Loading ---

@st.cache_data(ttl=2)  # Cache for 2 seconds to allow near real-time updates
def load_data():
    """Load historical and live candle data."""
    # 1. Historical Dataset
    if os.path.exists(DATASET_CSV):
        hist_df = pd.read_csv(DATASET_CSV)
        hist_df['timeOpen'] = pd.to_datetime(hist_df['timeOpen'], utc=True)
    else:
        hist_df = pd.DataFrame()

    # 2. Live Candles (Recent)
    if os.path.exists(LIVE_CANDLES_CSV):
        live_df = pd.read_csv(LIVE_CANDLES_CSV)
        live_df['timeOpen'] = pd.to_datetime(live_df['timeOpen'], utc=True, errors='coerce')
        live_df = live_df.dropna(subset=['timeOpen'])
    else:
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

    # 3. Recent Trades (Raw)
    trades_df = pd.DataFrame()
    if os.path.exists(LIVE_TRADES_CSV):
        try:
            temp_df = pd.read_csv(LIVE_TRADES_CSV, names=['symbol', 'price', 'qty', 'time', 'isBuyerMaker'])
            # Convert time with error coercion
            temp_df['time'] = pd.to_datetime(temp_df['time'], unit='ms', utc=True, errors='coerce')
            temp_df = temp_df.dropna(subset=['time'])
            
            if not temp_df.empty:
                trades_df = temp_df.sort_values('time', ascending=False).head(50)
        except Exception:
            # If any error occurs, return empty to avoid crashes
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
            # 1. Try .h5 (Deep Learning) first
            h5_path = os.path.join(MODELS_DIR, f"{filename_base}.h5")
            if os.path.exists(h5_path):
                try:
                    import tensorflow as tf
                    from tensorflow.keras.layers import LayerNormalization, MultiHeadAttention
                    
                    custom_objects = {
                        'LayerNormalization': LayerNormalization,
                        'MultiHeadAttention': MultiHeadAttention
                    }

                    # Try loading with compile=False and custom_objects
                    try:
                        return tf.keras.models.load_model(h5_path, custom_objects=custom_objects, compile=False)
                    except (TypeError, ValueError):
                        # Fallback 1: Try default load with custom_objects
                        return tf.keras.models.load_model(h5_path, custom_objects=custom_objects)
                except Exception as e:
                    st.warning(f"Found {h5_path} but failed to load: {e}")
            
            # 2. Try .pkl (Standard Best Model)
            pkl_path = os.path.join(MODELS_DIR, f"{filename_base}.pkl")
            if os.path.exists(pkl_path):
                with open(pkl_path, 'rb') as f:
                    return pickle.load(f)

            # 3. Tertiary Fallback: Look for explicit baseline backup
            # This is created by the updated train_models.py
            baseline_path = os.path.join(MODELS_DIR, f"{filename_base}_baseline.pkl")
            if os.path.exists(baseline_path):
                with open(baseline_path, 'rb') as f:
                    # st.info(f"Using fallback baseline model for {type_name}") # Optional info
                    return pickle.load(f)
            
            return None

        models['Reg'] = load_specific_model("Regression", "btc_model_reg")
        models['Cls'] = load_specific_model("Classification", "btc_model_cls")

        if not models['Reg'] or not models['Cls']:
             st.warning("Could not load one or more models. Waiting for training to complete...")

    except Exception as e:
        st.warning(f"Error loading resources: {e}")
        
    return models, scalers

def predict_live(df, models, scalers):
    """Generate prediction for the next minute."""
    if df.empty or not models or not scalers:
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
        if col in scalers:
            # Scaler expects 2D array
            val_scaled = scalers[col].transform([[val]])[0][0]
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
        reg_model = models['Reg']
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
                    if col in scalers:
                        col_vals = full_features_df[col].values.reshape(-1, 1)
                        norm_tail[col] = scalers[col].transform(col_vals).flatten()
                    else:
                        norm_tail[col] = full_features_df[col]
                
                # Get last 60
                window_size = 60
                if len(norm_tail) >= window_size:
                    X_seq = norm_tail.iloc[-window_size:].values.reshape(1, window_size, len(feature_cols))
                    pred_price = float(reg_model.predict(X_seq, verbose=0)[0][0])
                else:
                    return None
            else:
                # Sklearn
                X_vector = np.array([feature_input[c] for c in feature_cols]).reshape(1, -1)
                pred_price = reg_model.predict(X_vector)[0]

        # Check Classification Model Type
        cls_model = models['Cls']
        if hasattr(cls_model, 'predict'):
            is_keras_cls = hasattr(cls_model, 'inputs')
            
            if is_keras_cls:
                # Same logic as above
                full_features_df = make_features(df_tail)
                norm_tail = pd.DataFrame(index=full_features_df.index)
                for col in feature_cols:
                    if col in scalers:
                        col_vals = full_features_df[col].values.reshape(-1, 1)
                        norm_tail[col] = scalers[col].transform(col_vals).flatten()
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
        'direction': "UP 🟢" if pred_dir == 1 else "DOWN 🔴",
        'current_close': df_tail.iloc[-1]['close'],
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
    
    # Check setup status
    status = check_setup_status()
    if status and status.get("status") == "running":
        st.title("🚀 Initializing Bitcoin AI System")
        st.markdown(f"### {status.get('message', 'Loading...')}")
        st.progress(status.get("progress", 0.0))
        st.code(status.get("detail", ""))
        st.info("Please wait while the system downloads history and trains models...")
        time.sleep(1)
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
    timezone_opt = st.sidebar.selectbox("Timezone Display", ["UTC", "Local (Browser)", "Asia/Karachi", "US/Eastern"], index=0)
    
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
            if timezone_opt == "Local (Browser)":
                # Streamlit/Plotly handles this efficiently usually, but we can try naive
                pass 
            else:
                df_view['timeOpen'] = df_view['timeOpen'].dt.tz_convert(timezone_opt)
        except Exception:
            pass

    # Calculate Metrics
    current_price = df.iloc[-1]['close']
    last_price = df.iloc[-2]['close'] if len(df) > 1 else current_price
    price_delta = current_price - last_price
    volume_24h = df[df['timeOpen'] > (df['timeOpen'].max() - pd.Timedelta(hours=24))]['volume'].sum()

    # Top Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Price", f"${current_price:,.2f}", f"{price_delta:,.2f}")
    col3.metric("24h Volume", f"{volume_24h:,.0f}")
    
    # Prediction Section
    pred = predict_live(df, models, scalers)
    if pred:
        col2.metric("Predicted Next Close", f"${pred['next_close']:,.2f}", 
                    delta=f"{pred['next_close'] - current_price:,.2f}")
        col4.metric("Predicted Direction", pred['direction'])

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
        # Plot predicted point one minute ahead
        next_time = pred['time'] + pd.Timedelta(minutes=1)
        fig.add_trace(go.Scatter(
            x=[next_time], y=[pred['next_close']],
            mode='markers', marker=dict(color='purple', size=12, symbol='star'),
            name='AI Prediction'
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
