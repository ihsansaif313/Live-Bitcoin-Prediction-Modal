"""
Bitcoin Dataset Building Script
Merges historical and live candles, creates engineered features, and applies normalization.
"""

import pandas as pd
import numpy as np
import logging
import pickle
import os
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
HISTORICAL_CSV = "btc_historical_clean.csv"
LIVE_CSV = "btc_live_candles.csv"
SENTIMENT_CSV = "sentiment_minute.csv"
ORDERBOOK_CSV = "orderbook_depth.csv"
MACRO_CSV = "macro_factors.csv"
OUTPUT_DATASET = "btc_dataset.csv"
OUTPUT_FEATURES = "btc_features.csv"
OUTPUT_NORMALIZED = "btc_features_normalized.csv"
SCALERS_FILE = "scalers.pkl"
TRAIN_SPLIT = 0.8  # 80% for training


def load_candles() -> pd.DataFrame:
    """
    Load and merge historical and live candles.
    
    Returns:
        DataFrame with merged candles, deduplicated and sorted
    """
    logger.info("Loading candles...")
    
    candles_list = []
    
    # Load historical data
    if os.path.exists(HISTORICAL_CSV):
        historical = pd.read_csv(HISTORICAL_CSV, low_memory=False)
        historical['timeOpen'] = pd.to_datetime(historical['timeOpen'], utc=True, errors='coerce')
        historical = historical.dropna(subset=['timeOpen'])
        logger.info(f"Loaded {len(historical)} historical candles")
        candles_list.append(historical)
    else:
        logger.warning(f"{HISTORICAL_CSV} not found")
    # Load live data
    if os.path.exists(LIVE_CSV):
        live = pd.read_csv(LIVE_CSV, low_memory=False)
        live['timeOpen'] = pd.to_datetime(live['timeOpen'], utc=True, errors='coerce')
        live = live.dropna(subset=['timeOpen'])
        logger.info(f"Loaded {len(live)} live candles")
        candles_list.append(live)
    else:
        logger.warning(f"{LIVE_CSV} not found")
    
    if not candles_list:
        raise FileNotFoundError("No candle data found")
    
    # Merge
    candles = pd.concat(candles_list, ignore_index=True)
    
    # Deduplicate by timeOpen
    initial_count = len(candles)
    candles = candles.drop_duplicates(subset=['timeOpen'], keep='first')
    duplicates_removed = initial_count - len(candles)
    if duplicates_removed > 0:
        logger.info(f"Removed {duplicates_removed} duplicate candles")
    
    # Sort by timeOpen
    candles = candles.sort_values('timeOpen').reset_index(drop=True)
    
    logger.info(f"Total candles: {len(candles)}")
    logger.info(f"Date range: {candles['timeOpen'].min()} to {candles['timeOpen'].max()}")
    
    return candles


def load_sentiment_data() -> pd.DataFrame:
    """
    Load sentiment data from sentiment_minute.csv.
    
    Returns:
        DataFrame with sentiment features per minute, or empty DataFrame if file doesn't exist
    """
    logger.info("Loading sentiment data...")
    
    if not os.path.exists(SENTIMENT_CSV):
        logger.warning(f"{SENTIMENT_CSV} not found - sentiment features will be zero-filled")
        return pd.DataFrame()
    
    try:
        sentiment = pd.read_csv(SENTIMENT_CSV, low_memory=False)
        sentiment['timeOpen'] = pd.to_datetime(sentiment['timeOpen'], utc=True)
        
        # Select only the columns we need
        required_cols = ['timeOpen', 'sentiment_mean', 'sentiment_neg_mean', 
                        'relevance_score', 'events_count', 'negative_spike_flag']
        
        # Add missing columns with default values
        for col in required_cols:
            if col not in sentiment.columns and col != 'timeOpen':
                sentiment[col] = 0.0
        
        sentiment = sentiment[required_cols]
        
        logger.info(f"Loaded {len(sentiment)} sentiment records")
        logger.info(f"Sentiment date range: {sentiment['timeOpen'].min()} to {sentiment['timeOpen'].max()}")
        
        return sentiment
    except Exception as e:
        logger.error(f"Error loading sentiment data: {e}")
        return pd.DataFrame()


def load_orderbook_data() -> pd.DataFrame:
    """
    Load orderbook depth data from orderbook_depth.csv.
    """
    logger.info("Loading orderbook data...")
    
    if not os.path.exists(ORDERBOOK_CSV):
        logger.warning(f"{ORDERBOOK_CSV} not found - orderbook features will be zero-filled")
        return pd.DataFrame()
    
    try:
        orderbook = pd.read_csv(ORDERBOOK_CSV, low_memory=False)
        orderbook['timeOpen'] = pd.to_datetime(orderbook['timeOpen'], utc=True)
        
        required_cols = ['timeOpen', 'mid_price', 'spread', 'bid_depth_20', 'ask_depth_20', 
                         'imbalance', 'largest_wall_dist', 'cum_depth_10bps']
        
        for col in required_cols:
            if col not in orderbook.columns and col != 'timeOpen':
                orderbook[col] = 0.0
        
        orderbook = orderbook[required_cols]
        logger.info(f"Loaded {len(orderbook)} orderbook records")
        return orderbook
    except Exception as e:
        logger.error(f"Error loading orderbook data: {e}")
        return pd.DataFrame()


def load_macro_data() -> pd.DataFrame:
    """
    Load macro factors data from macro_factors.csv.
    """
    logger.info("Loading macro data...")
    
    if not os.path.exists(MACRO_CSV):
        logger.warning(f"{MACRO_CSV} not found - macro features will be zero-filled")
        return pd.DataFrame()
    
    try:
        macro = pd.read_csv(MACRO_CSV, low_memory=False)
        macro['timeOpen'] = pd.to_datetime(macro['timeOpen'], utc=True)
        
        required_cols = ['timeOpen', 'spx_close', 'dxy_close', 'returns_spx', 'returns_dxy', 'z_spx', 'z_dxy']
        
        for col in required_cols:
            if col not in macro.columns and col != 'timeOpen':
                macro[col] = 0.0
        
        macro = macro[required_cols]
        logger.info(f"Loaded {len(macro)} macro records")
        return macro
    except Exception as e:
        logger.error(f"Error loading macro data: {e}")
        return pd.DataFrame()


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI indicator."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate MACD indicator."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calculate_bollinger_bands(series: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Bollinger Bands."""
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    return upper_band, sma, lower_band


def make_features(candles: pd.DataFrame) -> pd.DataFrame:
    """
    Create engineered features from candles.
    
    Args:
        candles: DataFrame with raw candles
    
    Returns:
        DataFrame with engineered features aligned by timeOpen
    """
    logger.info("Creating features...")
    
    df = candles.copy()
    
    # Lag features
    df['close_lag_1'] = df['close'].shift(1)
    df['close_lag_5'] = df['close'].shift(5)
    df['close_lag_15'] = df['close'].shift(15)
    
    # Log returns
    df['log_return'] = np.log(df['close'] / df['close_lag_1'])
    
    # Moving averages
    df['ma_5'] = df['close'].rolling(window=5).mean()
    df['ma_15'] = df['close'].rolling(window=15).mean()
    df['ma_60'] = df['close'].rolling(window=60).mean()
    
    # Volatility (rolling std of returns)
    df['volatility_5'] = df['log_return'].rolling(window=5).std()
    df['volatility_15'] = df['log_return'].rolling(window=15).std()
    df['volatility_60'] = df['log_return'].rolling(window=60).std()
    
    # RSI
    df['rsi_14'] = calculate_rsi(df['close'], period=14)
    
    # MACD
    macd_line, signal_line, histogram = calculate_macd(df['close'])
    df['macd_line'] = macd_line
    df['macd_signal'] = signal_line
    df['macd_histogram'] = histogram
    
    # Bollinger Bands
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(df['close'])
    df['bb_upper'] = bb_upper
    df['bb_middle'] = bb_middle
    df['bb_lower'] = bb_lower
    
    # EMA
    df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
    
    # Select only feature columns (exclude raw candle data)
    feature_cols = [
        'timeOpen',
        'close_lag_1', 'close_lag_5', 'close_lag_15',
        'log_return',
        'ma_5', 'ma_15', 'ma_60',
        'volatility_5', 'volatility_15', 'volatility_60',
        'rsi_14',
        'macd_line', 'macd_signal', 'macd_histogram',
        'bb_upper', 'bb_middle', 'bb_lower',
        'ema_12', 'ema_26'
    ]
    
    features = df[feature_cols].copy()
    
    # Check for missing values
    missing_count = features.isnull().sum().sum()
    logger.info(f"Technical features created with {missing_count} missing values (expected due to lag/rolling)")
    
    return features


def merge_sentiment_features(features: pd.DataFrame, sentiment: pd.DataFrame) -> pd.DataFrame:
    """
    Merge sentiment features with technical features.
    """
    logger.info("Merging sentiment features...")
    
    if sentiment.empty:
        logger.warning("No sentiment data available - adding zero-filled sentiment columns")
        features['sentiment_mean'] = 0.0
        features['sentiment_neg_mean'] = 0.0
        features['relevance_score'] = 0.0
        features['events_count'] = 0.0
        features['negative_spike_flag'] = 0.0
        return features
    
    # Merge on timeOpen
    merged = pd.merge(features, sentiment, on='timeOpen', how='left')
    
    # Forward-fill sentiment values
    sentiment_cols = ['sentiment_mean', 'sentiment_neg_mean', 'relevance_score', 
                     'events_count', 'negative_spike_flag']
    
    for col in sentiment_cols:
        if col in merged.columns:
            merged[col] = merged[col].ffill().bfill().fillna(0.0)
    
    return merged


def merge_orderbook_features(features: pd.DataFrame, orderbook: pd.DataFrame) -> pd.DataFrame:
    """
    Merge orderbook features with technical features.
    """
    logger.info("Merging orderbook features...")
    
    orderbook_cols = ['mid_price', 'spread', 'bid_depth_20', 'ask_depth_20', 
                      'imbalance', 'largest_wall_dist', 'cum_depth_10bps']
                      
    if orderbook.empty:
        logger.warning("No orderbook data available - adding zero-filled columns")
        for col in orderbook_cols:
            features[col] = 0.0
        return features
    
    # Merge on timeOpen
    merged = pd.merge(features, orderbook, on='timeOpen', how='left')
    
    # Forward-fill orderbook values
    for col in orderbook_cols:
        if col in merged.columns:
            merged[col] = merged[col].ffill().bfill().fillna(0.0)
    
    return merged


def merge_macro_features(features: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """
    Merge macro factors with technical features.
    """
    logger.info("Merging macro features...")
    
    macro_cols = ['spx_close', 'dxy_close', 'returns_spx', 'returns_dxy', 'z_spx', 'z_dxy']
                      
    if macro.empty:
        logger.warning("No macro data available - adding zero-filled columns")
        for col in macro_cols:
            features[col] = 0.0
        return features
    
    # Merge on timeOpen
    merged = pd.merge(features, macro, on='timeOpen', how='left')
    
    # Forward-fill macro values (markets close, but the state persists)
    for col in macro_cols:
        if col in merged.columns:
            merged[col] = merged[col].ffill().bfill().fillna(0.0)
    
    logger.info(f"Merged features shape: {merged.shape}")
    non_zero_macro = (merged[macro_cols] != 0).any(axis=1).sum()
    logger.info(f"Macro features coverage: {non_zero_macro}/{len(merged)} rows have data")
    
    return merged


def fit_and_apply_scalers(features: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Fit scalers on training portion and apply to all data.
    
    Args:
        features: DataFrame with engineered features
    
    Returns:
        Tuple of (normalized features DataFrame, scalers dict)
    """
    logger.info("Fitting and applying scalers...")
    
    # Separate timeOpen from features
    time_col = features['timeOpen'].copy()
    feature_cols = [col for col in features.columns if col != 'timeOpen']
    
    # Calculate training split point
    train_size = int(len(features) * TRAIN_SPLIT)
    logger.info(f"Training on first {train_size} rows ({TRAIN_SPLIT*100}%)")
    
    # Get training data (excluding NaN rows)
    train_data = features[feature_cols].iloc[:train_size]
    
    # Fit scalers on training data only
    scalers = {}
    normalized_data = pd.DataFrame(index=features.index)
    
    for col in feature_cols:
        scaler = MinMaxScaler()
        
        # Fit on training data (drop NaN for fitting)
        train_col = train_data[col].dropna().values.reshape(-1, 1)
        
        if len(train_col) > 0:
            scaler.fit(train_col)
            
            # Transform all data
            col_data = features[col].values.reshape(-1, 1)
            normalized_data[col] = scaler.transform(col_data)
            
            scalers[col] = scaler
        else:
            logger.warning(f"Column {col} has no valid training data, skipping")
            normalized_data[col] = features[col]
    
    # Add timeOpen back
    normalized_data.insert(0, 'timeOpen', time_col)
    
    logger.info(f"Fitted {len(scalers)} scalers")
    
    return normalized_data, scalers


def save_outputs(candles: pd.DataFrame, features: pd.DataFrame, 
                 normalized: pd.DataFrame, scalers: dict) -> None:
    """
    Save all output files atomically to prevent race conditions.
    """
    logger.info("Saving outputs atomically...")
    
    output_jobs = [
        (candles, OUTPUT_DATASET),
        (features, OUTPUT_FEATURES),
        (normalized, OUTPUT_NORMALIZED)
    ]
    
    def safe_replace(tmp, target):
        max_retries = 5
        for i in range(max_retries):
            try:
                if os.path.exists(target):
                    os.replace(tmp, target)
                else:
                    os.rename(tmp, target)
                return True
            except PermissionError:
                if i < max_retries - 1:
                    time.sleep(0.5) # Wait for dashboard to release lock
                    continue
                raise
        return False

    for df, target_path in output_jobs:
        tmp_path = target_path + ".tmp"
        df.to_csv(tmp_path, index=False)
        safe_replace(tmp_path, target_path)
        logger.info(f"Updated {target_path}")
    
    # Save scalers
    scaler_tmp = SCALERS_FILE + ".tmp"
    with open(scaler_tmp, 'wb') as f:
        pickle.dump(scalers, f)
    safe_replace(scaler_tmp, SCALERS_FILE)
    logger.info(f"Updated {SCALERS_FILE}")


def main():
    """Main execution function."""
    logger.info("Starting dataset building...")
    
    try:
        # Load and merge candles
        candles = load_candles()
        
        # Load sentiment data
        sentiment = load_sentiment_data()
        
        # Load orderbook data
        orderbook = load_orderbook_data()
        
        # Load macro data
        macro = load_macro_data()
        
        # Create technical features
        features = make_features(candles)
        
        # Merge sentiment features
        features = merge_sentiment_features(features, sentiment)
        
        # Merge orderbook features
        features = merge_orderbook_features(features, orderbook)
        
        # Merge macro features
        features = merge_macro_features(features, macro)
        
        # Fit and apply scalers
        normalized, scalers = fit_and_apply_scalers(features)
        
        # Save outputs
        save_outputs(candles, features, normalized, scalers)
        
        logger.info("Dataset building completed successfully!")
        
        # Summary
        logger.info("\n=== Summary ===")
        logger.info(f"Total candles: {len(candles)}")
        logger.info(f"Date range: {candles['timeOpen'].min()} to {candles['timeOpen'].max()}")
        logger.info(f"Total features created: {len(features.columns) - 1}")  # -1 for timeOpen
        logger.info(f"  - Technical features: 20")
        logger.info(f"  - Sentiment features: 5")
        logger.info(f"  - Orderbook features: 7")
        logger.info(f"  - Macro features: 6")
        logger.info(f"Output files:")
        logger.info(f"  - {OUTPUT_DATASET}")
        logger.info(f"  - {OUTPUT_FEATURES}")
        logger.info(f"  - {OUTPUT_NORMALIZED}")
        logger.info(f"  - {SCALERS_FILE}")
        
    except Exception as e:
        logger.error(f"Error during dataset building: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
