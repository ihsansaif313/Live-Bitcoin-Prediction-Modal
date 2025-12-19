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
HISTORICAL_CSV = "btc_historical.csv"
LIVE_CSV = "btc_live_candles.csv"
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
        historical = pd.read_csv(HISTORICAL_CSV)
        historical['timeOpen'] = pd.to_datetime(historical['timeOpen'], utc=True)
        logger.info(f"Loaded {len(historical)} historical candles")
        candles_list.append(historical)
    else:
        logger.warning(f"{HISTORICAL_CSV} not found")
    
    # Load live data
    if os.path.exists(LIVE_CSV):
        live = pd.read_csv(LIVE_CSV)
        live['timeOpen'] = pd.to_datetime(live['timeOpen'], utc=True)
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
    logger.info(f"Features created with {missing_count} missing values (expected due to lag/rolling)")
    
    return features


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
    Save all output files.
    
    Args:
        candles: Raw merged candles
        features: Engineered features
        normalized: Normalized features
        scalers: Fitted scalers
    """
    logger.info("Saving outputs...")
    
    # Save raw dataset
    candles.to_csv(OUTPUT_DATASET, index=False)
    logger.info(f"Saved {len(candles)} candles to {OUTPUT_DATASET}")
    
    # Save features
    features.to_csv(OUTPUT_FEATURES, index=False)
    logger.info(f"Saved {len(features)} feature rows to {OUTPUT_FEATURES}")
    
    # Save normalized features
    normalized.to_csv(OUTPUT_NORMALIZED, index=False)
    logger.info(f"Saved {len(normalized)} normalized rows to {OUTPUT_NORMALIZED}")
    
    # Save scalers
    with open(SCALERS_FILE, 'wb') as f:
        pickle.dump(scalers, f)
    logger.info(f"Saved {len(scalers)} scalers to {SCALERS_FILE}")


def main():
    """Main execution function."""
    logger.info("Starting dataset building...")
    
    try:
        # Load and merge candles
        candles = load_candles()
        
        # Create features
        features = make_features(candles)
        
        # Fit and apply scalers
        normalized, scalers = fit_and_apply_scalers(features)
        
        # Save outputs
        save_outputs(candles, features, normalized, scalers)
        
        logger.info("Dataset building completed successfully!")
        
        # Summary
        logger.info("\n=== Summary ===")
        logger.info(f"Total candles: {len(candles)}")
        logger.info(f"Date range: {candles['timeOpen'].min()} to {candles['timeOpen'].max()}")
        logger.info(f"Features created: {len(features.columns) - 1}")  # -1 for timeOpen
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
