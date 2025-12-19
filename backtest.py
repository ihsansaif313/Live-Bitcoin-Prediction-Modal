"""
Bitcoin Backtesting Script
Simulates trading strategies based on trained models.
"""

import os
import pickle
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
FEATURES_CSV = "btc_features_normalized.csv"
DATASET_CSV = "btc_dataset.csv"
MODELS_DIR = "models"
REPORTS_DIR = "reports"
TRANSACTION_COST = 0.001  # 0.1% per trade

def load_data():
    """Load features, prices, and models."""
    logger.info("Loading data for backtest...")
    
    # Load features and prices
    features_df = pd.read_csv(FEATURES_CSV)
    prices_df = pd.read_csv(DATASET_CSV)
    
    # Align data
    combined = pd.merge(features_df, prices_df[['timeOpen', 'close']], on='timeOpen')
    combined = combined.dropna().reset_index(drop=True)
    
    # Models
    models = {}
    try:
        with open(os.path.join(MODELS_DIR, "btc_model_reg.pkl"), 'rb') as f:
            models['Reg'] = pickle.load(f)
        with open(os.path.join(MODELS_DIR, "btc_model_cls.pkl"), 'rb') as f:
            models['Cls'] = pickle.load(f)
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        return None, None
        
    return combined, models

def walk_forward_predict(combined_df: pd.DataFrame, models: Dict[str, Any]) -> pd.DataFrame:
    """
    Simulate predictions on a test set (last 15% of data).
    """
    logger.info("Generating predictions...")
    
    # Use last 15% as test set
    split_idx = int(len(combined_df) * 0.85)
    test_df = combined_df.iloc[split_idx:].copy()
    
    # Prepare features
    feature_cols = [c for c in combined_df.columns if c not in ['timeOpen', 'close']]
    X_test = test_df[feature_cols].values
    
    # Generate predictions
    test_df['Pred_Close_Next'] = models['Reg'].predict(X_test)
    test_df['Pred_Direction_Next'] = models['Cls'].predict(X_test) # 1=Up, 0=Down
    
    # Calculate actual next return (target for simulation)
    # We want to know the return of holding from Close_t to Close_t+1
    test_df['Actual_Return_Next'] = test_df['close'].astype(float).pct_change().shift(-1)
    
    return test_df.dropna()

def simulate_strategy(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Simulate trading strategies.
    
    Strategies:
    1. Buy & Hold: Hold BTC throughout.
    2. Directional: Long if Cls predicts Up.
    3. Threshold: Long if (Pred_Close_Next / Close - 1) > Threshold.
    """
    logger.info("Simulating strategies...")
    
    # 1. Buy & Hold
    # Cumulative return of simply holding
    df['Equity_BuyHold'] = (1 + df['Actual_Return_Next']).cumprod()
    
    # 2. Directional (Classification)
    # Signal: 1 if Up, 0 (Neutral/Cash) if Down
    # Assumption: We trade at Close_t and exit at Close_t+1 (or hold)
    df['Signal_Dir'] = df['Pred_Direction_Next']
    
    # Apply cost: Cost applies when signal changes (0->1 or 1->0) or (1->1 is hold, no cost)
    # Simplified cost: If Position_t != Position_t-1, pay cost.
    # We assume we re-evaluate every minute. If signal same, we hold.
    df['Position_Dir'] = df['Signal_Dir'] # 1 or 0
    df['Trade_Dir'] = df['Position_Dir'].diff().abs().fillna(0) # 1 if trade happened
    
    # Strategy Return: Position * Return - Cost
    df['Ret_Dir'] = (df['Position_Dir'] * df['Actual_Return_Next']) - (df['Trade_Dir'] * TRANSACTION_COST)
    df['Equity_Dir'] = (1 + df['Ret_Dir']).cumprod()
    
    # 3. Threshold (Regression)
    # Signal: Long if regression implies > 0.1% gain
    df['Pred_Ret_Next'] = (df['Pred_Close_Next'] / df['close']) - 1
    df['Signal_Thr'] = (df['Pred_Ret_Next'] > 0.001).astype(int)
    
    df['Position_Thr'] = df['Signal_Thr']
    df['Trade_Thr'] = df['Position_Thr'].diff().abs().fillna(0)
    
    df['Ret_Thr'] = (df['Position_Thr'] * df['Actual_Return_Next']) - (df['Trade_Thr'] * TRANSACTION_COST)
    df['Equity_Thr'] = (1 + df['Ret_Thr']).cumprod()
    
    return {
        'Buy & Hold': df['Equity_BuyHold'],
        'Directional (Cls)': df['Equity_Dir'],
        'Threshold (Reg)': df['Equity_Thr']
    }

def summarize_results(equity_curves: Dict[str, pd.Series]):
    """
    Calculate and save performance metrics.
    """
    logger.info("Summarizing results...")
    
    metrics = []
    
    for name, equity in equity_curves.items():
        # Total Return
        total_ret = (equity.iloc[-1] - 1) * 100
        
        # Max Drawdown
        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max
        max_dd = drawdown.min() * 100
        
        # Sharpe Ratio (assuming 1-min risk free = 0)
        # Annualized Sharpe (assuming 24/7 trading = 525600 mins/year)
        minute_rets = equity.pct_change().dropna()
        sharpe = (minute_rets.mean() / minute_rets.std()) * np.sqrt(525600)
        
        metrics.append({
            'Strategy': name,
            'Total Return %': round(total_ret, 2),
            'Max Drawdown %': round(max_dd, 2),
            'Sharpe Ratio': round(sharpe, 2)
        })
    
    metrics_df = pd.DataFrame(metrics)
    print("\nBacktest Summary:")
    print(metrics_df.to_string(index=False))
    
    # Save CSV
    metrics_df.to_csv(os.path.join(REPORTS_DIR, "backtest_summary.csv"), index=False)
    
    # Plot Equity Curves
    plt.figure(figsize=(12, 6))
    for name, equity in equity_curves.items():
        plt.plot(equity.index, equity.values, label=name)
    
    plt.title("Strategy Equity Curves (Test Set)")
    plt.xlabel("Time (Minutes)")
    plt.ylabel("Equity (Normalized to 1.0)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(REPORTS_DIR, "backtest_equity.png"))
    plt.close()

def main():
    data, models = load_data()
    if data is None or models is None:
        logger.error("Failed to load data/models.")
        return
        
    results_df = walk_forward_predict(data, models)
    if results_df.empty:
        logger.error("No predictions generated.")
        return
        
    equity_curves = simulate_strategy(results_df)
    summarize_results(equity_curves)
    logger.info("Backtest complete.")

if __name__ == "__main__":
    main()
