import pandas as pd
import numpy as np
import os
import yaml
from datetime import datetime, timezone

# Mock data
price = 95000.0
pred_up = {'confidence': 0.8, 'direction': 'UP'}
pred_down = {'confidence': 0.2, 'direction': 'DOWN'}
ob_data = {
    'bids': [[94990, 10.0], [94980, 5.0]],
    'asks': [[95010, 2.0], [95020, 1.0]]
}
macro_data = {'spx_z': -1.5, 'dxy_z': 1.5} # Risk-off
sent_data = pd.DataFrame([{'negative_spike_flag': 1}]) # Sentiment spike

# Mock load_config
def load_config():
    return {
        'signal_engine': {
            'confidence_threshold': 0.65,
            'max_spread': 15.0,
            'min_imbalance': 0.15,
            'persistence_file': "test_signals_v2.csv"
        }
    }

# Logic to test (Copied from binance_dashboard.py)
SIGNALS_V2_CSV = "test_signals_v2.csv"
GLOBAL_CACHE = {'signals': []}

def generate_signal_v2(price, pred_data, ob_data, macro_data, sent_data):
    config = load_config()
    engine_conf = config.get('signal_engine', {})
    conf_thresh = engine_conf.get('confidence_threshold', 0.65)
    max_spread = engine_conf.get('max_spread', 15.0)
    min_imb = engine_conf.get('min_imbalance', 0.15)
    
    p_up = pred_data.get('confidence', 0.5)
    
    bids = ob_data.get('bids', [])
    asks = ob_data.get('asks', [])
    spread = abs(float(asks[0][0]) - float(bids[0][0])) if bids and asks else 999.0
    bid_vol = sum([float(q) for p, q in bids[:10]]) if bids else 0
    ask_vol = sum([float(q) for p, q in asks[:10]]) if asks else 0
    imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol) if (bid_vol + ask_vol) > 0 else 0
    
    sent_spike = 0
    if not sent_data.empty:
        sent_spike = int(sent_data.iloc[-1].get('negative_spike_flag', 0))
    
    macro_regime = "NEUTRAL"
    spx_z = float(macro_data.get('spx_z', 0))
    dxy_z = float(macro_data.get('dxy_z', 0))
    if spx_z < -1.0 and dxy_z > 1.0: macro_regime = "RISK_OFF"
    
    signal = "NEUTRAL"
    reasons = []
    
    if p_up >= conf_thresh:
        if spread <= max_spread and imbalance >= min_imb and sent_spike == 0:
            signal = "BUY/UP"
            reasons.append("High Confidence + Liquidity Support")
        else:
            if spread > max_spread: reasons.append("Spread too wide")
            if imbalance < min_imb: reasons.append("Low buy pressure")
            if sent_spike: reasons.append("Negative sentiment spike")
            signal = "WAIT (Weak UP)"
    elif p_up <= (1 - conf_thresh) or (sent_spike == 1 and macro_regime == "RISK_OFF"):
        signal = "SELL/DOWN"
        if p_up <= (1 - conf_thresh): reasons.append("High Down Confidence")
        if sent_spike and macro_regime == "RISK_OFF": reasons.append("Sentiment Spike + Macro Risk-Off")
        
    return signal, reasons

# Test Case 1: Strong UP (should be filtered by sentiment spike in mock)
sig1, reasons1 = generate_signal_v2(price, pred_up, ob_data, macro_data, sent_data)
print(f"Test 1 (Strong UP with Spike): {sig1} - Reasons: {reasons1}")

# Test Case 2: Strong UP (Clean)
sent_clean = pd.DataFrame([{'negative_spike_flag': 0}])
sig2, reasons2 = generate_signal_v2(price, pred_up, ob_data, macro_data, sent_clean)
print(f"Test 2 (Strong UP Clean): {sig2} - Reasons: {reasons2}")

# Test Case 3: Sentiment Spike + Risk Off (SELL)
sig3, reasons3 = generate_signal_v2(price, {'confidence': 0.5}, ob_data, macro_data, sent_data)
print(f"Test 3 (Sentiment Spike + Risk Off): {sig3} - Reasons: {reasons3}")

# Test Case 4: Weak UP (Filter by Spread)
ob_wide = {'bids': [[94000, 10]], 'asks': [[95000, 10]]}
sig4, reasons4 = generate_signal_v2(price, pred_up, ob_wide, macro_data, sent_clean)
print(f"Test 4 (Wide Spread): {sig4} - Reasons: {reasons4}")
