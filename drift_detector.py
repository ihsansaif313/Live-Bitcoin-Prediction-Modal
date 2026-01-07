"""
Drift Detection Utility
Implements Kolmogorov-Smirnov (KS) test and Population Stability Index (PSI).
"""

import numpy as np
import pandas as pd
from scipy import stats
import logging

logger = logging.getLogger(__name__)

def calculate_psi(expected, actual, buckets=10):
    """
    Calculate the Population Stability Index (PSI) between two distributions.
    
    PSI = sum((Actual % - Expected %) * ln(Actual % / Expected %))
    """
    def scale_range(input_data, min_val, max_val):
        return (input_data - min_val) / (max_val - min_val) if max_val > min_val else input_data

    # Handle empty or uniform arrays
    if len(expected) == 0 or len(actual) == 0:
        return 0.0
        
    # Define buckets based on expected data
    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf
    
    expected_percents = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_percents = np.histogram(actual, bins=breakpoints)[0] / len(actual)
    
    # Avoid division by zero
    expected_percents = np.clip(expected_percents, 0.0001, 1.0)
    actual_percents = np.clip(actual_percents, 0.0001, 1.0)
    
    psi_value = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
    return psi_value

def check_drift(reference_df, current_df, threshold_ks=0.05, threshold_psi=0.2):
    """
    Check for feature drift between reference and current dataframes.
    
    Returns:
        dict: {feature_name: {'ks_p_value': float, 'psi': float, 'drift_detected': bool}}
    """
    drift_report = {}
    features = [c for c in reference_df.columns if c != 'timeOpen']
    
    drift_count = 0
    for feature in features:
        if feature not in current_df.columns:
            continue
            
        ref_vals = reference_df[feature].dropna()
        cur_vals = current_df[feature].dropna()
        
        if len(ref_vals) < 50 or len(cur_vals) < 50:
            continue
            
        # 1. KS Test (Low p-value = different distribution)
        ks_stat, ks_p = stats.ks_2samp(ref_vals, cur_vals)
        
        # 2. PSI
        psi_val = calculate_psi(ref_vals, cur_vals)
        
        # Drift logic: trigger if KS is low AND PSI is high (significant change)
        # Or just one of them depending on sensitivity requirements
        detected = (ks_p < threshold_ks) or (psi_val > threshold_psi)
        
        drift_report[feature] = {
            'ks_p': float(ks_p),
            'psi': float(psi_val),
            'detected': bool(detected)
        }
        
        if detected:
            drift_count += 1
            logger.warning(f"Drift detected in {feature}: p-val={ks_p:.4f}, PSI={psi_val:.4f}")
            
    return drift_report, drift_count

if __name__ == "__main__":
    # Test stub
    ref = pd.DataFrame({'f1': np.random.normal(0, 1, 1000)})
    cur = pd.DataFrame({'f1': np.random.normal(0.5, 1.2, 1000)})
    report, count = check_drift(ref, cur)
    print(f"Drift detected in {count} features.")
    print(report['f1'])
