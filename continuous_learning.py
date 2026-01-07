"""
Bitcoin Continuous Learning Script (V2)
Integrates drift detection (KS-test/PSI) and incremental model updates.
"""

import os
import time
import logging
import pickle
import torch
import datetime
import numpy as np
import pandas as pd
import shutil
import xgboost as xgb
from datetime import timezone
from typing import Dict, Any, Tuple

# Project modules
import build_dataset
from data_cleaner import detect_outliers, remove_noise
from drift_detector import check_drift
from train_models import (
    train_baselines, train_deep_models, evaluate, 
    load_and_prepare_data, save_all_models,
    LSTMModel, TransformerModel, WINDOW_SIZE, MODELS_DIR
)

# Configure logging
if not os.path.exists("logs"): os.makedirs("logs")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/continuous_learning.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
DATASET_CSV = "btc_dataset.csv"
FEATURES_NORMALIZED_CSV = "btc_features_normalized.csv"
DRIFT_THRESHOLD_KS = 0.05
DRIFT_THRESHOLD_PSI = 0.2
MONITOR_CSVS = ["btc_live_candles.csv", "sentiment_minute.csv", "orderbook_depth.csv", "macro_factors.csv"]
POLL_INTERVAL_SECONDS = 300 # 5 minutes

class ContinuousLearnerV2:
    def __init__(self):
        self.last_retrain_time = datetime.datetime.now(timezone.utc)
        self.reference_data = None # Baseline distribution
        self.load_reference_data()
        
    def load_reference_data(self):
        """Load the data used for initial training as a reference for drift."""
        try:
            if os.path.exists(FEATURES_NORMALIZED_CSV):
                df = pd.read_csv(FEATURES_NORMALIZED_CSV).iloc[:1000] # Use first 1000 rows as ref
                self.reference_data = df
                logger.info("Reference data loaded for drift detection.")
        except Exception as e:
            logger.error(f"Failed to load reference data: {e}")

    def get_latest_weights(self):
        """Load current best weights for warm-starting."""
        weights = {}
        try:
            for task in ['reg', 'cls']:
                pth_path = os.path.join(MODELS_DIR, f"btc_model_{task}.pth")
                if os.path.exists(pth_path):
                    checkpoint = torch.load(pth_path, map_location=torch.device('cpu'), weights_only=True)
                    model_type = checkpoint.get('model_type')
                    key = f"{model_type}_{task.capitalize()}"
                    weights[key] = checkpoint['state_dict']
            logger.info(f"Loaded {len(weights)} weight sets for warm-start.")
        except Exception as e:
            logger.error(f"Error loading weights: {e}")
        return weights

    def get_latest_baselines(self):
        """Load latest XGBoost models for incremental boosting."""
        models = {}
        try:
            for task in ['reg', 'cls']:
                pkl_path = os.path.join(MODELS_DIR, f"btc_model_{task}.pkl")
                if os.path.exists(pkl_path):
                    with open(pkl_path, 'rb') as f:
                        model = pickle.load(f)
                        if isinstance(model, (xgb.XGBRegressor, xgb.XGBClassifier)):
                            name = f"XGB_{task.capitalize()}"
                            models[name] = model
            logger.info(f"Loaded {len(models)} baseline models for incremental updates.")
        except Exception as e:
            logger.error(f"Error loading baselines: {e}")
        return models

    def run_cycle(self):
        logger.info("Starting monitoring cycle...")
        
        # 1. Check for data updates and rebuild dataset
        # In this implementation, we simply re-run build_dataset.main() 
        # which smartly merges all newer rows from sentiment/candles/etc.
        try:
            build_dataset.main()
            logger.info("Dataset and features refreshed.")
        except Exception as e:
            logger.error(f"Dataset build failed: {e}")
            return

        # 2. Check for Drift
        if self.reference_data is not None:
            try:
                current_data = pd.read_csv(FEATURES_NORMALIZED_CSV).tail(500)
                report, drift_count = check_drift(self.reference_data, current_data, DRIFT_THRESHOLD_KS, DRIFT_THRESHOLD_PSI)
                
                if drift_count > 3: # Threshold: Retrain if 3+ features showed drift
                    logger.warning(f"SIGNIFICANT DRIFT DETECTED ({drift_count} features). Triggering retraining...")
                    self.perform_incremental_retraining()
                else:
                    logger.info(f"Market stable. Drift detected in only {drift_count} features.")
            except Exception as e:
                logger.error(f"Drift detection failed: {e}")
        else:
            self.load_reference_data()

    def perform_incremental_retraining(self):
        logger.info("Initializing incremental retraining...")
        try:
            # Prepare data
            splits, feature_cols = load_and_prepare_data()
            
            # 1. Warm-start Deep Learning
            weights = self.get_latest_weights()
            deep_models, seq_test = train_deep_models(splits, len(feature_cols), existing_weights=weights)
            
            # 2. Incremental Baselines
            baselines = self.get_latest_baselines()
            baseline_results = train_baselines(splits, feature_cols, existing_models=baselines)
            
            # 3. Evaluation & Selection
            metrics = evaluate(baseline_results, deep_models, splits, seq_test)
            logger.info(f"Retraining complete. Metrics:\n{metrics}")
            
            # 4. Save best
            save_all_models(baseline_results, deep_models, metrics, len(feature_cols))
            
            # Update reference data to the new regime
            self.reference_data = pd.read_csv(FEATURES_NORMALIZED_CSV).tail(1000)
            self.last_retrain_time = datetime.datetime.now(timezone.utc)
            logger.info("Models updated and regime reference reset.")
            
        except Exception as e:
            logger.error(f"Retraining cycle failed: {e}", exc_info=True)

    def run(self):
        logger.info("Continuous Learning Service Active.")
        while True:
            try:
                self.run_cycle()
                time.sleep(POLL_INTERVAL_SECONDS)
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Fatal error in loop: {e}")
                time.sleep(60)

if __name__ == "__main__":
    service = ContinuousLearnerV2()
    service.run()
