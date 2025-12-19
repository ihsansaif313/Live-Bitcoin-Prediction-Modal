"""
Bitcoin Continuous Learning Script
Periodically updates dataset, retrains models, and handles inference.
"""

import os
import time
import logging
import pickle
import datetime
import numpy as np
import pandas as pd
import shutil
from typing import Dict, Any, Tuple

# Re-use existing modules
from build_dataset import make_features, fit_and_apply_scalers
try:
    from train_models import train_baselines, train_deep_models, evaluate, make_sequences
except ImportError:
    # Use fallback if train_models is not fully importable due to structure
    # But since we are in same dir, it should work.
    pass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("continuous_learning.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
LIVE_CANDLES_CSV = "btc_live_candles.csv"
DATASET_CSV = "btc_dataset.csv"
FEATURES_CSV = "btc_features_normalized.csv"
SCALERS_FILE = "scalers.pkl"
CHECKPOINTS_DIR = "models/checkpoints"
CURRENT_MODELS_DIR = "models"
RETRAIN_INTERVAL_CANDLES = 60  # Retrain every ~1 hour (60 minutes)
POLL_INTERVAL_SECONDS = 60

class ContinuousLearner:
    def __init__(self):
        self.last_processed_time = None
        self.candles_since_retrain = 0
        self.scalers = self._load_scalers()
        
    def _load_scalers(self):
        try:
            with open(SCALERS_FILE, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to load scalers: {e}")
            return None

    def load_new_candles(self) -> pd.DataFrame:
        """
        Check for new candles in btc_live_candles.csv.
        """
        try:
            if not os.path.exists(LIVE_CANDLES_CSV):
                return pd.DataFrame()

            df = pd.read_csv(LIVE_CANDLES_CSV)
            df['timeOpen'] = pd.to_datetime(df['timeOpen'], utc=True)
            
            # Determine last processed time from dataset file if first run
            if self.last_processed_time is None:
                if os.path.exists(DATASET_CSV):
                    existing = pd.read_csv(DATASET_CSV)
                    existing['timeOpen'] = pd.to_datetime(existing['timeOpen'], utc=True)
                    self.last_processed_time = existing['timeOpen'].max()
                else:
                    self.last_processed_time = pd.Timestamp.min.tz_localize('UTC')

            # Filter new rows
            new_data = df[df['timeOpen'] > self.last_processed_time]
            
            if not new_data.empty:
                logger.info(f"Found {len(new_data)} new candles")
                self.last_processed_time = new_data['timeOpen'].max()
                
            return new_data
        except Exception as e:
            logger.error(f"Error loading new candles: {e}")
            return pd.DataFrame()

    def update_dataset_and_features(self, new_candles: pd.DataFrame):
        """
        Append new candles and update features/scalers.
        """
        if new_candles.empty:
            return

        # 1. Append to Dataset
        if os.path.exists(DATASET_CSV):
            new_candles.to_csv(DATASET_CSV, mode='a', header=False, index=False)
        else:
            new_candles.to_csv(DATASET_CSV, index=False)
        
        # 2. Regenerate Features
        # We perform a full regeneration or load tail - for stability, full load is safer but slower.
        # Optimized: Load last N rows + new rows to calculate features correctly
        # But for 'build_dataset.py' re-use, we might just re-run parts of it.
        # To match exact logic, we'll reload full dataset and regenerate.
        # Optimization: In prod, we'd only compute tail.
        
        logger.info("Regenerating features for latest data...")
        full_df = pd.read_csv(DATASET_CSV)
        full_df['timeOpen'] = pd.to_datetime(full_df['timeOpen'], utc=True)
        
        # Calculate features (make_features handles lags/rolling)
        features_df = make_features(full_df)
        
        # 3. Apply Scaling (Using EXISTING scalers, do not refit)
        if self.scalers:
            normalized_data = pd.DataFrame(index=features_df.index)
            normalized_data['timeOpen'] = features_df['timeOpen']
            
            for col, scaler in self.scalers.items():
                if col in features_df.columns:
                    col_data = features_df[col].values.reshape(-1, 1)
                    normalized_data[col] = scaler.transform(col_data)
            
            # Save normalized features
            normalized_data.to_csv(FEATURES_CSV, index=False)
            logger.info("Updated features file")
        else:
            logger.warning("No scalers found, skipping normalization update")

    def retrain_models(self):
        """
        Retrain models on the updated dataset.
        """
        logger.info("Starting scheduled model retraining...")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_dir = os.path.join(CHECKPOINTS_DIR, timestamp)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        try:
            # Re-import to get fresh state/data
            from train_models import load_and_prepare_data, train_baselines, train_deep_models, evaluate, save_best
            
            splits, feature_cols = load_and_prepare_data()
            
            # Baseline
            baselines = train_baselines(splits, feature_cols)
            
            # Deep Learning (will use Python environment capabilities)
            deep_models, seq_test = train_deep_models(splits, len(feature_cols))
            
            # Evaluate
            metrics = evaluate(baselines, deep_models, splits, seq_test)
            logger.info(f"Retraining Metrics:\n{metrics}")
            
            # Save to Checkpoints
            save_best(baselines, deep_models, metrics)
            
            # Copy best models to Current
            for model_file in ["btc_model_reg.pkl", "btc_model_cls.pkl", "btc_model_reg.h5", "btc_model_cls.h5"]:
                src = os.path.join(CURRENT_MODELS_DIR, model_file)
                if os.path.exists(src):
                    dst = os.path.join(checkpoint_dir, model_file)
                    shutil.copy2(src, dst)
                    logger.info(f"Checkpointed {model_file} to {checkpoint_dir}")
            
            logger.info("Retraining complete.")
            
        except Exception as e:
            logger.error(f"Retraining failed: {e}", exc_info=True)

    def predict_next(self) -> Dict[str, Any]:
        """
        Inference function: Predict next minute close and direction.
        """
        try:
            # Load latest features
            df = pd.read_csv(FEATURES_CSV)
            if df.empty: return {}
            
            # Tail needs to be long enough for deep learning window
            # Assuming WINDOW_SIZE=60 from train_models
            window_size = 60
            if len(df) < window_size:
                logger.warning("Not enough data for inference")
                return {}
            
            # Prepare input
            feature_cols = [c for c in df.columns if c != 'timeOpen']
            
            # 1. Baseline Prediction (using last row)
            last_row = df.iloc[-1][feature_cols].values.reshape(1, -1)
            
            baseline_result = {}
            reg_path = os.path.join(CURRENT_MODELS_DIR, "btc_model_reg.pkl")
            if os.path.exists(reg_path):
                with open(reg_path, 'rb') as f:
                    model = pickle.load(f)
                    baseline_result['predicted_close'] = model.predict(last_row)[0]
            
            cls_path = os.path.join(CURRENT_MODELS_DIR, "btc_model_cls.pkl")
            if os.path.exists(cls_path):
                with open(cls_path, 'rb') as f:
                    model = pickle.load(f)
                    baseline_result['predicted_direction'] = "UP" if model.predict(last_row)[0] == 1 else "DOWN"

            return baseline_result
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return {}

    def run(self):
        """
        Main loop.
        """
        logger.info("Starting Continuous Learning Service...")
        while True:
            try:
                # 1. Data Update
                new_candles = self.load_new_candles()
                if not new_candles.empty:
                    self.update_dataset_and_features(new_candles)
                    self.candles_since_retrain += len(new_candles)
                    
                    # 2. Inference
                    prediction = self.predict_next()
                    if prediction:
                        logger.info(f"LATEST PREDICTION: {prediction}")
                
                # 3. Retraining Check
                if self.candles_since_retrain >= RETRAIN_INTERVAL_CANDLES:
                    self.retrain_models()
                    self.candles_since_retrain = 0
                
                time.sleep(POLL_INTERVAL_SECONDS)
                
            except KeyboardInterrupt:
                logger.info("Stopping service...")
                break
            except Exception as e:
                logger.error(f"Loop error: {e}", exc_info=True)
                time.sleep(POLL_INTERVAL_SECONDS)

if __name__ == "__main__":
    learner = ContinuousLearner()
    learner.run()
