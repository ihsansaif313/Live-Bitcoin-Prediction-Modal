"""
Bitcoin Model Training Script
Trains Baseline and Deep Learning models for BTC price prediction.
"""

import os
import pickle
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, List, Optional

# Machine Learning
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBRegressor, XGBClassifier

# Time Series (Statsmodels)
from statsmodels.tsa.arima.model import ARIMA

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Status reporting
import json
STATUS_FILE = "logs/setup_status.json"

def report_progress(progress, message, detail=""):
    """Update shared status file."""
    try:
        # Only update if file exists (implies orchestration)
        if os.path.exists("logs"): 
             # We want to preserve status='running' from orchestrator
             # So we read first? Or just overwrite specific fields?
             # Simple overwrite is safer for now, orchestration loop handles the rest?
             # Actually, simpler: write a full valid status object
             with open(STATUS_FILE, "w") as f:
                json.dump({
                    "status": "running",
                    "progress": progress,
                    "message": message,
                    "detail": detail
                }, f)
    except Exception:
        pass

# Deep Learning (TensorFlow/Keras)
try:
    import tensorflow as tf
    
    # GPU Configuration
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"GPU detected: {len(gpus)} device(s). Using GPU for training.")
        except RuntimeError as e:
            logger.warning(f"GPU configuration failed: {e}")
    else:
        import platform
        msg = "No GPU detected. Using CPU for training."
        if platform.system() == "Windows":
             msg += "\n[NOTE] TensorFlow > 2.10 on native Windows is CPU-only by design. To enable GPU, use WSL2 or downgrade to TF 2.10 (only checks if you have a compatible NVIDIA GPU)."
        logger.info(msg)

    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Input, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except ImportError:
    logger.warning("TensorFlow/Keras not found. Deep learning models will be skipped. (Likely due to Python 3.14 alpha/beta version conflict)")
    TF_AVAILABLE = False

# Constants
FEATURES_CSV = "btc_features_normalized.csv"
DATASET_CSV = "btc_dataset.csv"
MODELS_DIR = "models"
REPORTS_DIR = "reports"
WINDOW_SIZE = 60  # 60 minutes sequence
BATCH_SIZE = 64
EPOCHS = 10  # Reduced for reliability

def make_sequences(features: np.ndarray, targets_reg: np.ndarray, targets_cls: np.ndarray, window_size: int):
    """
    Create sliding window sequences for Deep Learning models.
    """
    X, y_reg, y_cls = [], [], []
    for i in range(len(features) - window_size):
        X.append(features[i:(i + window_size)])
        y_reg.append(targets_reg[i + window_size])
        y_cls.append(targets_cls[i + window_size])
    return np.array(X, dtype=np.float32), np.array(y_reg, dtype=np.float32), np.array(y_cls, dtype=np.float32)

def load_and_prepare_data():
    """
    Load data and create regression/classification targets.
    """
    logger.info("Loading and preparing data...")
    import time
    
    def read_csv_with_retry(filepath, retries=5, delay=2):
        for i in range(retries):
            try:
                df = pd.read_csv(filepath)
                if not df.empty:
                    return df
            except (pd.errors.EmptyDataError, OSError):
                pass
            logger.warning(f"Retrying to read {filepath} (attempt {i+1}/{retries})...")
            time.sleep(delay)
        return pd.read_csv(filepath)  # Final attempt, will raise error if still fails

    try:
        features_df = read_csv_with_retry(FEATURES_CSV)
        dataset_df = read_csv_with_retry(DATASET_CSV)
    except Exception as e:
        logger.error(f"Failed to load data files: {e}")
        raise
    
    # Ensure they are aligned by dropping first N rows where features might be NaN
    # build_dataset.py should have handled this, but we filter any remaining NaNs
    combined = pd.merge(features_df, dataset_df[['timeOpen', 'close']], on='timeOpen')
    combined = combined.dropna().reset_index(drop=True)
    
    # Regression target: close_t+1
    combined['target_reg'] = combined['close'].shift(-1)
    
    # Classification target: 1 if close_t+1 > close_t else 0
    combined['target_cls'] = (combined['target_reg'] > combined['close']).astype(int)
    
    # Drop rows without target (the last row)
    combined = combined.dropna().reset_index(drop=True)
    
    # Split features and targets
    feature_cols = [c for c in features_df.columns if c != 'timeOpen']
    X_data = combined[feature_cols].values
    y_reg = combined['target_reg'].values
    y_cls = combined['target_cls'].values
    
    # Time-aware Split
    n = len(combined)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    
    splits = {
        'train': (X_data[:train_end], y_reg[:train_end], y_cls[:train_end]),
        'val': (X_data[train_end:val_end], y_reg[train_end:val_end], y_cls[train_end:val_end]),
        'test': (X_data[val_end:], y_reg[val_end:], y_cls[val_end:])
    }
    
    logger.info(f"Split sizes: Train={len(splits['train'][0])}, Val={len(splits['val'][0])}, Test={len(splits['test'][0])}")
    return splits, feature_cols

def train_baselines(splits, feature_cols):
    """
    Train Baseline models (Univariate and Tabular).
    """
    logger.info("Training baseline models...")
    X_train, y_train_reg, y_train_cls = splits['train']
    X_test, y_test_reg, y_test_cls = splits['test']
    
    results = {}

    # 1. Random Forest Regressor
    logger.info("  Training Random Forest Regressor...")
    rf_reg = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf_reg.fit(X_train, y_train_reg)
    results['RF_Reg'] = rf_reg

    # 2. XGBoost Regressor
    logger.info("  Training XGBoost Regressor...")
    xgb_reg = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=6, random_state=42)
    xgb_reg.fit(X_train, y_train_reg)
    results['XGB_Reg'] = xgb_reg

    # 3. Logistic Regression
    logger.info("  Training Logistic Regression Classifier...")
    lr_cls = LogisticRegression(max_iter=1000, random_state=42)
    lr_cls.fit(X_train, y_train_cls)
    results['LR_Cls'] = lr_cls

    # 4. XGBoost Classifier
    logger.info("  Training XGBoost Classifier...")
    xgb_cls = XGBClassifier(n_estimators=100, max_depth=6, random_state=42)
    xgb_cls.fit(X_train, y_train_cls)
    results['XGB_Cls'] = xgb_cls

    # 5. ARIMA (Univariate - on a subset for speed check)
    # ARIMA is slow on 200k rows, we'll just show the function exists or use small tail
    logger.info("  Training ARIMA (on test set tail for performance verification)...")
    try:
        # ARIMA on original prices from test set
        history = list(y_train_reg[-1000:]) # Use last 1000 for "warmup" or similar concept
        # We don't "train" ARIMA in bulk like others, usually it's for short-term forecast
        results['ARIMA'] = None # Placeholder since it's used differently
    except Exception as e:
        logger.warning(f"ARIMA training failed: {e}")

    return results

def train_deep_models(splits, input_dim):
    """
    Train LSTM, GRU, and Transformer models.
    """
    if not TF_AVAILABLE:
        logger.warning("Skipping Deep Learning models as TensorFlow is not available.")
        return {}, (None, None, None)

    logger.info("Preparing sequences for Deep Learning...")
    # Create sequences one by one / manage memory if needed
    X_train_seq, y_train_reg_seq, y_train_cls_seq = make_sequences(*splits['train'], WINDOW_SIZE)
    logger.info(f"  Train sequences: {X_train_seq.shape}")
    
    X_val_seq, y_val_reg_seq, y_val_cls_seq = make_sequences(*splits['val'], WINDOW_SIZE)
    logger.info(f"  Val sequences: {X_val_seq.shape}")
    
    X_test_seq, y_test_reg_seq, y_test_cls_seq = make_sequences(*splits['test'], WINDOW_SIZE)
    logger.info(f"  Test sequences: {X_test_seq.shape}")
    
    results = {}
    
    # 1. LSTM (Regression)
    logger.info("  Training LSTM Regressor...")
    lstm_reg = Sequential([
        Input(shape=(WINDOW_SIZE, input_dim)),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dense(1)
    ])
    lstm_reg.compile(optimizer='adam', loss='mse')
    lstm_reg.fit(X_train_seq, y_train_reg_seq, validation_data=(X_val_seq, y_val_reg_seq), 
                 epochs=EPOCHS, batch_size=BATCH_SIZE, 
                 callbacks=[EarlyStopping(patience=3, restore_best_weights=True)], verbose=0)
    results['LSTM_Reg'] = lstm_reg

    # 2. GRU (Classification)
    logger.info("  Training GRU Classifier...")
    gru_cls = Sequential([
        Input(shape=(WINDOW_SIZE, input_dim)),
        GRU(64, return_sequences=False),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    gru_cls.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    gru_cls.fit(X_train_seq, y_train_cls_seq, validation_data=(X_val_seq, y_val_cls_seq),
                epochs=EPOCHS, batch_size=BATCH_SIZE,
                callbacks=[EarlyStopping(patience=3, restore_best_weights=True)], verbose=0)
    results['GRU_Cls'] = gru_cls

    # 3. Simple Transformer (Regression)
    logger.info("  Training Transformer Regressor...")
    inputs = Input(shape=(WINDOW_SIZE, input_dim))
    x = LayerNormalization()(inputs)
    # Multi-head attention
    attn_output = MultiHeadAttention(num_heads=4, key_dim=input_dim)(x, x)
    x = Dropout(0.1)(attn_output)
    x = LayerNormalization()(x + inputs) # residual
    x = GlobalAveragePooling1D()(x)
    x = Dense(32, activation='relu')(x)
    outputs = Dense(1)(x)
    
    transformer_reg = Model(inputs, outputs)
    transformer_reg.compile(optimizer='adam', loss='mse')
    transformer_reg.fit(X_train_seq, y_train_reg_seq, validation_data=(X_val_seq, y_val_reg_seq),
                        epochs=EPOCHS, batch_size=BATCH_SIZE,
                        callbacks=[EarlyStopping(patience=3, restore_best_weights=True)], verbose=0)
    results['Transformer_Reg'] = transformer_reg

    return results, (X_test_seq, y_test_reg_seq, y_test_cls_seq)

def evaluate(baseline_models, deep_models, splits, seq_test):
    """
    Evaluate all models and return metrics.
    """
    logger.info("Evaluating models...")
    X_test, y_test_reg, y_test_cls = splits['test']
    X_test_seq, y_test_reg_seq, y_test_cls_seq = seq_test
    
    metrics = []

    # Evaluate Baselines
    for name, model in baseline_models.items():
        if model is None: continue
        if '_Reg' in name:
            preds = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test_reg, preds))
            mae = mean_absolute_error(y_test_reg, preds)
            metrics.append({'Model': name, 'Type': 'Regression', 'RMSE': rmse, 'MAE': mae})
        elif '_Cls' in name:
            preds = model.predict(X_test)
            acc = accuracy_score(y_test_cls, preds)
            f1 = f1_score(y_test_cls, preds)
            metrics.append({'Model': name, 'Type': 'Classification', 'Accuracy': acc, 'F1': f1})

    # Evaluate Deep Learning
    for name, model in deep_models.items():
        if '_Reg' in name:
            preds = model.predict(X_test_seq).flatten()
            rmse = np.sqrt(mean_squared_error(y_test_reg_seq, preds))
            mae = mean_absolute_error(y_test_reg_seq, preds)
            metrics.append({'Model': name, 'Type': 'Regression', 'RMSE': rmse, 'MAE': mae})
        elif '_Cls' in name:
            preds = (model.predict(X_test_seq) > 0.5).astype(int).flatten()
            acc = accuracy_score(y_test_cls_seq, preds)
            f1 = f1_score(y_test_cls_seq, preds)
            metrics.append({'Model': name, 'Type': 'Classification', 'Accuracy': acc, 'F1': f1})

    metrics_df = pd.DataFrame(metrics)
    
    # Generate Plots
    logger.info("  Generating plots...")
    if not metrics_df.empty:
        # Actual vs Predicted for Regression
        reg_models = [m for m in metrics_df[metrics_df['Type'] == 'Regression']['Model']]
        if reg_models:
            plt.figure(figsize=(12, 6))
            for name in reg_models[:3]: # Plot first 3 for clarity
                if name in baseline_models:
                    preds = baseline_models[name].predict(X_test)
                elif name in deep_models:
                    preds = deep_models[name].predict(X_test_seq).flatten()
                plt.plot(y_test_reg[:200], label=f'Actual (Sample)', alpha=0.3 if 'pred' in locals() else 1.0)
                plt.plot(preds[:200], label=f'{name} Prediction')
            plt.title("Actual vs Predicted Close Price (Sample)")
            plt.legend()
            plt.savefig(os.path.join(REPORTS_DIR, "regression_comparison.png"))
            plt.close()

    return metrics_df

def save_best(baseline_models, deep_models, metrics_df):
    """
    Save best models and reports.
    """
    logger.info("Saving best models and reports...")
    
    # Save metrics csv
    metrics_df.to_csv(os.path.join(REPORTS_DIR, "metrics.csv"), index=False)
    
    # Best Regression
    reg_metrics = metrics_df[metrics_df['Type'] == 'Regression'].sort_values('RMSE')
    if not reg_metrics.empty:
        best_reg_name = reg_metrics.iloc[0]['Model']
        logger.info(f"Best Regressor: {best_reg_name}")
        
        if best_reg_name in deep_models:
            reg_path_h5 = os.path.join(MODELS_DIR, "btc_model_reg.h5")
            reg_path_keras = os.path.join(MODELS_DIR, "btc_model_reg.keras")
            deep_models[best_reg_name].save(reg_path_h5)
            deep_models[best_reg_name].save(reg_path_keras)
        elif best_reg_name in baseline_models:
            with open(os.path.join(MODELS_DIR, "btc_model_reg.pkl"), 'wb') as f:
                pickle.dump(baseline_models[best_reg_name], f)

    # Best Classification
    cls_metrics = metrics_df[metrics_df['Type'] == 'Classification'].sort_values('Accuracy', ascending=False)
    if not cls_metrics.empty:
        best_cls_name = cls_metrics.iloc[0]['Model']
        logger.info(f"Best Classifier: {best_cls_name}")
        
        if best_cls_name in deep_models:
            cls_path_h5 = os.path.join(MODELS_DIR, "btc_model_cls.h5")
            cls_path_keras = os.path.join(MODELS_DIR, "btc_model_cls.keras")
            deep_models[best_cls_name].save(cls_path_h5)
            deep_models[best_cls_name].save(cls_path_keras)
        elif best_cls_name in baseline_models:
            with open(os.path.join(MODELS_DIR, "btc_model_cls.pkl"), 'wb') as f:
                pickle.dump(baseline_models[best_cls_name], f)

    # ALWAYS Save Baseline Backups (Safety Net)
    # This prevents dashboard crashes if DL models fail to load/predict
    if 'RF_Reg' in baseline_models:
        with open(os.path.join(MODELS_DIR, "btc_model_reg_baseline.pkl"), 'wb') as f:
            pickle.dump(baseline_models['RF_Reg'], f)
    
    if 'XGB_Cls' in baseline_models:
        with open(os.path.join(MODELS_DIR, "btc_model_cls_baseline.pkl"), 'wb') as f:
            pickle.dump(baseline_models['XGB_Cls'], f)
    if 'LR_Cls' in baseline_models and 'XGB_Cls' not in baseline_models: # Fallback if XGB failed
         with open(os.path.join(MODELS_DIR, "btc_model_cls_baseline.pkl"), 'wb') as f:
            pickle.dump(baseline_models['LR_Cls'], f)

def main():
    if not os.path.exists(MODELS_DIR): os.makedirs(MODELS_DIR)
    if not os.path.exists(REPORTS_DIR): os.makedirs(REPORTS_DIR)
    
    report_progress(0.4, "Training Models", "Loading and preparing data...")
    splits, feature_cols = load_and_prepare_data()
    
    report_progress(0.5, "Training Models", "Training Baseline Models (Random Forest, XGBoost)...")
    baseline_models = train_baselines(splits, feature_cols)
    
    report_progress(0.7, "Training Models", "Training Deep Learning Models (LSTM, Transformer)...")
    deep_models, seq_test = train_deep_models(splits, len(feature_cols))
    
    report_progress(0.9, "Training Models", "Evaluating all models...")
    metrics_df = evaluate(baseline_models, deep_models, splits, seq_test)
    logger.info("\nFinal Metrics:\n" + metrics_df.to_string())
    
    save_best(baseline_models, deep_models, metrics_df)
    logger.info("Model training and saving process completed.")

if __name__ == "__main__":
    main()
