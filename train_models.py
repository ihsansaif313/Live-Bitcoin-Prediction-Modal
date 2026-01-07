"""
Bitcoin Model Training Script
Trains Baseline (SKLearn/XGB) and State-of-the-Art Deep Learning (PyTorch) models for BTC price prediction.
"""

import os
import pickle
import logging
import gc
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from typing import Tuple, Dict, List, Optional
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, f1_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# Status reporting
import json
STATUS_FILE = "logs/setup_status.json"

def report_progress(progress, message, detail=""):
    try:
        if os.path.exists("logs"):
            with open(STATUS_FILE, "w") as f:
                json.dump({
                    "status": "running",
                    "progress": progress,
                    "message": message,
                    "detail": detail
                }, f)
    except Exception:
        pass

# Constants
FEATURES_CSV = "btc_features_normalized.csv"
DATASET_CSV = "btc_dataset.csv"
MODELS_DIR = "models"
REPORTS_DIR = "reports"
# Defaults (overridden by config in main)
WINDOW_SIZE = 180 
BATCH_SIZE = 128
EPOCHS = 15
LEARNING_RATE = 0.001

# --- PyTorch Model Definitions ---

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=3, output_dim=1, task='regression'):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.task = task

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        if self.task == 'classification':
            out = torch.sigmoid(out)
        return out

class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=8, num_layers=3, output_dim=1, task='regression'):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=256, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, output_dim)
        self.task = task

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        x = self.fc(x[:, -1, :]) # Take last sequence element
        if self.task == 'classification':
            x = torch.sigmoid(x)
        return x

# --- Helper Functions ---

class TimeSeriesDataset(Dataset):
    def __init__(self, features, targets_reg, targets_cls, window_size):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets_reg = torch.tensor(targets_reg, dtype=torch.float32)
        self.targets_cls = torch.tensor(targets_cls, dtype=torch.float32)
        self.window_size = window_size
        self.length = len(features) - window_size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Lazy slicing: Creating the sequence only when requested
        # X: (window_size, num_features)
        X = self.features[idx : idx + self.window_size]
        # y: scalar
        y_reg = self.targets_reg[idx + self.window_size]
        y_cls = self.targets_cls[idx + self.window_size]
        return X, y_reg, y_cls

def load_and_prepare_data():
    logger.info("Loading and preparing data...")
    # Load Config for threshold
    config = {}
    if os.path.exists("config.yaml"):
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
    
    threshold = config.get('params', {}).get('min_target_return', 0.001)
    logger.info(f"Using classification threshold: {threshold*100:.2f}%")

    try:
        features_df = pd.read_csv(FEATURES_CSV, on_bad_lines='skip', low_memory=False)
        dataset_df = pd.read_csv(DATASET_CSV, on_bad_lines='skip', low_memory=False)
    except Exception as e:
        logger.error(f"Failed to load data files: {e}")
        raise
    combined = pd.merge(features_df, dataset_df[['timeOpen', 'close']], on='timeOpen')
    combined = combined.dropna().reset_index(drop=True)
    combined['target_reg'] = (combined['close'].shift(-15) - combined['close']) / combined['close']
    combined['target_reg'] = combined['target_reg'].clip(-0.10, 0.10)
    
    # Classification logic: Label as 1 (UP) ONLY if return > threshold
    # Moves within [-threshold, threshold] are treated as 0 (NO BIAS/STAY)
    combined['target_cls'] = (combined['target_reg'] >= threshold).astype(int)
    
    combined = combined.dropna().reset_index(drop=True)
    feature_cols = [c for c in features_df.columns if c != 'timeOpen']
    X_data = combined[feature_cols].values
    y_reg = combined['target_reg'].values
    y_cls = combined['target_cls'].values
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

def train_pytorch_model(model, train_loader, val_loader, model_name="model", task='regression', existing_weights=None):
    if existing_weights:
        try:
            model.load_state_dict(existing_weights)
            logger.info(f"Warm-starting {model_name} with existing weights...")
        except:
            logger.warning(f"Could not load existing weights for {model_name} - training from scratch.")
    model.to(device)
    criterion = nn.MSELoss() if task == 'regression' else nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    temp_file = f"best_temp_{model_name}.pth"
    best_val_loss = float('inf')
    patience = 5
    counter = 0
    
    # Pre-save initial state in case training fails early
    torch.save(model.state_dict(), temp_file)
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device).unsqueeze(1)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        logger.info(f"{model_name} - Epoch {epoch + 1}/{EPOCHS}, Loss: {train_loss / len(train_loader):.6f}, Val Loss: {val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), temp_file)
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                logger.info(f"Early stopping {model_name} at epoch {epoch}")
                break
    
    try:
        model.load_state_dict(torch.load(temp_file, weights_only=True))
        if os.path.exists(temp_file): os.remove(temp_file)
    except Exception as e:
        logger.error(f"Error loading best {model_name}: {e}")
        
    return model

# --- Main Training Logic ---

def train_baselines(splits, feature_cols, existing_models=None):
    X_train, y_reg_train, y_cls_train = splits['train']
    models = {}
    
    logger.info("  Training XGBoost Regressor...")
    xgb_reg = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
    xgb_model_reg = None
    if existing_models and 'XGB_Reg' in existing_models:
        xgb_model_reg = existing_models['XGB_Reg']
    xgb_reg.fit(X_train, y_reg_train, xgb_model=xgb_model_reg)
    models['XGB_Reg'] = xgb_reg
    
    logger.info("  Training Random Forest Regressor...")
    rf_reg = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf_reg.fit(X_train, y_reg_train)
    models['RF_Reg'] = rf_reg
    
    logger.info("  Training XGBoost Classifier...")
    xgb_cls = xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42, eval_metric='logloss')
    xgb_model_cls = None
    if existing_models and 'XGB_Cls' in existing_models:
        xgb_model_cls = existing_models['XGB_Cls']
    xgb_cls.fit(X_train, y_cls_train, xgb_model=xgb_model_cls)
    models['XGB_Cls'] = xgb_cls
    
    return models

def train_deep_models(splits, input_dim, existing_weights=None):
    if not existing_weights: existing_weights = {}
    logger.info("Preparing using Lazy Loading Dataset for PyTorch...")
    
    # Use custom dataset for memory efficiency
    train_dataset = TimeSeriesDataset(*splits['train'], WINDOW_SIZE)
    val_dataset = TimeSeriesDataset(*splits['val'], WINDOW_SIZE)
    test_dataset = TimeSeriesDataset(*splits['test'], WINDOW_SIZE)
    
    # Create DataLoaders (workers=0 for Windows compatibility)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Get a sample for dimension check
    sample_X, _, _ = train_dataset[0]
    seq_len, n_feats = sample_X.shape
    logger.info(f"Input shape: ({seq_len}, {n_feats})")
    
    models = {}
    
    # Train LSTM Regressor
    # Note: Dataset returns (X, y_reg, y_cls). We need to adapt the loader loop or the training function 
    # slightly, but train_pytorch_model expects (X, y) pairs.
    # We will wrap the loaders to yield the correct target.
    
    class TaskWrapper:
        def __init__(self, loader, task_idx):
            self.loader = loader
            self.task_idx = task_idx # 1 for reg, 2 for cls
        def __iter__(self):
            for batch in self.loader:
                yield batch[0], batch[self.task_idx]
        def __len__(self):
            return len(self.loader)

    logger.info("  Training PyTorch LSTM Regressor...")
    lstm_reg = LSTMModel(input_dim=input_dim, task='regression')
    train_loader_reg = TaskWrapper(train_loader, 1) # Yields X, y_reg
    val_loader_reg = TaskWrapper(val_loader, 1)
    models['LSTM_Reg'] = train_pytorch_model(lstm_reg, train_loader_reg, val_loader_reg, "LSTM_Reg", 'regression', existing_weights.get('LSTM_Reg'))
    
    logger.info("  Training PyTorch Transformer Regressor...")
    trans_reg = TransformerModel(input_dim=input_dim, task='regression')
    models['Transformer_Reg'] = train_pytorch_model(trans_reg, train_loader_reg, val_loader_reg, "Transformer_Reg", 'regression', existing_weights.get('Transformer_Reg'))
    
    logger.info("  Training PyTorch LSTM Classifier...")
    lstm_cls = LSTMModel(input_dim=input_dim, task='classification')
    train_loader_cls = TaskWrapper(train_loader, 2) # Yields X, y_cls
    val_loader_cls = TaskWrapper(val_loader, 2)
    models['LSTM_Cls'] = train_pytorch_model(lstm_cls, train_loader_cls, val_loader_cls, 'LSTM_Cls', 'classification', existing_weights.get('LSTM_Cls'))

    logger.info("  Training PyTorch Transformer Classifier...")
    trans_cls = TransformerModel(input_dim=input_dim, task='classification')
    models['Transformer_Cls'] = train_pytorch_model(trans_cls, train_loader_cls, val_loader_cls, 'Transformer_Cls', 'classification', existing_weights.get('Transformer_Cls'))

    # Prepare test data for evaluation
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Extract test sequences and targets for evaluation function
    X_test_seq_list = []
    y_test_reg_seq_list = []
    y_test_cls_seq_list = []
    for batch_X, batch_y_reg, batch_y_cls in test_loader:
        X_test_seq_list.append(batch_X.cpu().numpy())
        y_test_reg_seq_list.append(batch_y_reg.cpu().numpy())
        y_test_cls_seq_list.append(batch_y_cls.cpu().numpy())
    
    X_test_seq = np.concatenate(X_test_seq_list, axis=0)
    y_test_reg_seq = np.concatenate(y_test_reg_seq_list, axis=0)
    y_test_cls_seq = np.concatenate(y_test_cls_seq_list, axis=0)

    return models, (X_test_seq, y_test_reg_seq, y_test_cls_seq)

def evaluate(baseline_models, deep_models, splits, seq_test):
    logger.info("Evaluating models...")
    X_test_tab, y_test_reg_tab, y_test_cls_tab = splits['test']
    X_test_seq, y_test_reg_seq, y_test_cls_seq = seq_test
    metrics = []
    
    for name, model in baseline_models.items():
        if '_Reg' in name:
            preds = model.predict(X_test_tab)
            rmse = np.sqrt(mean_squared_error(y_test_reg_tab, preds))
            metrics.append({'Model': name, 'Type': 'Regression', 'RMSE': rmse})
        elif '_Cls' in name:
            preds = model.predict(X_test_tab)
            acc = accuracy_score(y_test_cls_tab, preds)
            metrics.append({'Model': name, 'Type': 'Classification', 'Accuracy': acc})
            
    for name, model in deep_models.items():
        model.eval()
        all_preds = []
        test_dataset = TensorDataset(torch.tensor(X_test_seq).to(torch.float32), torch.tensor(y_test_reg_seq).to(torch.float32))
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
        
        with torch.no_grad():
            for batch_X, _ in test_loader:
                batch_X = batch_X.to(device)
                outputs = model(batch_X).cpu().numpy().flatten()
                all_preds.extend(outputs)
        
        preds = np.array(all_preds)
        if '_Reg' in name:
            rmse = np.sqrt(mean_squared_error(y_test_reg_seq, preds))
            metrics.append({'Model': name, 'Type': 'Regression', 'RMSE': rmse})
        elif '_Cls' in name:
            preds_bin = (preds > 0.5).astype(int)
            acc = accuracy_score(y_test_cls_seq, preds_bin)
            metrics.append({'Model': name, 'Type': 'Classification', 'Accuracy': acc})
    return pd.DataFrame(metrics)

def safe_replace(tmp, target):
    max_retries = 5
    for i in range(max_retries):
        try:
            os.replace(tmp, target)
            return True
        except PermissionError:
            if i < max_retries - 1:
                time.sleep(0.5)
                continue
            raise
    return False

def save_all_models(baseline_models, deep_models, metrics_df, input_dim):
    logger.info("Saving all models for consensus...")
    if not os.path.exists(MODELS_DIR): os.makedirs(MODELS_DIR)
    
    # Save specific types for ensembling
    # 1. LSTM Reg
    if 'LSTM_Reg' in deep_models:
        torch.save({
            'state_dict': deep_models['LSTM_Reg'].state_dict(),
            'model_type': 'LSTM', 'input_dim': input_dim, 'task': 'regression'
        }, os.path.join(MODELS_DIR, "btc_lstm_reg.pth"))
    
    # 2. Transformer Reg
    if 'Transformer_Reg' in deep_models:
        torch.save({
            'state_dict': deep_models['Transformer_Reg'].state_dict(),
            'model_type': 'Transformer', 'input_dim': input_dim, 'task': 'regression'
        }, os.path.join(MODELS_DIR, "btc_trans_reg.pth"))
        
    # 3. XGBoost Reg
    if 'XGB_Reg' in baseline_models:
        with open(os.path.join(MODELS_DIR, "btc_xgb_reg.pkl"), 'wb') as f:
            pickle.dump(baseline_models['XGB_Reg'], f)

    # 4. Standard "Best" targets for backward compatibility
    # Best Regression
    reg_metrics = metrics_df[metrics_df['Type'] == 'Regression'].sort_values('RMSE')
    if not reg_metrics.empty:
        best_name = reg_metrics.iloc[0]['Model']
        if best_name in deep_models:
            torch.save({
                'state_dict': deep_models[best_name].state_dict(),
                'model_type': 'LSTM' if 'LSTM' in best_name else 'Transformer',
                'input_dim': input_dim, 'task': 'regression'
            }, os.path.join(MODELS_DIR, "btc_model_reg.pth"))
        else:
            with open(os.path.join(MODELS_DIR, "btc_model_reg.pkl"), 'wb') as f:
                pickle.dump(baseline_models[best_name], f)
                 
    # Best Classification
    cls_metrics = metrics_df[metrics_df['Type'] == 'Classification'].sort_values('Accuracy', ascending=False)
    if not cls_metrics.empty:
        best_name = cls_metrics.iloc[0]['Model']
        if best_name in deep_models:
            torch.save({
                'state_dict': deep_models[best_name].state_dict(),
                'model_type': 'LSTM' if 'LSTM' in best_name else 'Transformer',
                'input_dim': input_dim, 'task': 'classification'
            }, os.path.join(MODELS_DIR, "btc_model_cls.pth"))
        else:
            with open(os.path.join(MODELS_DIR, "btc_model_cls.pkl"), 'wb') as f:
                pickle.dump(baseline_models[best_name], f)

def main():
    # Load config for global parameters
    config = {}
    if os.path.exists("config.yaml"):
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
    
    global WINDOW_SIZE
    WINDOW_SIZE = config.get('params', {}).get('window_size', 180)
    logger.info(f"Using Window Size: {WINDOW_SIZE} minutes")

    report_progress(0.4, "Training Models", "Preparing data...")
    splits, feature_cols = load_and_prepare_data()
    baseline_models = train_baselines(splits, feature_cols)
    deep_models, seq_test = train_deep_models(splits, len(feature_cols))
    metrics_df = evaluate(baseline_models, deep_models, splits, seq_test)
    logger.info("\nMetrics:\n" + metrics_df.to_string())
    save_all_models(baseline_models, deep_models, metrics_df, len(feature_cols))
    # Cleanup any leftover pth.tmp or temp files
    for f in os.listdir('.'):
        if f.startswith('best_temp_') and f.endswith('.pth'):
            try: os.remove(f)
            except: pass

if __name__ == "__main__":
    main()
