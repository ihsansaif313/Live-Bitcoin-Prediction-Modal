"""
Bitcoin Pipeline Orchestrator
Coordinates data collection, training, and visualization components.
"""

import os
import sys
import time
import yaml
import subprocess
import logging
from typing import List, Dict

# Configure logging
if not os.path.exists("logs"):
    os.makedirs("logs")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/orchestrator.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def run_step(script_name: str, description: str):
    """Run a script synchronously and wait for completion."""
    logger.info(f"Running Step: {description} ({script_name})...")
    try:
        subprocess.check_call([sys.executable, script_name])
        logger.info(f"Step {description} completed successfully.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Step {description} failed with return code {e.returncode}")
        sys.exit(1)

def ensure_historical(config: Dict):
    """Ensure historical data and initial dataset exist."""
    dataset_path = config['paths']['dataset']
    if not os.path.exists(dataset_path):
        logger.info("Dataset not found. Starting initialization sequence...")
        run_step("historical_data.py", "Download Historical Data")
        run_step("build_dataset.py", "Build Initial Dataset")
    else:
        logger.info("Dataset found. processing to next step.")

def initial_train(config: Dict):
    """Ensure initial models exist."""
    models_dir = config['paths']['models_dir']
    
    # Check for either .pkl, .h5 or .keras regression model
    reg_pkl = os.path.join(models_dir, "btc_model_reg.pkl")
    reg_h5 = os.path.join(models_dir, "btc_model_reg.h5")
    reg_keras = os.path.join(models_dir, "btc_model_reg.keras")
    
    has_model = os.path.exists(reg_pkl) or os.path.exists(reg_h5) or os.path.exists(reg_keras)
    
    if not os.path.exists(models_dir) or not has_model:
        logger.info("Models not found. Starting initial training...")
        run_step("train_models.py", "Train Initial Models")
    else:
         logger.info("Models found. Skipping initial training.")

def start_process(cmd: List[str], name: str, log_file: str) -> subprocess.Popen:
    """Start a background process."""
    logger.info(f"Starting Service: {name}")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    
    with open(log_file, "a") as f:
        p = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            cwd=os.getcwd(),
            creationflags=subprocess.CREATE_NEW_CONSOLE,
            env=env
        )
    return p

def cleanup_data(config: Dict):
    """Delete all data and models for a fresh start."""
    logger.info("Performing Fresh Start Cleanup...")
    
    # Files to delete
    files = [
        config['paths']['historical_data'],
        config['paths']['dataset'],
        config['paths']['live_candles'],
        "btc_trades_live.csv",
        "btc_features.csv",
        "btc_features_normalized.csv",
        "scalers.pkl",
        os.path.join("reports", "metrics.csv"),
        os.path.join("reports", "backtest_summary.csv"),
        os.path.join("reports", "evaluation_metrics.csv"),
        os.path.join("reports", "backtest_equity.png"),
        os.path.join("reports", "confusion_matrix.png"),
        os.path.join("reports", "actual_vs_predicted.png")
    ]
    
    for f in files:
        if os.path.exists(f):
            try:
                os.remove(f)
                logger.info(f"Deleted {f}")
            except Exception as e:
                logger.error(f"Failed to delete {f}: {e}")

    # Models directory
    models_dir = config['paths']['models_dir']
    if os.path.exists(models_dir):
        try:
            import shutil
            shutil.rmtree(models_dir)
            logger.info(f"Deleted {models_dir}")
        except Exception as e:
            logger.error(f"Failed to delete models dir: {e}")

    # Clear log files
    logs_dir = config['paths']['logs_dir']
    if os.path.exists(logs_dir):
        logger.info("Clearing log files...")
        for log_file in os.listdir(logs_dir):
            if log_file.endswith(".log"):
                log_path = os.path.join(logs_dir, log_file)
                try:
                    with open(log_path, 'w') as f:
                        f.truncate(0)
                    logger.info(f"Cleared {log_file}")
                except Exception as e:
                    logger.error(f"Failed to clear {log_file}: {e}")

import threading
import json

# Status file path
STATUS_FILE = "logs/setup_status.json"

def update_status(status, progress, message, detail=""):
    """Write status to JSON file for dashboard to read."""
    try:
        with open(STATUS_FILE, "w") as f:
            json.dump({
                "status": status,
                "progress": progress,
                "message": message,
                "detail": detail
            }, f)
    except Exception as e:
        logger.error(f"Failed to update status: {e}")

def run_setup_sequence(config: Dict):
    """Run all setup steps in order, updating status."""
    try:
        # 0. Cleanup
        if config['params'].get('fresh_start', False):
            update_status("running", 0.05, "Cleaning up old data", "Deleting previous CSVs and Models...")
            cleanup_data(config)
        
        # 1. Historical Data
        dataset_path = config['paths']['dataset']
        if not os.path.exists(dataset_path):
            update_status("running", 0.1, "Downloading Historical Data", "Fetching 6 months of OHLC data (this may take several minutes)...")
            run_step("historical_data.py", "Download Historical Data")
            
            # New Step: Data Cleaning
            update_status("running", 0.2, "Cleaning Data", "Detecting outliers and removing noise...")
            from data_cleaner import clean_historical_data
            clean_historical_data(config['paths']['historical_data'], config['paths']['historical_data_clean'])
            
            update_status("running", 0.3, "Building Dataset", "Generating technical indicators and features...")
            run_step("build_dataset.py", "Build Initial Dataset")
        
        # 2. Model Training
        models_dir = config['paths']['models_dir']
        # Check for either .pkl, .h5 or .keras regression model
        reg_pkl = os.path.join(models_dir, "btc_model_reg.pkl")
        reg_h5 = os.path.join(models_dir, "btc_model_reg.h5")
        reg_keras = os.path.join(models_dir, "btc_model_reg.keras")
        
        has_model = os.path.exists(reg_pkl) or os.path.exists(reg_h5) or os.path.exists(reg_keras)
        
        if not os.path.exists(models_dir) or not has_model:
            update_status("running", 0.4, "Training Models", "Initializing training process...")
            run_step("train_models.py", "Train Initial Models")
            
        update_status("complete", 1.0, "Setup Complete", "Launching live services...")
        
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        update_status("error", 0.0, "Setup Failed", str(e))
        # We don't exit here so the dashboard can verify the error file
        pass

def main():
    config = load_config()
    logs_dir = config['paths']['logs_dir']
    processes: List[subprocess.Popen] = []

    # Reset status file
    update_status("running", 0.0, "Initializing", "Starting Orchestrator...")

    # 1. Start Dashboard IMMEDIATELY
    streamlit_cmd = [sys.executable, "-m", "streamlit", "run", "btc_dashboard.py"]
    p_dash = start_process(
        streamlit_cmd,
        "Dashboard",
        f"{logs_dir}/dashboard.log"
    )
    processes.append(p_dash)

    # 2. Run Setup in Background Thread
    setup_thread = threading.Thread(target=run_setup_sequence, args=(config,))
    setup_thread.start()
    
    # 3. Wait for Setup to Complete
    while setup_thread.is_alive():
        time.sleep(1)
    
    # Check if setup succeeded
    try:
        with open(STATUS_FILE, "r") as f:
            status = json.load(f)
        if status.get("status") == "error":
            logger.error("Setup sequence failed. Stopping.")
            p_dash.terminate()
            return
    except:
        pass

    logger.info("Setup finished. Starting backend services...")

    try:
        # 4. Start Live Stream
        p_stream = start_process(
            [sys.executable, "live_stream.py"],
            "Live Stream",
            f"{logs_dir}/live_stream.log"
        )
        processes.append(p_stream)
        time.sleep(2) 
        
        # 5. Start Aggregator
        p_agg = start_process(
            [sys.executable, "aggregate_live_to_candles.py"],
            "Aggregator",
            f"{logs_dir}/aggregator.log"
        )
        processes.append(p_agg)
        
        # 6. Start Continuous Learner
        p_brain = start_process(
            [sys.executable, "continuous_learning.py"],
            "Continuous Learner",
            f"{logs_dir}/learner.log"
        )
        processes.append(p_brain)
        
        logger.info("All services started. Press Ctrl+C to shutdown.")
        
        # Monitor Loop
        while True:
            for p in processes:
                if p.poll() is not None:
                    # Dashboard might be closed by user, don't scream about it
                    if p != p_dash: 
                        logger.warning(f"Process {p.args} exited unexpectedly with code {p.returncode}")
            time.sleep(5)
            
    except KeyboardInterrupt:
        logger.info("Shutdown signal received. Terminating processes...")
    finally:
        for p in processes:
            if p.poll() is None:
                p.terminate()
        logger.info("System shutdown complete.")

if __name__ == "__main__":
    main()
