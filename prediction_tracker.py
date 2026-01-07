
import pandas as pd
import os
import time
from datetime import datetime, timedelta

PREDICTION_HISTORY_FILE = "prediction_history.csv"

def init_tracker(overwrite=False):
    """Initialize the prediction history file. if overwrite=True, deletes any existing file."""
    if overwrite and os.path.exists(PREDICTION_HISTORY_FILE):
        try: os.remove(PREDICTION_HISTORY_FILE)
        except: pass

    if not os.path.exists(PREDICTION_HISTORY_FILE):
        df = pd.DataFrame(columns=[
            "timestamp", "current_price", "predicted_direction", 
            "predicted_price", "actual_price_15m", "actual_direction", "is_correct"
        ])
        df.to_csv(PREDICTION_HISTORY_FILE, index=False)

def log_prediction(current_price, predicted_price, predicted_direction):
    """
    Log a new prediction.
    timestamp: prediction time (now)
    current_price: price at prediction time
    predicted_price: target price in 15m
    predicted_direction: "UP" or "DOWN"
    """
    if not os.path.exists(PREDICTION_HISTORY_FILE):
        init_tracker()
        
    
    # Simple directory-based lock (atomic on Windows/Linux)
    lock_path = "prediction.lock"
    have_lock = False
    try:
        os.mkdir(lock_path)
        have_lock = True
    except FileExistsError:
        return False # Locked by another thread/process, skip logging (debounce behavior)
    except Exception:
        return False
        
    try:
        # Check for duplicates (last log within 60s)
        if os.path.exists(PREDICTION_HISTORY_FILE):
            try:
                # Read only tail efficiently? For CSV, we verify last line
                df_verify = pd.read_csv(PREDICTION_HISTORY_FILE)
                if not df_verify.empty:
                    last_time_str = str(df_verify.iloc[-1]['timestamp'])
                    last_time = pd.Timestamp(last_time_str).to_pydatetime()
                    if last_time.tzinfo is None:
                        last_time = last_time.replace(tzinfo=None) 
                        
                    time_diff = datetime.utcnow() - last_time
                    if time_diff.total_seconds() < 55: # Buffer slightly less than 60s
                        return False # Duplicate
            except Exception: 
                # If read fails, DO NOT LOG blindly. It might be corrupt.
                # Let update_outcomes clean it up later.
                return False 

        new_row = {
            "timestamp": datetime.utcnow().isoformat(),
            "current_price": current_price,
            "predicted_direction": predicted_direction,
            "predicted_price": predicted_price,
            "actual_price_15m": None,
            "actual_direction": None,
            "is_correct": None
        }
        
        # Append to CSV efficiently
        df = pd.DataFrame([new_row])
        # Ensure timestamp is ISO string
        df['timestamp'] = df['timestamp'].apply(str) 
        df.to_csv(PREDICTION_HISTORY_FILE, mode='a', header=False, index=False)
        return True
    except Exception as e:
        print(f"Error logging prediction: {e}")
        return False
    finally:
        if have_lock:
            try: os.rmdir(lock_path)
            except: pass

def update_outcomes(current_price):
    """
    Check pending predictions that are 15+ minutes old and update them with actual outcomes.
    Returns the number of updated predictions.
    """
    if not os.path.exists(PREDICTION_HISTORY_FILE):
        return 0
        
    # Lock Check
    lock_path = "prediction.lock"
    have_lock = False
    try:
        os.mkdir(lock_path)
        have_lock = True
    except FileExistsError:
        return 0 # Busy
    except Exception:
        return 0

    try:
        df = pd.read_csv(PREDICTION_HISTORY_FILE)
        if df.empty:
            return 0
            
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        now = datetime.utcnow()
        updated_count = 0
        
        # Find pending predictions (where is_correct is NaN/None) older than 15 mins
        # cutoff_time = now - timedelta(minutes=15)
        # Strictly > 15m
        cutoff_time = now - timedelta(minutes=15)
        
        # Ensure naive comparison
        if df['timestamp'].dt.tz is not None:
             df['timestamp'] = df['timestamp'].dt.tz_localize(None)

        mask = (df['is_correct'].isna()) & (df['timestamp'] <= cutoff_time)
        
        if not df.loc[mask].empty:
            # Calculate outcomes
            for idx in df[mask].index:
                pred_price_at_time = df.loc[idx, 'current_price']
                predicted_dir = df.loc[idx, 'predicted_direction']
                
                # Determine actual direction over the 15m interval
                actual_dir = "UP" if current_price > pred_price_at_time else "DOWN"
                
                # Check if correct (Directional Accuracy)
                is_correct = (predicted_dir == actual_dir)
                
                df.loc[idx, 'actual_price_15m'] = current_price
                df.loc[idx, 'actual_direction'] = actual_dir
                df.loc[idx, 'is_correct'] = is_correct
                updated_count += 1
            
            if updated_count > 0:
                # Ensure timestamps are written back as ISO strings (with T separator) to match log_prediction
                df['timestamp'] = df['timestamp'].apply(lambda x: x.isoformat() if hasattr(x, 'isoformat') else str(x))
                df.to_csv(PREDICTION_HISTORY_FILE, index=False)
                
        return updated_count
    except (pd.errors.EmptyDataError, pd.errors.ParserError, KeyError) as e:
        print(f"Error reading prediction history (corrupt/schema mismatch): {e}. Resetting file.")
        init_tracker()
        return 0
    except Exception as e:
        print(f"Error updating outcomes: {e}")
        return 0
    finally:
        if have_lock:
            try: os.rmdir(lock_path)
            except: pass

def get_accuracy_metrics():
    """
    Calculate accuracy metrics.
    Returns a dict with overall accuracy, recent history, etc.
    """
    if not os.path.exists(PREDICTION_HISTORY_FILE):
        return {"overall_accuracy": 0, "total_predictions": 0, "recent_history": []}
        
    try:
        df = pd.read_csv(PREDICTION_HISTORY_FILE)
        # Filter only completed predictions
        completed = df.dropna(subset=['is_correct'])
        
        if completed.empty:
            return {"overall_accuracy": 0, "total_predictions": 0, "recent_history": []}
            
        total = len(completed)
        correct = completed['is_correct'].sum() # True counts as 1
        accuracy = (correct / total) * 100
        
        # Recent history (last 10)
        recent = completed.sort_values('timestamp', ascending=False).head(10)
        recent_history = []
        for _, row in recent.iterrows():
            recent_history.append({
                'time': row['timestamp'],
                'predicted': row['predicted_direction'],
                'actual': row['actual_direction'],
                'correct': bool(row['is_correct'])
            })
            
        return {
            "overall_accuracy": round(accuracy, 1),
            "total_predictions": total,
            "recent_history": recent_history
        }
    except (pd.errors.EmptyDataError, pd.errors.ParserError, KeyError) as e:
        print(f"Error reading prediction history (corrupt/schema mismatch): {e}. Resetting file.")
        init_tracker()
        return {"overall_accuracy": 0, "total_predictions": 0, "recent_history": []}
    except Exception as e:
        print(f"Error calculating accuracy: {e}")
        return {"overall_accuracy": 0, "total_predictions": 0, "recent_history": []}
