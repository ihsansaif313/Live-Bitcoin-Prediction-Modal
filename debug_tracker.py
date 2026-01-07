
import pandas as pd
from prediction_tracker import update_outcomes, PREDICTION_HISTORY_FILE
from datetime import datetime, timedelta
import os

def inspect_tracker():
    if not os.path.exists(PREDICTION_HISTORY_FILE):
        print("prediction_history.csv does not exist.")
        return

    print("--- Reading CSV ---")
    df = pd.read_csv(PREDICTION_HISTORY_FILE)
    print(df.head(10))
    print("\n--- DataFrame Info ---")
    print(df.info())
    
    # Check naive vs aware timestamps
    print("\n--- Timestamp Check ---")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(df['timestamp'].head())
    
    now = datetime.utcnow()
    cutoff = now - timedelta(minutes=15)
    print(f"\nCurrent UTC: {now}")
    print(f"Cutoff (15m ago): {cutoff}")
    
    mask = (df['is_correct'].isna()) & (df['timestamp'] <= cutoff)
    pending = df[mask]
    
    print(f"\nPending Updates Found: {len(pending)}")
    if not pending.empty:
        print(pending[['timestamp', 'current_price', 'predicted_direction']])
        
        # Test update
        print("\n--- Testing Update ---")
        # Simulate a price slightly varying
        current_price = df.iloc[0]['current_price'] * 1.01 
        updated = update_outcomes(current_price)
        print(f"Update returned: {updated}")
    else:
        print("No pending updates found.")

if __name__ == "__main__":
    inspect_tracker()
