import os
import pandas as pd

def clean_csv(filename, expected_cols):
    print(f"Cleaning {filename} (expected {expected_cols} columns)...")
    if not os.path.exists(filename):
        print(f"File {filename} not found. Skipping.")
        return

    try:
        # First attempt: Try reading with pandas and skip bad lines
        df = pd.read_csv(filename, on_bad_lines='skip', low_memory=False)
        df.to_csv(filename, index=False)
        print(f"Successfully cleaned {filename} using pandas (skipped bad lines).")
    except Exception as e:
        print(f"Pandas cleaning failed for {filename}: {e}. Attempting manual row-by-row cleaning...")
        
        cleaned_lines = []
        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            header = f.readline()
            cleaned_lines.append(header)
            
            for i, line in enumerate(f):
                if line.strip():
                    cols = line.strip().split(',')
                    if len(cols) == expected_cols:
                        cleaned_lines.append(line)
                    else:
                        print(f"Removing corrupted line {i+2}: {line[:50]}... (Cols: {len(cols)})")
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.writelines(cleaned_lines)
        print(f"Manual cleaning complete for {filename}.")

if __name__ == "__main__":
    # btc_features_normalized.csv has 20 columns
    clean_csv("btc_features_normalized.csv", 20)
    # btc_dataset.csv has 8 columns
    clean_csv("btc_dataset.csv", 8)
    # btc_features.csv has 20 columns
    clean_csv("btc_features.csv", 20)
    # btc_live_candles.csv has 8 columns
    clean_csv("btc_live_candles.csv", 8)
