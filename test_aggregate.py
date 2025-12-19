import pandas as pd
from datetime import datetime, timedelta, timezone

# Read trades
df = pd.read_csv("btc_trades_live.csv")
df["time"] = pd.to_datetime(df["time"], format="mixed", utc=True)

# Floor to minutes
df["minute"] = df["time"].dt.floor("1min")

# Get current time and filter completed minutes
current = datetime.now(timezone.utc).replace(second=0, microsecond=0)
completed = df[df["minute"] < current]

# Aggregate
candles = completed.groupby("minute").agg(
    open=("price", "first"),
    high=("price", "max"),
    low=("price", "min"),
    close=("price", "last"),
    volume=("qty", "sum"),
    numberOfTrades=("tradeId", "count")
).reset_index()

# Rename and add timeClose
candles.rename(columns={"minute": "timeOpen"}, inplace=True)
candles["timeClose"] = candles["timeOpen"] + timedelta(seconds=59, milliseconds=999)

# Reorder columns
candles = candles[["timeOpen", "timeClose", "open", "high", "low", "close", "volume", "numberOfTrades"]]

# Save
candles.to_csv("btc_live_candles.csv", index=False)
print(f"Created {len(candles)} candles")
print(candles)
