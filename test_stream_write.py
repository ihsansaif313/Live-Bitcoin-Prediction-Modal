
import websocket
import json
import pandas as pd
import os
import time

def on_message(ws, message):
    print("Received message!")
    print(message[:100])
    ws.close()

def test_write():
    print("Testing file write...")
    try:
        df = pd.DataFrame({'a': [1, 2, 3]})
        df.to_csv("btc_test_write.csv", index=False)
        print("Write successful: btc_test_write.csv")
    except Exception as e:
        print(f"Write failed: {e}")

def test_socket():
    print("Testing websocket...")
    url = "wss://stream.binance.com:9443/ws/btcusdt@trade"
    ws = websocket.WebSocketApp(url, on_message=on_message)
    ws.run_forever()

if __name__ == "__main__":
    test_write()
    test_socket()
