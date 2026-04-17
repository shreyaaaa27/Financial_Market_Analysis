import numpy as np
import MetaTrader5 as mt5
import pandas as pd


def get_mt5_data(symbol="EURUSD", n=500):

    if not mt5.initialize():
        return None

    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, n)
    mt5.shutdown()

    df = pd.DataFrame(rates)
    return df


def load_fusion_dataset(window=20):

    df = get_mt5_data()

    data = df[['open','high','low','close']].values

    X_img, X_seq, y = [], [], []

    for i in range(len(data) - window - 1):

        window_data = data[i:i+window]

        # ---------------- IMAGE (CNN) ----------------
        # convert OHLC to "image-like matrix"
        img = window_data
        X_img.append(img)

        # ---------------- SEQUENCE (LSTM) ----------------
        seq = window_data[:, 3].reshape(window, 1)
        X_seq.append(seq)

        # ---------------- LABEL ----------------
        label = 1 if data[i+window][3] > data[i+window-1][3] else 0
        y.append(label)

    return np.array(X_img), np.array(X_seq), np.array(y)