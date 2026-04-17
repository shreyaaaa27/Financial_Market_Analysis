import os
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
import mplfinance as mpf
import cv2


# -----------------------------
# 1. FETCH MT5 DATA
# -----------------------------
def get_market_data(symbol="EURUSD", n=200):

    if not mt5.initialize():
        print("MT5 initialization failed")
        return None

    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, n)
    mt5.shutdown()

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')

    df = df.rename(columns={
        "time": "Date",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "tick_volume": "Volume"
    })

    df.set_index("Date", inplace=True)

    return df


# -----------------------------
# 2. LABEL LOGIC (BUY / SELL / HOLD)
# -----------------------------
def get_label(df, i, window=20):

    start = df['Close'].iloc[i]
    end = df['Close'].iloc[i + window]

    change = (end - start) / start * 100

    if change > 0.15:
        return "BUY"
    elif change < -0.15:
        return "SELL"
    else:
        return "HOLD"


# -----------------------------
# 3. GENERATE LABELED CANDLES (MAX 50)
# -----------------------------
def generate_labeled_candles(df, save_dir="candles", window=20, max_images=50):

    os.makedirs(save_dir, exist_ok=True)

    dataset = []

    limit = min(len(df) - window - 1, max_images)

    for i in range(limit):

        chunk = df.iloc[i:i+window].copy()

        label = get_label(df, i, window)

        fname = f"{save_dir}/candle_{i}_{label}.png"

        # plot candle
        mpf.plot(
            chunk,
            type='candle',
            style='charles',
            axisoff=True,
            savefig=dict(fname=fname, dpi=100, bbox_inches='tight')
        )

        # -----------------------------
        # ADD LABEL TEXT ON IMAGE
        # -----------------------------
        img = cv2.imread(fname)

        color = (0, 255, 0) if label == "BUY" else (0, 0, 255) if label == "SELL" else (255, 255, 0)

        cv2.putText(
            img,
            label,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2
        )

        cv2.imwrite(fname, img)

        dataset.append([fname, label])

    return dataset


# -----------------------------
# 4. MAIN FUNCTION
# -----------------------------
def main():

    print("Fetching MT5 data...")

    df = get_market_data(n=300)

    if df is None:
        return

    print("Generating labeled candlestick images (max 50)...")

    dataset = generate_labeled_candles(
        df,
        save_dir="candles",
        window=20,
        max_images=50
    )

    # save dataset
    dataset_df = pd.DataFrame(dataset, columns=["image", "label"])

    dataset_df.to_csv("candlestick_labeled_dataset.csv", index=False)

    print("\nDone!")
    print("Dataset saved as candlestick_labeled_dataset.csv")
    print(dataset_df.head())


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    main()