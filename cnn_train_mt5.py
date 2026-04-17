import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

import tensorflow as tf
from tensorflow.keras import layers, models

warnings.filterwarnings("ignore")

# -----------------------------
# 1. CONNECT + FETCH MT5 DATA
# -----------------------------
def get_market_data(symbol="EURUSD", n=1000):

    print("Connecting to MetaTrader 5...")

    if not mt5.initialize():
        print("MT5 initialization failed")
        return None

    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, n)
    mt5.shutdown()

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')

    print("Data fetched successfully:", df.shape)

    return df


# -----------------------------
# 2. CREATE CNN DATASET
# -----------------------------
def create_dataset(df, window_size=30):

    X = []
    y = []

    data = df[['open', 'high', 'low', 'close']].values

    for i in range(len(data) - window_size - 1):

        window = data[i:i + window_size]

        next_close = data[i + window_size][3]
        last_close = window[-1][3]

        # Label: 1 if price goes up else 0
        label = 1 if next_close > last_close else 0

        X.append(window)
        y.append(label)

    X = np.array(X)
    y = np.array(y)

    print("Dataset created:")
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    return X, y


# -----------------------------
# 3. BUILD CNN MODEL
# -----------------------------
def build_cnn(input_shape):

    model = models.Sequential()

    model.add(layers.Conv1D(64, 3, activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling1D(2))

    model.add(layers.Conv1D(128, 3, activation='relu'))
    model.add(layers.MaxPooling1D(2))

    model.add(layers.Flatten())

    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


# -----------------------------
# 4. TRAIN + EVALUATION
# -----------------------------
def train_and_evaluate(model, X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    print("Training CNN model...")

    history = model.fit(
        X_train, y_train,
        epochs=25,
        batch_size=32,
        validation_split=0.2
    )

    # -------------------------
    # PREDICTIONS
    # -------------------------
    y_prob = model.predict(X_test)
    y_pred = (y_prob > 0.5).astype(int)

    # -------------------------
    # CLASSIFICATION REPORT
    # -------------------------
    print("\n===== CLASSIFICATION REPORT =====\n")
    print(classification_report(y_test, y_pred))

    # -------------------------
    # CONFUSION MATRIX
    # -------------------------
    cm = confusion_matrix(y_test, y_pred)

    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("CNN Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # -------------------------
    # ROC CURVE
    # -------------------------
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.show()

    # -------------------------
    # LOSS GRAPH
    # -------------------------
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title("Loss Curve")
    plt.legend()
    plt.show()

    return model


# -----------------------------
# 5. MAIN FUNCTION
# -----------------------------
def main():

    df = get_market_data()

    if df is None:
        return

    X, y = create_dataset(df)

    # reshape for CNN
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2])

    model = build_cnn((X.shape[1], X.shape[2]))

    trained_model = train_and_evaluate(model, X, y)

    print("\nTraining completed successfully!")


# -----------------------------
# RUN PROGRAM
# -----------------------------
if __name__ == "__main__":
    main()