import MetaTrader5 as mt5
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc
)


# -------------------------------------------------
# Fetch Market Data
# -------------------------------------------------
def get_market_data(symbol="EURUSD", bars=500):
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, bars)
    df = pd.DataFrame(rates)

    # Feature Engineering
    df['returns'] = df['close'].pct_change()
    df['ma_5'] = df['close'].rolling(window=5).mean()
    df['ma_10'] = df['close'].rolling(window=10).mean()

    df.dropna(inplace=True)
    return df


# -------------------------------------------------
# Train Pattern Recognition Models
# -------------------------------------------------
def train_pattern_model(df):
    df = df.copy()

    # -------- Trend Labels --------
    # 1 → Bullish (Price goes UP)
    # 0 → Bearish (Price goes DOWN)
    df['target'] = (df['returns'].shift(-1) > 0).astype(int)
    df.dropna(inplace=True)

    X = df[['returns', 'ma_5', 'ma_10']]
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    results = {}

    # -------- Model 1: Logistic Regression --------
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)

    results["lr"] = {
        "model": lr,
        "y_test": y_test,
        "y_pred": lr.predict(X_test),
        "y_prob": lr.predict_proba(X_test)[:, 1]
    }

    # -------- Model 2: Support Vector Machine --------
    svm = SVC(probability=True)
    svm.fit(X_train, y_train)

    results["svm"] = {
        "model": svm,
        "y_test": y_test,
        "y_pred": svm.predict(X_test),
        "y_prob": svm.predict_proba(X_test)[:, 1]
    }

    return results