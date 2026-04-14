import MetaTrader5 as mt5
from trade_analysis import get_market_data, train_pattern_model
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from scipy.stats import wilcoxon
import warnings
warnings.filterwarnings("ignore")


def plot_confusion_matrix(cm, title):
    plt.imshow(cm)
    plt.title(title)
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


def main():
    print("Initializing MetaTrader 5...")

    if not mt5.initialize():
        print("MT5 connection failed")
        return

    print("MT5 connected successfully")
    data = get_market_data()

    print("Training models...")
    results = train_pattern_model(data)

    for name, res in results.items():
        print(f"\n===== {name.upper()} MODEL =====")
        print("Classification Report:")
        print(classification_report(res["y_test"], res["y_pred"]))

        cm = confusion_matrix(res["y_test"], res["y_pred"])
        plot_confusion_matrix(cm, f"{name.upper()} Confusion Matrix")

        # ROC Curve
        fpr, tpr, _ = roc_curve(res["y_test"], res["y_prob"])
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, label=f"{name.upper()} AUC = {roc_auc:.2f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{name.upper()} ROC Curve")
        plt.legend()
        plt.show()

    # -------- Wilcoxon Signed Rank Test --------
    lr_preds = results["lr"]["y_pred"]
    svm_preds = results["svm"]["y_pred"]

    stat, p = wilcoxon(lr_preds, svm_preds)
    print("\nWilcoxon Signed Rank Test")
    print("Statistic:", stat)
    print("p-value:", p)

    mt5.shutdown()


if __name__ == "__main__":
    main()