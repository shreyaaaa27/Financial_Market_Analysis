import numpy as np
from sklearn.model_selection import train_test_split
from fusion_dataset import load_fusion_dataset
from fusion_model import build_fusion_model

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc


# -----------------------------
# EVALUATION FUNCTION
# -----------------------------
def evaluate_model(model, X_img_test, X_seq_test, y_test):

    print("Running evaluation...")

    y_prob = model.predict([X_img_test, X_seq_test])
    y_pred = (y_prob > 0.5).astype(int)

    # CONFUSION MATRIX
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix - CNN + LSTM Fusion")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # CLASSIFICATION REPORT
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    # ROC CURVE
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0,1],[0,1],'--')
    plt.title("ROC Curve - Fusion Model")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.show()


# -----------------------------
# MAIN FUNCTION
# -----------------------------
def main():

    print("Loading MT5 dataset...")

    X_img, X_seq, y = load_fusion_dataset(window=20)

    # reshape CNN input
    X_img = X_img.reshape(X_img.shape[0], X_img.shape[1], X_img.shape[2], 1)

    print("Shapes:")
    print(X_img.shape, X_seq.shape, y.shape)

    # split dataset
    X_img_train, X_img_test, X_seq_train, X_seq_test, y_train, y_test = train_test_split(
        X_img, X_seq, y, test_size=0.2, random_state=42
    )

    # build model
    model = build_fusion_model(
        img_shape=X_img.shape[1:],
        seq_shape=X_seq.shape[1:]
    )

    # train model
    print("Training model...")
    model.fit(
        [X_img_train, X_seq_train],
        y_train,
        validation_data=([X_img_test, X_seq_test], y_test),
        epochs=10,
        batch_size=32
    )

    # save model
    model.save("fusion_model.h5")
    print("Model saved!")

    # ✅ CALL EVALUATION HERE (IMPORTANT)
    evaluate_model(model, X_img_test, X_seq_test, y_test)


if __name__ == "__main__":
    main()