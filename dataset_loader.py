import os
import cv2
import numpy as np
import pandas as pd

def load_fusion_dataset(
    img_dir="dataset/images",
    label_file="dataset/labels.csv",
    img_size=64,
    seq_len=20
):
    labels_df = pd.read_csv(label_file)

    X_img, X_seq, y = [], [], []

    for _, row in labels_df.iterrows():
        img_path = os.path.join(img_dir, row["image"])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        img = cv2.resize(img, (img_size, img_size))
        img = img / 255.0
        img = img.reshape(img_size, img_size, 1)

        seq = np.array(row["returns"].split(",")).astype(float)
        seq = seq[-seq_len:].reshape(seq_len, 1)

        X_img.append(img)
        X_seq.append(seq)
        y.append(row["label"])

    return (
        np.array(X_img),
        np.array(X_seq),
        np.array(y)
    )