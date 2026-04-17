# 📈 Financial Market Recognition using CNN, Fusion Models & MT5

## 🔍 Project Overview

This project is an **AI-based Financial Market Recognition system** that analyzes **candlestick price patterns** from live market data and predicts short-term market trends (**Bullish / Bearish**).

Unlike traditional stock prediction projects, this system:

* Uses **visual candlestick patterns** (just like human traders)
* Applies **Deep Learning (CNN + Fusion Model)**
* Works with **live data from MetaTrader 5 (MT5)**
* Provides **Explainable AI (Grad-CAM heatmaps)** to show *why* a prediction was made

This makes the project suitable for **academic evaluation, viva, and internships**.

---

## 🎯 Problem Statement

Financial markets are highly volatile and noisy. Traditional indicators often fail to capture **visual price-action patterns** that traders rely on.

**Goal:**

> Automatically recognize candlestick patterns and predict whether the market will move **up (Bullish)** or **down (Bearish)** using Deep Learning.

---

## 🧠 Key Concepts Used

* Candlestick Pattern Recognition
* Convolutional Neural Networks (CNN)
* Fusion Learning (Visual + Numerical Features)
* Explainable AI (Grad-CAM)
* Real-time Market Data (MT5 API)

---

## 🏗️ System Architecture

### 1️⃣ Data Collection (MT5)

* Live OHLC (Open, High, Low, Close) data fetched using **MetaTrader 5 API**
* Market: Forex (EURUSD by default)

### 2️⃣ Candlestick Image Generation

* OHLC data converted into **candlestick chart images**
* Images represent real trading patterns visually

### 3️⃣ Dataset Labeling

* **Bullish (1):** Next price goes up
* **Bearish (0):** Next price goes down

### 4️⃣ CNN Model

* Learns **visual candlestick patterns**
* Outputs trend probability

### 5️⃣ Fusion Model (Novelty)

* Combines:

  * CNN visual features
  * Technical features (returns, moving averages)
* Learns **cross-pattern relationships**

### 6️⃣ Explainability (Grad-CAM)

* Generates heatmaps over candlestick images
* Shows **which candle regions influenced the prediction**

---

## ✨ Novelty of the Project

### ✅ What makes this project different?

| Feature          | Traditional Projects | This Project               |
| ---------------- | -------------------- | -------------------------- |
| Data             | Numerical indicators | Visual candlestick images  |
| Model            | Single ML model      | CNN + Fusion Deep Learning |
| Interpretability | Black-box            | Explainable (Grad-CAM)     |
| Environment      | Offline datasets     | Live MT5 market data       |
| Output           | Prediction only      | Prediction + visual proof  |

➡️ This project **bridges the gap between research papers and real trading systems**.

---

## 📂 Project Structure

```
Financial-Market-Recognition/
│
├── candlestick_vision.py      # Generate candlestick images
├── dataset_loader.py          # Load image dataset
├── cnn_model.py               # CNN architecture
├── train_cnn.py               # CNN training
├── fusion_dataset.py          # Fusion data preparation
├── fusion_model.py            # CNN + feature fusion model
├── train_fusion.py            # Fusion model training
├── grad_cam.py                # Explainable AI heatmaps
├── main.py                    # Main execution pipeline
├── test_mt5.py                # MT5 connection test
│
├── candles/                   # Generated candlestick images
├── heatmaps/                  # Grad-CAM outputs
│
├── requirements.txt           # Dependencies
├── README.md                  # Project documentation
└── .gitignore                 # Ignored files
```

---

## ▶️ How to Run the Project

### 1️⃣ Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Test MT5 Connection

```bash
python test_mt5.py
```

### 4️⃣ Train CNN Model

```bash
python train_cnn.py
```

### 5️⃣ Train Fusion Model

```bash
python train_fusion.py
```

### 6️⃣ Run Full Pipeline

```bash
python main.py
```

---

## 📊 Outputs

* Candlestick images
* CNN accuracy & loss plots
* Confusion matrix
* ROC curve
* Fusion model predictions
* Grad-CAM heatmaps explaining decisions

---

## 📌 Applications

* Algorithmic trading research
* Financial pattern recognition
* Explainable AI in finance
* Academic projects & demonstrations

---

## 🧑‍🎓 Academic Use

This project is suitable for:

* 3rd year / Final year engineering projects
* Minor specialization (AI / ML / FinTech)
* Internship portfolios

---

## 🚀 Future Enhancements

* LSTM / Transformer integration
* Multi-timeframe analysis
* Trading strategy automation
* Risk management module
* RAG-based financial explanation system

---

## 📜 Disclaimer

This project is for **educational and research purposes only**. It does not constitute financial advice.

---

## 🤝 Author

**Shreya Singh**
Computer Science & Engineering (AI)
