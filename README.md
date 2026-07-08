# 📈 Stock Price Prediction — LSTM + ARIMA

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/akshay-birla-03/stock-price-prediction-model/blob/main/notebooks/Run_in_Colab.ipynb)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)](#)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white)](#)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A hybrid stock-price forecaster combining a **deep-learning LSTM** (non-linear, long-range
temporal dependencies) with a **statistical ARIMA** model (linear autocorrelation), then
blending their outputs. Data is pulled **live from Yahoo Finance**.

▶️ **Run it now, no setup:** click the **Open in Colab** badge — it clones, installs and runs
the forecaster in your browser.

## Highlights

- **Live data** via \`yfinance\` (any ticker, e.g. \`TATAELXSI.NS\`, \`RELIANCE.NS\`)
- **LSTM** on a 60-day look-back window for sequence forecasting
- **ARIMA** statistical baseline on the same series
- **Hybrid** prediction combining both models
- **Evaluation**: MSE and R², plus actual-vs-predicted plots

## Quickstart (local)

\`\`\`bash
git clone https://github.com/akshay-birla-03/stock-price-prediction-model.git
cd stock-price-prediction-model
pip install -r requirements.txt
python stock_price_prediction_lstm_arima.py   # edit the \`ticker\` variable first
\`\`\`

## Project layout

\`\`\`
stock_price_prediction_lstm_arima.py     # fetch -> LSTM + ARIMA -> evaluate -> plot
stock_price_prediction_documentation.md  # detailed write-up
notebooks/Run_in_Colab.ipynb             # one-click Colab runner
requirements.txt
\`\`\`

## Tech

Python · TensorFlow/Keras · statsmodels · scikit-learn · yfinance · pandas · NumPy · Matplotlib

## Disclaimer

For research and educational use only — **not** financial advice.

---
Author: **Akshay Birla** · [GitHub](https://github.com/akshay-birla-03) · MIT License
