# Stack Investor
This project is for educational and research purposes only and should not be treated as financial advice.

Stack Investor is a machine learning project for **Buy / Hold / Sell** stock signal classification using:

- **Technical indicators** from daily OHLCV stock data
- **Daily news sentiment** aggregated from financial/news headlines
- **Supervised learning models**, primarily **XGBoost**

This repository contains both:

1. the original **course project / experiment pipeline**
2. a **live app prototype** built with Streamlit

---

## Project Overview

The goal of Stack Investor is to predict next-day stock signals (**Buy**, **Hold**, or **Sell**) for a set of large-cap U.S. stocks.

The project began as a machine learning experiment comparing:

- **Technical-only features**
- **Technical + sentiment features**

It was later extended into a live application that can:

- load saved trained models
- fetch current stock price data
- build live technical features
- display live predictions in a Streamlit dashboard
- ingest live headlines from RSS feeds
- aggregate live daily sentiment features

---

## Main Historical Results

On the shared stock-news historical window, the best-performing model was **XGBoost with technical + sentiment features**.

| Experiment | Model | Accuracy | Macro F1 | Weighted F1 |
|---|---|---:|---:|---:|
| Technical only | Hold baseline | 0.1797 | 0.1016 | 0.0548 |
| Technical only | Logistic Regression | 0.3112 | 0.3101 | 0.3086 |
| Technical only | XGBoost | 0.3547 | 0.3478 | 0.3601 |
| Technical + Sentiment | Hold baseline | 0.1797 | 0.1016 | 0.0548 |
| Technical + Sentiment | Logistic Regression | 0.3116 | 0.3114 | 0.3103 |
| Technical + Sentiment | XGBoost | 0.3775 | 0.3720 | 0.3847 |

### Key Takeaway

Adding market-wide daily sentiment produced only a very small improvement for Logistic Regression, but it gave a clear improvement for XGBoost. This suggests that the interaction between technical indicators and sentiment is nonlinear.

---

## Current Live App Status

### Working now

- Saved **technical-only live model**
- Saved **historical technical + sentiment model**
- Streamlit dashboard
- Live technical-only predictions
- Live news ingestion from RSS feeds
- Live daily sentiment aggregation
- Model mode selection in the UI
- Price chart for selected ticker
- Refresh button for running the live pipeline

### Current limitation

- The **Technical + Sentiment** live mode is wired in, but it requires enough accumulated recent headline history aligned with trading dates before it can reliably produce live predictions.
- Because of that, **Technical Only** is currently the most stable live demo mode.

---

## Repository Structure

```text
.
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── features.py
│   ├── train.py
│   ├── predict.py
│   ├── news_ingest.py
│   ├── live_sentiment.py
│   ├── update_live_news.py
│   └── main.py
├── models/
│   ├── xgboost_technical.pkl
│   └── xgboost_sentiment.pkl
├── artifacts/
│   ├── manifest_technical.json
│   ├── manifest_sentiment.json
├── data/
│   ├── raw/
│   │   └── news_sources.json
│   └── processed/
├── legacy/
│   ├── improved_stock_project.py
│   ├── build_reddit_sentiment_features.py
│   └── run_reddit_sentiment_ablation.py
├── README.md
├── requirements.txt
└── .gitignore
```

## Future Improvements

- Add scheduled automatic updates every 2 hours
- Improve live sentiment alignment with market sessions
- Use ticker-specific news instead of market-wide sentiment
- Add a rule-based baseline such as RSI or moving-average crossover
- Deploy the Streamlit dashboard
- Try finance-specific sentiment models such as FinBERT


## Features Used

### Technical Features

The technical feature set includes:

- 1-day, 5-day, and 10-day returns
- Simple moving averages
- Exponential moving averages
- Price relative to moving averages
- Moving-average crossover indicators
- Rolling volatility
- RSI(14)
- MACD, signal, and histogram
- Bollinger Band features
- Volume changes
- Rolling volume averages

### Sentiment Features

The sentiment feature set includes:

- Mean daily sentiment
- Sentiment standard deviation
- Minimum daily sentiment
- Maximum daily sentiment
- Average positive sentiment score
- Average neutral sentiment score
- Average negative sentiment score
- Total headline count
- Positive headline count
- Negative headline count
- Neutral headline count

## Models

The project uses the following models:

- Hold baseline
- Logistic Regression
- XGBoost

The live app primarily uses saved XGBoost models.

## Installation

Install dependencies with:

```bash
pip install -r requirements.txt
```

If needed, install directly with:

```bash
pip install pandas numpy scikit-learn xgboost yfinance vaderSentiment feedparser python-dateutil joblib streamlit
```

## How to Run

### 1. Launch the live dashboard

```bash
python -m streamlit run src/main.py
```

This opens the Streamlit UI in your browser.

### 2. Generate live technical-only predictions

```bash
python -m src.predict --model-filename xgboost_technical.pkl --manifest-filename manifest_technical.json --predictions-filename predictions_technical_live.csv
```

This uses the saved technical-only model and writes live predictions.

### 3. Update live headlines and live sentiment

```bash
python -m src.update_live_news
```

This updates:

- `data/raw/live_headlines.csv`
- `data/processed/live_daily_sentiment.csv`
- `artifacts/live_news_status.json`

### 4. Train the technical-only saved model

```bash
python -m src.train --start-date 2016-01-01 --end-date 2025-12-31 --model-filename xgboost_technical.pkl --manifest-filename manifest_technical.json --predictions-filename predictions_technical.csv
```

### 5. Train the historical technical + sentiment model

```bash
python -m src.train --start-date 2009-01-01 --end-date 2016-07-01 --sentiment-csv reddit_daily_sentiment.csv --model-filename xgboost_sentiment.pkl --manifest-filename manifest_sentiment.json --predictions-filename predictions_sentiment.csv
```

### 6. Run historical experiment code

The original historical experiment pipeline is preserved in the legacy/ folder.

These scripts were used for:

- historical technical-only vs technical+sentiment experiments
- sentiment feature generation from RedditNews
- ablation analysis

Legacy files:

- `legacy/improved_stock_project.py`
- `legacy/build_reddit_sentiment_features.py`
- `legacy/run_reddit_sentiment_ablation.py`

## Live App Workflow

The current live app works like this:

### Technical Only mode

- Load the saved technical-only model
- Download fresh stock price data
- Build technical indicators
- Generate Buy / Hold / Sell predictions
- Display predictions in the dashboard

### Technical + Sentiment mode

- Ingest live headlines from RSS feeds
- Build live daily sentiment features
- Attempt to merge sentiment with live market features
- Load the sentiment-trained saved model
- Generate predictions if enough aligned live sentiment history exists

## Data Sources

This project uses:

- Yahoo Finance OHLCV data via yfinance
- Kaggle RedditNews dataset for historical dated headlines
- RSS feed sources for the live news prototype

## Notes and Limitations

- The historical sentiment setup uses market-wide daily sentiment, not ticker-specific news.
- Because of this, all stocks on the same day receive the same sentiment summary.
- The live sentiment mode currently needs more accumulated trading-day-aligned headline history before it can reliably produce live predictions.
- The technical-only live mode is the most stable live demo path right now.
- The live app is a prototype and should not be treated as financial advice.

## Recommended Demo Flow

If you are demonstrating the project live, the best path is:

1. Open the Streamlit app
2. Use Technical Only mode
3. Click Refresh Predictions
4. Show the latest Buy / Hold / Sell outputs
5. Show the price chart for a selected ticker
6. Optionally show the Technical + Sentiment mode and explain that it is structurally wired in but still warming up as live sentiment history accumulates