# Stack Investor

A machine learning project for **Buy / Hold / Sell** stock signal classification using:
- **Technical indicators** from daily OHLCV price data
- **Daily news sentiment** aggregated from financial/news headlines
- **Supervised classification models** including Logistic Regression and XGBoost

## Project Summary

This project predicts next-day stock signals (**Buy**, **Hold**, or **Sell**) for a set of large-cap U.S. stocks. Two versions of the pipeline are compared:

1. **Technical-only model**
2. **Technical + sentiment model**

The main goal is to test whether adding daily news sentiment improves prediction quality beyond technical indicators alone.

## Main Results

On the shared stock-news time window, the best model was **XGBoost with technical + sentiment features**.

| Experiment | Model | Accuracy | Macro F1 | Weighted F1 |
|---|---|---:|---:|---:|
| Technical only | Hold baseline | 0.1797 | 0.1016 | 0.0548 |
| Technical only | Logistic Regression | 0.3112 | 0.3101 | 0.3086 |
| Technical only | XGBoost | 0.3547 | 0.3478 | 0.3601 |
| Technical + Sentiment | Hold baseline | 0.1797 | 0.1016 | 0.0548 |
| Technical + Sentiment | Logistic Regression | 0.3116 | 0.3114 | 0.3103 |
| Technical + Sentiment | XGBoost | 0.3775 | 0.3720 | 0.3847 |

### Key Takeaway

Adding daily market-wide sentiment produced only a tiny improvement for Logistic Regression, but it gave a **clear improvement for XGBoost**, suggesting that the interaction between technical indicators and sentiment is nonlinear.

## Repo Structure

```text
.
├── improved_stock_project.py
├── build_reddit_sentiment_features.py
├── run_reddit_sentiment_ablation.py
├── RedditNews.csv                     # raw news dataset (optional to keep out of GitHub if large)
├── reddit_daily_sentiment.csv         # generated daily sentiment features
├── outputs/                           # generated technical-only outputs
├── outputs_reddit_tech_only/          # generated outputs for technical-only ablation
├── outputs_reddit_with_sentiment/     # generated outputs for technical+sentiment ablation
├── outputs_reddit_ablation_comparison.csv
├── README.md
└── .gitignore
```

## Features Used

### Technical Features
- Returns over multiple windows
- Simple and exponential moving averages
- Moving-average crossover indicators
- Rolling volatility
- RSI(14)
- MACD, signal, and histogram
- Bollinger Band features
- Volume changes and rolling averages

### Sentiment Features
- Mean daily sentiment
- Sentiment standard deviation
- Minimum and maximum daily sentiment
- Average positive, neutral, and negative sentiment scores
- Total headline count
- Positive, negative, and neutral headline counts

## Models

- **Hold baseline**
- **Logistic Regression**
- **XGBoost**

## How It Works

### 1. Build daily sentiment features
This script reads `RedditNews.csv`, scores each headline with VADER, and aggregates sentiment by date.

```bash
python build_reddit_sentiment_features.py
```

This creates:
- `reddit_daily_sentiment.csv`

### 2. Run the technical vs sentiment ablation
This script compares:
- technical-only features
- technical + sentiment features

```bash
python run_reddit_sentiment_ablation.py
```

This creates:
- `outputs_reddit_tech_only/`
- `outputs_reddit_with_sentiment/`
- `outputs_reddit_ablation_comparison.csv`

### 3. Run the standalone technical pipeline
You can also run the main stock project directly:

```bash
python improved_stock_project.py
```

## Installation

Install dependencies with:

```bash
pip install pandas numpy scikit-learn xgboost yfinance vaderSentiment
```

## Data Sources

- Yahoo Finance OHLCV data via `yfinance`
- Kaggle RedditNews dataset for dated headlines

## Notes and Limitations

- The current sentiment setup uses **market-wide daily sentiment**, not ticker-specific news.
- Because of this, all stocks on the same day receive the same sentiment summary.
- This is still useful for testing whether overall market sentiment adds value, but it is weaker than ticker-linked news sentiment.

## Suggested GitHub Upload Strategy

### Keep in the repo
- `improved_stock_project.py`
- `build_reddit_sentiment_features.py`
- `run_reddit_sentiment_ablation.py`
- `README.md`
- `.gitignore`
- small summary CSVs if you want to show final results

### Usually do **not** commit
- large raw datasets from Kaggle
- generated prediction CSVs
- confusion matrices
- backtest curves
- feature importance CSVs
- any rerunnable output folders

## Future Improvements

- Add a rule-based baseline such as RSI or moving-average crossover
- Use ticker-specific news instead of market-wide sentiment
- Try finance-specific sentiment models such as FinBERT
- Expand the ticker universe and time range

## Author / Course Context

This project was developed as a machine learning course project focused on applying supervised learning and sentiment analysis to stock signal classification.
