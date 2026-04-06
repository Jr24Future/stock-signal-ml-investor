from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from src.config import AppConfig


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    return df


def split_download_result(raw: pd.DataFrame, tickers: Sequence[str]) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    raw = raw.copy()

    if raw.empty:
        return out

    if not isinstance(raw.columns, pd.MultiIndex):
        df = raw.reset_index().copy()
        ticker = tickers[0]
        df["Ticker"] = ticker
        out[ticker] = df
        return out

    level0 = set(map(str, raw.columns.get_level_values(0)))
    level1 = set(map(str, raw.columns.get_level_values(1)))
    fields = {"Open", "High", "Low", "Close", "Adj Close", "Volume", "Dividends", "Stock Splits"}

    if any(field in level0 for field in fields):
        for ticker in tickers:
            cols = [c for c in raw.columns if str(c[1]) == ticker]
            if not cols:
                continue
            df = raw.loc[:, cols].copy()
            df.columns = [str(c[0]) for c in df.columns]
            df = df.reset_index()
            df["Ticker"] = ticker
            out[ticker] = df
    elif any(ticker in level0 for ticker in tickers):
        for ticker in tickers:
            cols = [c for c in raw.columns if str(c[0]) == ticker]
            if not cols:
                continue
            df = raw.loc[:, cols].copy()
            df.columns = [str(c[1]) for c in df.columns]
            df = df.reset_index()
            df["Ticker"] = ticker
            out[ticker] = df
    elif any(ticker in level1 for ticker in tickers):
        for ticker in tickers:
            cols = [c for c in raw.columns if str(c[1]) == ticker]
            if not cols:
                continue
            df = raw.loc[:, cols].copy()
            df.columns = [str(c[0]) for c in df.columns]
            df = df.reset_index()
            df["Ticker"] = ticker
            out[ticker] = df
    else:
        raise ValueError("Could not infer the yfinance column layout from the download result.")

    return out


def download_batch(batch: Sequence[str], cfg: AppConfig, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
    raw = yf.download(
        tickers=list(batch),
        start=start_date,
        end=end_date,
        auto_adjust=cfg.use_auto_adjusted_prices,
        progress=False,
        group_by="ticker",
        threads=False,
    )
    return split_download_result(raw, batch)


def download_single_ticker(ticker: str, cfg: AppConfig, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    raw = yf.download(
        tickers=ticker,
        start=start_date,
        end=end_date,
        auto_adjust=cfg.use_auto_adjusted_prices,
        progress=False,
        threads=False,
    )
    if raw.empty:
        return None
    df = raw.reset_index().copy()
    df["Ticker"] = ticker
    return df


def download_price_data(cfg: AppConfig, start_date: str, end_date: str) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []

    for i in range(0, len(cfg.tickers), cfg.price_batch_size):
        batch = cfg.tickers[i : i + cfg.price_batch_size]
        batch_success = False

        for attempt in range(1, cfg.max_download_retries + 1):
            try:
                batch_results = download_batch(batch, cfg, start_date, end_date)
                if batch_results:
                    frames.extend(batch_results.values())
                    batch_success = True
                    break
            except Exception as exc:
                if attempt == cfg.max_download_retries:
                    print(f"Batch download failed for {batch}: {exc}")
                else:
                    time.sleep(cfg.retry_sleep_seconds * attempt)

        if batch_success:
            continue

        for ticker in batch:
            downloaded = None
            for attempt in range(1, cfg.max_download_retries + 1):
                try:
                    downloaded = download_single_ticker(ticker, cfg, start_date, end_date)
                    if downloaded is not None and not downloaded.empty:
                        frames.append(downloaded)
                        break
                except Exception as exc:
                    if attempt == cfg.max_download_retries:
                        print(f"Ticker download failed for {ticker}: {exc}")
                    else:
                        time.sleep(cfg.retry_sleep_seconds * attempt)

            if downloaded is None or downloaded.empty:
                print(f"Skipping {ticker}: no data returned.")

    if not frames:
        raise RuntimeError("No market data was downloaded.")

    df = pd.concat(frames, ignore_index=True)
    df = normalize_columns(df)

    required_cols = {"Date", "Open", "High", "Low", "Close", "Volume", "Ticker"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Downloaded data is missing required columns: {sorted(missing)}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "Open", "High", "Low", "Close", "Volume", "Ticker"]).copy()
    df["Ticker"] = df["Ticker"].astype(str)
    df = df.sort_values(["Ticker", "Date"]).drop_duplicates(["Ticker", "Date"]).reset_index(drop=True)
    return df


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    return 100 - (100 / (1 + rs))


def add_technical_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    df = df.copy().sort_values(["Ticker", "Date"])
    grouped = df.groupby("Ticker", group_keys=False)

    df["ret_1"] = grouped["Close"].pct_change(1)
    df["ret_5"] = grouped["Close"].pct_change(5)
    df["ret_10"] = grouped["Close"].pct_change(10)

    for window in [5, 10, 20, 50]:
        df[f"sma_{window}"] = grouped["Close"].transform(lambda s: s.rolling(window).mean())
        df[f"ema_{window}"] = grouped["Close"].transform(lambda s: s.ewm(span=window, adjust=False).mean())
        df[f"close_over_sma_{window}"] = df["Close"] / (df[f"sma_{window}"] + 1e-12) - 1

    df["sma10_gt_sma20"] = (df["sma_10"] > df["sma_20"]).astype(int)
    df["sma20_gt_sma50"] = (df["sma_20"] > df["sma_50"]).astype(int)

    for window in [5, 10, 20]:
        df[f"volatility_{window}"] = grouped["ret_1"].transform(lambda s: s.rolling(window).std())

    df["high_low_range"] = (df["High"] - df["Low"]) / (df["Close"] + 1e-12)
    df["close_open_gap"] = (df["Close"] - df["Open"]) / (df["Open"] + 1e-12)

    df["rsi_14"] = grouped["Close"].transform(lambda s: rsi(s, 14))
    ema12 = grouped["Close"].transform(lambda s: s.ewm(span=12, adjust=False).mean())
    ema26 = grouped["Close"].transform(lambda s: s.ewm(span=26, adjust=False).mean())
    df["macd"] = ema12 - ema26
    df["macd_signal"] = grouped["macd"].transform(lambda s: s.ewm(span=9, adjust=False).mean())
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    rolling20 = grouped["Close"].transform(lambda s: s.rolling(20).mean())
    rolling20_std = grouped["Close"].transform(lambda s: s.rolling(20).std())
    df["bb_upper"] = rolling20 + 2 * rolling20_std
    df["bb_lower"] = rolling20 - 2 * rolling20_std
    df["bb_percent_b"] = (df["Close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-12)
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / (rolling20 + 1e-12)

    df["vol_chg_1"] = grouped["Volume"].pct_change(1)
    for window in [10, 20]:
        df[f"vol_avg_{window}"] = grouped["Volume"].transform(lambda s: s.rolling(window).mean())
        df[f"vol_over_avg_{window}"] = df["Volume"] / (df[f"vol_avg_{window}"] + 1e-12)

    feature_cols = [
        "ret_1", "ret_5", "ret_10",
        "sma_5", "sma_10", "sma_20", "sma_50",
        "ema_5", "ema_10", "ema_20", "ema_50",
        "close_over_sma_5", "close_over_sma_10", "close_over_sma_20", "close_over_sma_50",
        "sma10_gt_sma20", "sma20_gt_sma50",
        "volatility_5", "volatility_10", "volatility_20",
        "high_low_range", "close_open_gap",
        "rsi_14", "macd", "macd_signal", "macd_hist",
        "bb_upper", "bb_lower", "bb_percent_b", "bb_width",
        "vol_chg_1", "vol_avg_10", "vol_avg_20", "vol_over_avg_10", "vol_over_avg_20",
    ]
    return df, feature_cols


def merge_optional_sentiment(df: pd.DataFrame, cfg: AppConfig) -> Tuple[pd.DataFrame, List[str]]:
    if not cfg.sentiment_csv_path:
        return df, []

    sentiment_path = Path(cfg.sentiment_csv_path)
    if not sentiment_path.exists():
        raise FileNotFoundError(f"Sentiment file not found: {sentiment_path}")

    sent = pd.read_csv(sentiment_path)
    sent.columns = [str(c).strip() for c in sent.columns]
    lower_map = {c.lower(): c for c in sent.columns}

    date_col = cfg.sentiment_date_col or lower_map.get("date") or lower_map.get("datetime") or lower_map.get("published")
    if date_col is None:
        raise ValueError("Could not find a sentiment date column.")

    ticker_col = cfg.sentiment_ticker_col or lower_map.get("ticker") or lower_map.get("symbol")

    if cfg.sentiment_numeric_cols:
        numeric_cols = cfg.sentiment_numeric_cols
    else:
        numeric_cols = [
            c for c in sent.columns
            if c not in {date_col, ticker_col} and pd.api.types.is_numeric_dtype(sent[c])
        ]

    if not numeric_cols:
        raise ValueError("No numeric sentiment columns found in sentiment CSV.")

    sent = sent.copy()
    sent[date_col] = pd.to_datetime(sent[date_col], errors="coerce")
    sent = sent.dropna(subset=[date_col]).copy()
    sent["Date"] = sent[date_col].dt.normalize()

    group_cols = ["Date"]
    if ticker_col is not None:
        sent[ticker_col] = sent[ticker_col].astype(str)
        sent["Ticker"] = sent[ticker_col]
        group_cols.append("Ticker")

    sent = sent.groupby(group_cols, as_index=False)[numeric_cols].mean()

    if "Ticker" in sent.columns:
        sent = sent.sort_values(["Ticker", "Date"])
        for col in numeric_cols:
            sent[col] = sent.groupby("Ticker")[col].shift(1)
        merged = df.merge(sent, on=["Date", "Ticker"], how="left")
    else:
        sent = sent.sort_values(["Date"])
        for col in numeric_cols:
            sent[col] = sent[col].shift(1)
        merged = df.merge(sent, on=["Date"], how="left")

    return merged, numeric_cols


def add_labels(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    df = df.copy().sort_values(["Ticker", "Date"])
    grouped = df.groupby("Ticker", group_keys=False)
    df["next_return"] = grouped["Close"].shift(-1) / df["Close"] - 1
    df["label"] = 1
    df.loc[df["next_return"] > threshold, "label"] = 2
    df.loc[df["next_return"] < -threshold, "label"] = 0
    return df


def build_training_frame(
    cfg: AppConfig,
    start_date: str,
    end_date: str,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    df = download_price_data(cfg, start_date=start_date, end_date=end_date)
    df, technical_features = add_technical_features(df)
    df, sentiment_features = merge_optional_sentiment(df, cfg)
    df = add_labels(df, threshold=cfg.label_threshold)

    all_features = technical_features + sentiment_features
    required_cols = all_features + ["Date", "Ticker", "label", "next_return"]
    df = df.dropna(subset=required_cols).copy()
    df = df.sort_values(["Date", "Ticker"]).reset_index(drop=True)
    return df, technical_features, sentiment_features


def build_live_frame(
    cfg: AppConfig,
    start_date: str,
    end_date: str,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    df = download_price_data(cfg, start_date=start_date, end_date=end_date)
    df, technical_features = add_technical_features(df)
    df, sentiment_features = merge_optional_sentiment(df, cfg)

    all_features = technical_features + sentiment_features
    required_cols = all_features + ["Date", "Ticker", "Close"]
    df = df.dropna(subset=required_cols).copy()
    df = df.sort_values(["Date", "Ticker"]).reset_index(drop=True)
    return df, technical_features, sentiment_features