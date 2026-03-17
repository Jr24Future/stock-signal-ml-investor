from __future__ import annotations

import json
import math
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


LABEL_TO_NAME = {0: "Sell", 1: "Hold", 2: "Buy"}


@dataclass
class Config:
    tickers: List[str] = field(
        default_factory=lambda: [
            "AAPL", "MSFT", "AMZN", "GOOG", "META",
            "NVDA", "TSLA", "JPM", "XOM", "NFLX",
        ]
    )
    start_date: str = "2016-01-01"
    end_date: str = "2025-12-31"
    label_threshold: float = 0.003
    output_dir: str = "outputs"
    price_batch_size: int = 5
    max_download_retries: int = 3
    retry_sleep_seconds: float = 2.0
    use_auto_adjusted_prices: bool = True
    train_frac: float = 0.70
    valid_frac: float = 0.15
    random_state: int = 42
    sentiment_csv_path: Optional[str] = None
    sentiment_date_col: Optional[str] = None
    sentiment_ticker_col: Optional[str] = None
    sentiment_numeric_cols: Optional[List[str]] = None


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    return df


def _split_download_result(raw: pd.DataFrame, tickers: Sequence[str]) -> Dict[str, pd.DataFrame]:
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


def _download_batch(batch: Sequence[str], cfg: Config) -> Dict[str, pd.DataFrame]:
    raw = yf.download(
        tickers=list(batch),
        start=cfg.start_date,
        end=cfg.end_date,
        auto_adjust=cfg.use_auto_adjusted_prices,
        progress=False,
        group_by="ticker",
        threads=False,
    )
    return _split_download_result(raw, batch)


def _download_single_ticker(ticker: str, cfg: Config) -> Optional[pd.DataFrame]:
    raw = yf.download(
        tickers=ticker,
        start=cfg.start_date,
        end=cfg.end_date,
        auto_adjust=cfg.use_auto_adjusted_prices,
        progress=False,
        threads=False,
    )
    if raw.empty:
        return None
    df = raw.reset_index().copy()
    df["Ticker"] = ticker
    return df


def download_price_data(cfg: Config) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    tickers = cfg.tickers

    for i in range(0, len(tickers), cfg.price_batch_size):
        batch = tickers[i : i + cfg.price_batch_size]
        batch_success = False

        for attempt in range(1, cfg.max_download_retries + 1):
            try:
                batch_results = _download_batch(batch, cfg)
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
                    downloaded = _download_single_ticker(ticker, cfg)
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
        raise RuntimeError("No market data was downloaded. Check your internet connection or yfinance installation.")

    df = pd.concat(frames, ignore_index=True)
    df = _normalize_columns(df)

    expected_cols = {"Date", "Open", "High", "Low", "Close", "Volume", "Ticker"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Downloaded data is missing required columns: {sorted(missing)}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "Open", "High", "Low", "Close", "Volume", "Ticker"]).copy()
    df["Ticker"] = df["Ticker"].astype(str)
    df = df.sort_values(["Ticker", "Date"]).drop_duplicates(["Ticker", "Date"]).reset_index(drop=True)
    return df


def merge_optional_sentiment(df: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, List[str]]:
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
        raise ValueError(
            "Could not find a sentiment date column. Provide sentiment_date_col in Config or rename a column to 'date'."
        )

    ticker_col = cfg.sentiment_ticker_col or lower_map.get("ticker") or lower_map.get("symbol")

    if cfg.sentiment_numeric_cols:
        numeric_cols = cfg.sentiment_numeric_cols
    else:
        numeric_cols = [
            c for c in sent.columns
            if c not in {date_col, ticker_col} and pd.api.types.is_numeric_dtype(sent[c])
        ]

    if not numeric_cols:
        raise ValueError(
            "No numeric sentiment columns found. The sentiment CSV should contain aggregated numeric columns like mean_sentiment or headline_count."
        )

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


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
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

    df["rsi_14"] = grouped["Close"].transform(lambda s: _rsi(s, 14))
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


def add_labels(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    df = df.copy().sort_values(["Ticker", "Date"])
    grouped = df.groupby("Ticker", group_keys=False)
    df["next_return"] = grouped["Close"].shift(-1) / df["Close"] - 1
    df["label"] = 1
    df.loc[df["next_return"] > threshold, "label"] = 2
    df.loc[df["next_return"] < -threshold, "label"] = 0
    return df


def build_dataset(cfg: Config) -> Tuple[pd.DataFrame, List[str], List[str]]:
    df = download_price_data(cfg)
    df, technical_features = add_technical_features(df)
    df, sentiment_features = merge_optional_sentiment(df, cfg)
    df = add_labels(df, threshold=cfg.label_threshold)

    all_features = technical_features + sentiment_features
    required_cols = all_features + ["Date", "Ticker", "label", "next_return"]
    df = df.dropna(subset=required_cols).copy()
    df = df.sort_values(["Date", "Ticker"]).reset_index(drop=True)
    return df, technical_features, sentiment_features


def chronological_split(df: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    unique_dates = np.array(sorted(df["Date"].unique()))
    n_dates = len(unique_dates)

    train_end = int(n_dates * cfg.train_frac)
    valid_end = int(n_dates * (cfg.train_frac + cfg.valid_frac))

    train_dates = set(unique_dates[:train_end])
    valid_dates = set(unique_dates[train_end:valid_end])
    test_dates = set(unique_dates[valid_end:])

    train_df = df[df["Date"].isin(train_dates)].copy()
    valid_df = df[df["Date"].isin(valid_dates)].copy()
    test_df = df[df["Date"].isin(test_dates)].copy()
    return train_df, valid_df, test_df


def summarize_labels(df: pd.DataFrame) -> pd.Series:
    return (
        df["label"]
        .map(LABEL_TO_NAME)
        .value_counts(normalize=True)
        .reindex(["Sell", "Hold", "Buy"])
        .fillna(0.0)
    )


def predict_with_thresholds(proba: np.ndarray, buy_threshold: float, sell_threshold: float) -> np.ndarray:
    base = np.argmax(proba, axis=1)
    pred = np.ones(len(base), dtype=int)

    buy_mask = (base == 2) & (proba[:, 2] >= buy_threshold)
    sell_mask = (base == 0) & (proba[:, 0] >= sell_threshold)

    pred[buy_mask] = 2
    pred[sell_mask] = 0
    return pred


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
    }


def classification_report_dict(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, dict]:
    return classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        target_names=["Sell", "Hold", "Buy"],
        output_dict=True,
        zero_division=0,
    )


def tune_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    random_state: int,
) -> Tuple[Pipeline, Dict[str, float]]:
    candidates = [0.1, 1.0, 3.0, 10.0]
    best_model: Optional[Pipeline] = None
    best_metrics: Optional[Dict[str, float]] = None

    for c_value in candidates:
        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=c_value,
                        solver="lbfgs",
                        class_weight="balanced",
                        max_iter=3000,
                        random_state=random_state,
                    ),
                ),
            ]
        )
        model.fit(X_train, y_train)
        pred_valid = model.predict(X_valid)
        metrics = evaluate_predictions(y_valid, pred_valid)
        metrics["C"] = c_value

        if best_metrics is None or metrics["macro_f1"] > best_metrics["macro_f1"]:
            best_model = model
            best_metrics = metrics

    assert best_model is not None and best_metrics is not None
    return best_model, best_metrics


def tune_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    random_state: int,
) -> Tuple[XGBClassifier, Dict[str, float], Tuple[float, float]]:
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

    param_grid = [
        {"max_depth": 3, "learning_rate": 0.03, "min_child_weight": 1},
        {"max_depth": 3, "learning_rate": 0.05, "min_child_weight": 3},
        {"max_depth": 4, "learning_rate": 0.03, "min_child_weight": 1},
        {"max_depth": 4, "learning_rate": 0.05, "min_child_weight": 3},
    ]
    threshold_grid = [0.33, 0.40, 0.45, 0.50, 0.55]

    best_model: Optional[XGBClassifier] = None
    best_metrics: Optional[Dict[str, float]] = None
    best_thresholds = (0.50, 0.50)

    for params in param_grid:
        model = XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            n_estimators=1200,
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            min_child_weight=params["min_child_weight"],
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.0,
            reg_lambda=1.0,
            tree_method="hist",
            early_stopping_rounds=50,
            eval_metric="mlogloss",
            random_state=random_state,
            verbosity=0,
        )
        model.fit(
            X_train,
            y_train,
            sample_weight=sample_weights,
            eval_set=[(X_valid, y_valid)],
            verbose=False,
        )
        valid_proba = model.predict_proba(X_valid)

        for buy_threshold in threshold_grid:
            for sell_threshold in threshold_grid:
                pred_valid = predict_with_thresholds(valid_proba, buy_threshold, sell_threshold)
                metrics = evaluate_predictions(y_valid, pred_valid)
                metrics.update(params)
                metrics["buy_threshold"] = buy_threshold
                metrics["sell_threshold"] = sell_threshold

                if best_metrics is None or metrics["macro_f1"] > best_metrics["macro_f1"]:
                    best_model = model
                    best_metrics = metrics
                    best_thresholds = (buy_threshold, sell_threshold)

    assert best_model is not None and best_metrics is not None
    return best_model, best_metrics, best_thresholds


def fit_hold_baseline(y_test: np.ndarray) -> np.ndarray:
    return np.ones_like(y_test)


def simple_backtest(df: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    out = df[["Date", "Ticker", "next_return", pred_col]].copy()
    signal_map = {0: -1, 1: 0, 2: 1}
    out["signal"] = out[pred_col].map(signal_map)
    out["strategy_return"] = out["signal"] * out["next_return"]

    daily = (
        out.groupby("Date", as_index=False)
        .agg(
            strategy_return=("strategy_return", "mean"),
            benchmark_return=("next_return", "mean"),
            trades=("signal", lambda s: int((s != 0).sum())),
        )
        .sort_values("Date")
    )
    daily["strategy_curve"] = (1 + daily["strategy_return"].fillna(0)).cumprod() - 1
    daily["benchmark_curve"] = (1 + daily["benchmark_return"].fillna(0)).cumprod() - 1
    return daily


def compute_backtest_summary(curve_df: pd.DataFrame) -> Dict[str, float]:
    returns = curve_df["strategy_return"].fillna(0)
    if len(returns) == 0:
        return {
            "total_return": 0.0,
            "annualized_return": 0.0,
            "annualized_volatility": 0.0,
            "sharpe_like": 0.0,
            "max_drawdown": 0.0,
        }

    total_return = float(curve_df["strategy_curve"].iloc[-1])
    avg_daily = float(returns.mean())
    vol_daily = float(returns.std(ddof=0))
    annualized_return = float((1 + avg_daily) ** 252 - 1) if avg_daily > -1 else -1.0
    annualized_volatility = float(vol_daily * math.sqrt(252))
    sharpe_like = float((avg_daily / vol_daily) * math.sqrt(252)) if vol_daily > 0 else 0.0

    equity = (1 + returns).cumprod()
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    max_drawdown = float(drawdown.min()) if len(drawdown) else 0.0

    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_like": sharpe_like,
        "max_drawdown": max_drawdown,
    }


def run_experiment(cfg: Config) -> Dict[str, dict]:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df, technical_features, sentiment_features = build_dataset(cfg)
    all_features = technical_features + sentiment_features

    train_df, valid_df, test_df = chronological_split(df, cfg)

    X_train = train_df[all_features].to_numpy()
    y_train = train_df["label"].to_numpy()
    X_valid = valid_df[all_features].to_numpy()
    y_valid = valid_df["label"].to_numpy()
    X_test = test_df[all_features].to_numpy()
    y_test = test_df["label"].to_numpy()

    results: Dict[str, dict] = {
        "config": asdict(cfg),
        "dataset": {
            "rows_total": int(len(df)),
            "rows_train": int(len(train_df)),
            "rows_valid": int(len(valid_df)),
            "rows_test": int(len(test_df)),
            "num_features": int(len(all_features)),
            "technical_features": technical_features,
            "sentiment_features": sentiment_features,
            "train_label_distribution": summarize_labels(train_df).to_dict(),
            "valid_label_distribution": summarize_labels(valid_df).to_dict(),
            "test_label_distribution": summarize_labels(test_df).to_dict(),
        },
    }

    hold_test_pred = fit_hold_baseline(y_test)
    hold_metrics = evaluate_predictions(y_test, hold_test_pred)
    results["hold_baseline"] = {
        "validation": None,
        "test": hold_metrics,
        "classification_report": classification_report_dict(y_test, hold_test_pred),
    }

    lr_model, lr_valid_metrics = tune_logistic_regression(
        X_train, y_train, X_valid, y_valid, cfg.random_state
    )
    lr_test_pred = lr_model.predict(X_test)
    lr_test_metrics = evaluate_predictions(y_test, lr_test_pred)
    results["logistic_regression"] = {
        "validation": lr_valid_metrics,
        "test": lr_test_metrics,
        "classification_report": classification_report_dict(y_test, lr_test_pred),
    }

    xgb_model, xgb_valid_metrics, xgb_thresholds = tune_xgboost(
        X_train, y_train, X_valid, y_valid, cfg.random_state
    )
    xgb_test_proba = xgb_model.predict_proba(X_test)
    xgb_test_pred = predict_with_thresholds(xgb_test_proba, *xgb_thresholds)
    xgb_test_metrics = evaluate_predictions(y_test, xgb_test_pred)
    results["xgboost"] = {
        "validation": xgb_valid_metrics,
        "test": xgb_test_metrics,
        "thresholds": {
            "buy_threshold": xgb_thresholds[0],
            "sell_threshold": xgb_thresholds[1],
        },
        "best_iteration": int(getattr(xgb_model, "best_iteration", -1)),
        "classification_report": classification_report_dict(y_test, xgb_test_pred),
    }

    test_predictions = test_df[["Date", "Ticker", "Close", "next_return", "label"]].copy()
    test_predictions["hold_pred"] = hold_test_pred
    test_predictions["lr_pred"] = lr_test_pred
    test_predictions["xgb_pred"] = xgb_test_pred
    test_predictions["xgb_p_sell"] = xgb_test_proba[:, 0]
    test_predictions["xgb_p_hold"] = xgb_test_proba[:, 1]
    test_predictions["xgb_p_buy"] = xgb_test_proba[:, 2]
    test_predictions.to_csv(output_dir / "test_predictions.csv", index=False)

    for model_name, pred in {
        "hold": hold_test_pred,
        "logistic_regression": lr_test_pred,
        "xgboost": xgb_test_pred,
    }.items():
        cm = confusion_matrix(y_test, pred, labels=[0, 1, 2])
        cm_df = pd.DataFrame(cm, index=["Sell", "Hold", "Buy"], columns=["Sell", "Hold", "Buy"])
        cm_df.to_csv(output_dir / f"confusion_matrix_{model_name}.csv")

    importances = pd.DataFrame(
        {
            "feature": all_features,
            "importance": xgb_model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importances.to_csv(output_dir / "xgboost_feature_importance.csv", index=False)

    backtest_summaries = {}
    for pred_col, name in [("lr_pred", "logistic_regression"), ("xgb_pred", "xgboost")]:
        curve_df = simple_backtest(test_predictions, pred_col=pred_col)
        curve_df.to_csv(output_dir / f"backtest_curve_{name}.csv", index=False)
        backtest_summaries[name] = compute_backtest_summary(curve_df)
    results["backtest"] = backtest_summaries

    summary_rows = [
        {
            "model": "Hold baseline",
            "accuracy": hold_metrics["accuracy"],
            "macro_f1": hold_metrics["macro_f1"],
            "weighted_f1": hold_metrics["weighted_f1"],
        },
        {
            "model": "Logistic Regression",
            "accuracy": lr_test_metrics["accuracy"],
            "macro_f1": lr_test_metrics["macro_f1"],
            "weighted_f1": lr_test_metrics["weighted_f1"],
        },
        {
            "model": "XGBoost",
            "accuracy": xgb_test_metrics["accuracy"],
            "macro_f1": xgb_test_metrics["macro_f1"],
            "weighted_f1": xgb_test_metrics["weighted_f1"],
        },
    ]
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "metrics_summary.csv", index=False)

    with open(output_dir / "experiment_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n=== Dataset Summary ===")
    print(f"Total rows: {results['dataset']['rows_total']}")
    print(f"Train/Valid/Test: {results['dataset']['rows_train']} / {results['dataset']['rows_valid']} / {results['dataset']['rows_test']}")
    print(f"Technical features: {len(technical_features)}")
    print(f"Sentiment features: {len(sentiment_features)}")
    print("\nTrain label distribution:")
    for label_name, frac in results["dataset"]["train_label_distribution"].items():
        print(f"  {label_name}: {frac:.4f}")

    print("\n=== Test Metrics ===")
    for row in summary_rows:
        print(
            f"{row['model']:<20} accuracy={row['accuracy']:.4f} "
            f"macro_f1={row['macro_f1']:.4f} weighted_f1={row['weighted_f1']:.4f}"
        )

    print("\n=== Best Validation Settings ===")
    print(f"Logistic Regression C: {lr_valid_metrics['C']}")
    print(
        "XGBoost params: "
        f"max_depth={xgb_valid_metrics['max_depth']}, "
        f"learning_rate={xgb_valid_metrics['learning_rate']}, "
        f"min_child_weight={xgb_valid_metrics['min_child_weight']}, "
        f"buy_threshold={xgb_valid_metrics['buy_threshold']}, "
        f"sell_threshold={xgb_valid_metrics['sell_threshold']}"
    )

    return results


if __name__ == "__main__":
    cfg = Config(
        tickers=[
            "AAPL", "MSFT", "AMZN", "GOOG", "META",
            "NVDA", "TSLA", "JPM", "XOM", "NFLX",
        ],
        start_date="2016-01-01",
        end_date="2025-12-31",
        label_threshold=0.003,
        output_dir="outputs",
        sentiment_csv_path=None,
    )
    run_experiment(cfg)