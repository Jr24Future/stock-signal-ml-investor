from __future__ import annotations

import argparse
import json
from datetime import datetime
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from src.config import AppConfig
from src.features import build_training_frame


LABEL_TO_NAME = {0: "Sell", 1: "Hold", 2: "Buy"}


def chronological_split(
    df: pd.DataFrame,
    cfg: AppConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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


def summarize_labels(df: pd.DataFrame) -> Dict[str, float]:
    return (
        df["label"]
        .map(LABEL_TO_NAME)
        .value_counts(normalize=True)
        .reindex(["Sell", "Hold", "Buy"])
        .fillna(0.0)
        .to_dict()
    )


def predict_with_thresholds(
    proba: np.ndarray,
    buy_threshold: float,
    sell_threshold: float,
) -> np.ndarray:
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

    best_model = None
    best_metrics = None
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


def train_and_save_model(
    cfg: AppConfig,
    start_date: str,
    end_date: str,
) -> Dict[str, object]:
    cfg.ensure_directories()

    df, technical_features, sentiment_features = build_training_frame(
        cfg=cfg,
        start_date=start_date,
        end_date=end_date,
    )
    all_features = technical_features + sentiment_features

    train_df, valid_df, test_df = chronological_split(df, cfg)

    X_train = train_df[all_features].to_numpy()
    y_train = train_df["label"].to_numpy()
    X_valid = valid_df[all_features].to_numpy()
    y_valid = valid_df["label"].to_numpy()
    X_test = test_df[all_features].to_numpy()
    y_test = test_df["label"].to_numpy()

    model, valid_metrics, thresholds = tune_xgboost(
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        random_state=cfg.random_state,
    )

    test_proba = model.predict_proba(X_test)
    test_pred = predict_with_thresholds(test_proba, *thresholds)
    test_metrics = evaluate_predictions(y_test, test_pred)

    model_bundle = {
        "model": model,
        "feature_names": all_features,
        "technical_features": technical_features,
        "sentiment_features": sentiment_features,
        "buy_threshold": thresholds[0],
        "sell_threshold": thresholds[1],
        "label_mapping": LABEL_TO_NAME,
        "trained_at_utc": datetime.utcnow().isoformat(),
    }
    joblib.dump(model_bundle, cfg.model_path)

    manifest = {
        "model_type": "xgboost",
        "model_path": str(cfg.model_path),
        "feature_names": all_features,
        "technical_features": technical_features,
        "sentiment_features": sentiment_features,
        "buy_threshold": thresholds[0],
        "sell_threshold": thresholds[1],
        "train_date_range": {
            "start_date": start_date,
            "end_date": end_date,
        },
        "dataset_summary": {
            "rows_total": int(len(df)),
            "rows_train": int(len(train_df)),
            "rows_valid": int(len(valid_df)),
            "rows_test": int(len(test_df)),
            "train_label_distribution": summarize_labels(train_df),
            "valid_label_distribution": summarize_labels(valid_df),
            "test_label_distribution": summarize_labels(test_df),
        },
        "validation_metrics": valid_metrics,
        "test_metrics": test_metrics,
        "best_iteration": int(getattr(model, "best_iteration", -1)),
        "tickers": cfg.tickers,
        "trained_at_utc": datetime.utcnow().isoformat(),
    }

    with open(cfg.feature_manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    test_out = test_df[["Date", "Ticker", "Close", "next_return", "label"]].copy()
    test_out["pred"] = test_pred
    test_out["p_sell"] = test_proba[:, 0]
    test_out["p_hold"] = test_proba[:, 1]
    test_out["p_buy"] = test_proba[:, 2]
    test_out.to_csv(cfg.prediction_output_path, index=False)

    print("\n=== Training Complete ===")
    print(f"Saved model: {cfg.model_path}")
    print(f"Saved manifest: {cfg.feature_manifest_path}")
    print(f"Saved test predictions: {cfg.prediction_output_path}")

    print("\n=== Dataset Summary ===")
    print(f"Total rows: {len(df)}")
    print(f"Train/Valid/Test: {len(train_df)} / {len(valid_df)} / {len(test_df)}")
    print(f"Technical features: {len(technical_features)}")
    print(f"Sentiment features: {len(sentiment_features)}")

    print("\n=== Validation Metrics ===")
    print(valid_metrics)

    print("\n=== Test Metrics ===")
    print(test_metrics)

    print("\n=== Classification Report (Test) ===")
    print(
        classification_report(
            y_test,
            test_pred,
            labels=[0, 1, 2],
            target_names=["Sell", "Hold", "Buy"],
            zero_division=0,
        )
    )

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and save the live XGBoost model.")
    parser.add_argument("--start-date", required=True, help="Training start date, e.g. 2009-01-01")
    parser.add_argument("--end-date", required=True, help="Training end date, e.g. 2016-07-01")
    parser.add_argument("--sentiment-csv", default=None, help="Optional sentiment CSV path")
    parser.add_argument("--model-filename", default="xgboost_model.pkl", help="Output model filename")
    parser.add_argument(
        "--manifest-filename",
        default="feature_manifest.json",
        help="Output feature manifest filename",
    )
    parser.add_argument(
        "--predictions-filename",
        default="latest_predictions.csv",
        help="Output predictions filename",
    )
    args = parser.parse_args()

    cfg = AppConfig()
    cfg.sentiment_csv_path = args.sentiment_csv
    cfg.model_filename = args.model_filename
    cfg.feature_manifest_filename = args.manifest_filename
    cfg.prediction_output_filename = args.predictions_filename

    train_and_save_model(
        cfg=cfg,
        start_date=args.start_date,
        end_date=args.end_date,
    )


if __name__ == "__main__":
    main()