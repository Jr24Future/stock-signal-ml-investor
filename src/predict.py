from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd

from src.config import AppConfig
from src.features import build_live_frame


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


def label_name(label: int) -> str:
    mapping = {0: "Sell", 1: "Hold", 2: "Buy"}
    return mapping[int(label)]


def load_manifest(cfg: AppConfig) -> Dict:
    with open(cfg.feature_manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_live_prediction(
    cfg: AppConfig,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    cfg.ensure_directories()

    if not cfg.model_path.exists():
        raise FileNotFoundError(f"Model not found: {cfg.model_path}")
    if not cfg.feature_manifest_path.exists():
        raise FileNotFoundError(f"Feature manifest not found: {cfg.feature_manifest_path}")

    bundle = joblib.load(cfg.model_path)
    manifest = load_manifest(cfg)

    model = bundle["model"]
    feature_names: List[str] = bundle["feature_names"]
    buy_threshold = float(bundle["buy_threshold"])
    sell_threshold = float(bundle["sell_threshold"])

    df, _, _ = build_live_frame(
        cfg=cfg,
        start_date=start_date,
        end_date=end_date,
    )

    if df.empty:
        raise RuntimeError("No live feature rows were built.")

    latest_date = df["Date"].max()
    latest_df = df[df["Date"] == latest_date].copy()

    missing_features = [f for f in feature_names if f not in latest_df.columns]
    if missing_features:
        raise ValueError(f"Missing required features in live frame: {missing_features}")

    X_live = latest_df[feature_names].to_numpy()
    proba = model.predict_proba(X_live)
    pred = predict_with_thresholds(proba, buy_threshold, sell_threshold)

    out = latest_df[["Date", "Ticker", "Close"]].copy()
    out["pred_label"] = pred
    out["signal"] = out["pred_label"].apply(label_name)
    out["p_sell"] = proba[:, 0]
    out["p_hold"] = proba[:, 1]
    out["p_buy"] = proba[:, 2]
    out["model_path"] = str(cfg.model_path)
    out["generated_at_utc"] = datetime.utcnow().isoformat()

    out = out.sort_values(["Ticker"]).reset_index(drop=True)
    out.to_csv(cfg.prediction_output_path, index=False)

    print("\n=== Live Prediction Complete ===")
    print(f"Latest feature date: {latest_date.date()}")
    print(f"Saved predictions: {cfg.prediction_output_path}")
    print("\nPredictions:")
    print(out[["Ticker", "signal", "p_sell", "p_hold", "p_buy"]].to_string(index=False))

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run live predictions using the saved model.")
    parser.add_argument(
        "--start-date",
        default=(datetime.utcnow() - timedelta(days=120)).strftime("%Y-%m-%d"),
        help="Feature history start date",
    )
    parser.add_argument(
        "--end-date",
        default=datetime.utcnow().strftime("%Y-%m-%d"),
        help="Feature history end date",
    )
    parser.add_argument("--sentiment-csv", default=None, help="Optional sentiment CSV path")
    parser.add_argument("--model-filename", default="xgboost_model.pkl", help="Saved model filename")
    parser.add_argument(
        "--manifest-filename",
        default="feature_manifest.json",
        help="Saved feature manifest filename",
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

    run_live_prediction(
        cfg=cfg,
        start_date=args.start_date,
        end_date=args.end_date,
    )


if __name__ == "__main__":
    main()