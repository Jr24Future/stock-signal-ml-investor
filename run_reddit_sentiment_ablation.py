# run_reddit_sentiment_ablation.py

import json
from pathlib import Path

import pandas as pd

# Make sure improved_stock_project.py is in the same folder
from improved_stock_project import Config, run_experiment

TICKERS = [
    "AAPL", "MSFT", "AMZN", "GOOG", "META",
    "NVDA", "TSLA", "JPM", "XOM", "NFLX",
]

START_DATE = "2009-01-01"
END_DATE = "2016-07-01"
LABEL_THRESHOLD = 0.003


def load_metrics(output_dir: str) -> pd.DataFrame:
    path = Path(output_dir) / "metrics_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Could not find metrics file: {path}")
    return pd.read_csv(path)


def load_results(output_dir: str) -> dict:
    path = Path(output_dir) / "experiment_results.json"
    if not path.exists():
        raise FileNotFoundError(f"Could not find results file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_tech_only():
    cfg = Config(
        tickers=TICKERS,
        start_date=START_DATE,
        end_date=END_DATE,
        label_threshold=LABEL_THRESHOLD,
        output_dir="outputs_reddit_tech_only",
        sentiment_csv_path=None,
    )
    print("\nRunning technical-only experiment...")
    return run_experiment(cfg)


def run_with_sentiment():
    sentiment_file = Path("reddit_daily_sentiment.csv")
    if not sentiment_file.exists():
        raise FileNotFoundError(
            "Could not find reddit_daily_sentiment.csv. "
            "Run build_reddit_sentiment_features.py first."
        )

    cfg = Config(
        tickers=TICKERS,
        start_date=START_DATE,
        end_date=END_DATE,
        label_threshold=LABEL_THRESHOLD,
        output_dir="outputs_reddit_with_sentiment",
        sentiment_csv_path=str(sentiment_file),
        sentiment_date_col="Date",
        sentiment_ticker_col=None,  # market-wide daily sentiment
        sentiment_numeric_cols=[
            "mean_sentiment",
            "std_sentiment",
            "min_sentiment",
            "max_sentiment",
            "mean_neg",
            "mean_neu",
            "mean_pos",
            "headline_count",
            "positive_count",
            "negative_count",
            "neutral_count",
        ],
    )
    print("\nRunning technical + sentiment experiment...")
    return run_experiment(cfg)


def compare_and_save():
    tech_metrics = load_metrics("outputs_reddit_tech_only")
    sent_metrics = load_metrics("outputs_reddit_with_sentiment")

    tech_metrics = tech_metrics.copy()
    tech_metrics["experiment"] = "technical_only"

    sent_metrics = sent_metrics.copy()
    sent_metrics["experiment"] = "technical_plus_sentiment"

    combined = pd.concat([tech_metrics, sent_metrics], ignore_index=True)
    combined = combined[["experiment", "model", "accuracy", "macro_f1", "weighted_f1"]]

    out_path = Path("outputs_reddit_ablation_comparison.csv")
    combined.to_csv(out_path, index=False)

    print("\n=== Ablation Comparison ===")
    print(combined.to_string(index=False))
    print(f"\nSaved comparison to: {out_path.resolve()}")

    tech_results = load_results("outputs_reddit_tech_only")
    sent_results = load_results("outputs_reddit_with_sentiment")

    print("\n=== Dataset Summary Comparison ===")
    print(
        f"Technical-only rows: {tech_results['dataset']['rows_total']} | "
        f"features: {tech_results['dataset']['num_features']} | "
        f"sentiment features: {len(tech_results['dataset']['sentiment_features'])}"
    )
    print(
        f"With sentiment rows: {sent_results['dataset']['rows_total']} | "
        f"features: {sent_results['dataset']['num_features']} | "
        f"sentiment features: {len(sent_results['dataset']['sentiment_features'])}"
    )


def main():
    run_tech_only()
    run_with_sentiment()
    compare_and_save()


if __name__ == "__main__":
    main()