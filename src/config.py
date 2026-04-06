from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
LOGS_DIR = BASE_DIR / "logs"


@dataclass
class AppConfig:
    tickers: List[str] = field(
        default_factory=lambda: [
            "AAPL", "MSFT", "AMZN", "GOOG", "META",
            "NVDA", "TSLA", "JPM", "XOM", "NFLX",
        ]
    )

    label_threshold: float = 0.003
    train_frac: float = 0.70
    valid_frac: float = 0.15
    random_state: int = 42

    use_auto_adjusted_prices: bool = True
    price_batch_size: int = 5
    max_download_retries: int = 3
    retry_sleep_seconds: float = 2.0

    model_filename: str = "xgboost_model.pkl"
    feature_manifest_filename: str = "feature_manifest.json"
    prediction_output_filename: str = "latest_predictions.csv"

    sentiment_csv_path: Optional[str] = None
    sentiment_date_col: Optional[str] = "Date"
    sentiment_ticker_col: Optional[str] = None
    sentiment_numeric_cols: Optional[List[str]] = field(
        default_factory=lambda: [
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
        ]
    )

    def ensure_directories(self) -> None:
        for path in [DATA_DIR, RAW_DIR, PROCESSED_DIR, MODELS_DIR, ARTIFACTS_DIR, LOGS_DIR]:
            path.mkdir(parents=True, exist_ok=True)

    @property
    def model_path(self) -> Path:
        return MODELS_DIR / self.model_filename

    @property
    def feature_manifest_path(self) -> Path:
        return ARTIFACTS_DIR / self.feature_manifest_filename

    @property
    def prediction_output_path(self) -> Path:
        return ARTIFACTS_DIR / self.prediction_output_filename