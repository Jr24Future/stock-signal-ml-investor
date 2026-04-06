from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.config import ARTIFACTS_DIR, PROCESSED_DIR, RAW_DIR
from src.live_sentiment import build_live_daily_sentiment
from src.news_ingest import run_ingestion


STATUS_PATH = ARTIFACTS_DIR / "live_news_status.json"
HEADLINES_PATH = RAW_DIR / "live_headlines.csv"
SENTIMENT_PATH = PROCESSED_DIR / "live_daily_sentiment.csv"
SOURCE_CONFIG_PATH = RAW_DIR / "news_sources.json"


def write_status(payload: dict) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(STATUS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    started_at = datetime.now(timezone.utc).isoformat()

    try:
        headlines_df = run_ingestion(
            source_config=SOURCE_CONFIG_PATH,
            output_path=HEADLINES_PATH,
        )

        sentiment_df = build_live_daily_sentiment(
            input_csv=str(HEADLINES_PATH),
            output_csv=str(SENTIMENT_PATH),
        )

        status = {
            "ok": True,
            "started_at_utc": started_at,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "headlines_path": str(HEADLINES_PATH),
            "sentiment_path": str(SENTIMENT_PATH),
            "headline_rows_total": int(len(headlines_df)),
            "sentiment_daily_rows": int(len(sentiment_df)),
            "latest_sentiment_date": (
                str(sentiment_df["Date"].max()) if len(sentiment_df) > 0 else None
            ),
        }

        write_status(status)

        print("\n=== Live News Update Complete ===")
        print(json.dumps(status, indent=2))

    except Exception as exc:
        status = {
            "ok": False,
            "started_at_utc": started_at,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "error": str(exc),
        }
        write_status(status)
        raise


if __name__ == "__main__":
    main()