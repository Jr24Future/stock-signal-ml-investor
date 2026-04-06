from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any

import feedparser
import pandas as pd
from dateutil import parser as dtparser

from src.config import RAW_DIR


DEFAULT_SOURCE_CONFIG = RAW_DIR / "news_sources.json"
DEFAULT_OUTPUT = RAW_DIR / "live_headlines.csv"


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text).replace("\n", " ").replace("\r", " ").strip()
    return " ".join(text.split())


def safe_parse_datetime(value: str | None) -> str:
    if not value:
        return datetime.now(timezone.utc).isoformat()
    try:
        dt = dtparser.parse(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    except Exception:
        return datetime.now(timezone.utc).isoformat()


def stable_id(source: str, title: str, published_at: str) -> str:
    raw = f"{source}|{title}|{published_at}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_feed_urls(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Feed config not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    feeds = payload.get("feeds", [])
    if not feeds:
        raise ValueError("No feed URLs found in news_sources.json")

    return [str(url).strip() for url in feeds if str(url).strip()]


def fetch_feed(url: str) -> List[Dict[str, Any]]:
    parsed = feedparser.parse(url)
    rows: List[Dict[str, Any]] = []

    for entry in parsed.entries:
        title = normalize_text(getattr(entry, "title", ""))
        summary = normalize_text(getattr(entry, "summary", ""))
        link = normalize_text(getattr(entry, "link", ""))
        published_raw = (
            getattr(entry, "published", None)
            or getattr(entry, "updated", None)
            or getattr(entry, "created", None)
        )
        published_at = safe_parse_datetime(published_raw)
        item_id = stable_id(url, title, published_at)

        if not title:
            continue

        rows.append(
            {
                "id": item_id,
                "source_feed": url,
                "title": title,
                "summary": summary,
                "link": link,
                "published_at_utc": published_at,
                "ingested_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )

    return rows


def merge_and_deduplicate(existing_path: Path, new_rows: List[Dict[str, Any]]) -> pd.DataFrame:
    new_df = pd.DataFrame(new_rows)

    if existing_path.exists():
        old_df = pd.read_csv(existing_path)
        combined = pd.concat([old_df, new_df], ignore_index=True)
    else:
        combined = new_df

    if combined.empty:
        return combined

    combined = combined.drop_duplicates(subset=["id"]).copy()
    combined["published_at_utc"] = pd.to_datetime(combined["published_at_utc"], errors="coerce", utc=True)
    combined["ingested_at_utc"] = pd.to_datetime(combined["ingested_at_utc"], errors="coerce", utc=True)
    combined = combined.sort_values(["published_at_utc", "title"]).reset_index(drop=True)
    return combined


def run_ingestion(source_config: Path, output_path: Path) -> pd.DataFrame:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    feed_urls = load_feed_urls(source_config)

    all_rows: List[Dict[str, Any]] = []
    for url in feed_urls:
        try:
            rows = fetch_feed(url)
            all_rows.extend(rows)
            print(f"Fetched {len(rows)} headlines from {url}")
        except Exception as exc:
            print(f"Failed to fetch {url}: {exc}")

    combined = merge_and_deduplicate(output_path, all_rows)
    combined.to_csv(output_path, index=False)

    print("\n=== News Ingestion Complete ===")
    print(f"Feeds checked: {len(feed_urls)}")
    print(f"Total stored headlines: {len(combined)}")
    print(f"Saved to: {output_path.resolve()}")

    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch live headlines from RSS feeds.")
    parser.add_argument("--source-config", default=str(DEFAULT_SOURCE_CONFIG), help="Path to news_sources.json")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Path to output CSV")
    args = parser.parse_args()

    run_ingestion(
        source_config=Path(args.source_config),
        output_path=Path(args.output),
    )


if __name__ == "__main__":
    main()