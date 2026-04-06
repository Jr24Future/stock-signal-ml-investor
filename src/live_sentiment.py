from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from pandas.tseries.offsets import BDay
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src.config import PROCESSED_DIR, RAW_DIR


DEFAULT_INPUT = RAW_DIR / "live_headlines.csv"
DEFAULT_OUTPUT = PROCESSED_DIR / "live_daily_sentiment.csv"


def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text).strip()
    text = text.replace("\\n", " ").replace("\n", " ").replace("\r", " ")
    return " ".join(text.split())


def label_from_compound(score: float) -> str:
    if score >= 0.05:
        return "positive"
    if score <= -0.05:
        return "negative"
    return "neutral"


def map_to_effective_trading_date(ts: pd.Timestamp) -> pd.Timestamp:
    d = ts.normalize().tz_localize(None)

    # Saturday -> Monday
    if d.weekday() == 5:
        return d + BDay(1)

    # Sunday -> Monday
    if d.weekday() == 6:
        return d + BDay(1)

    return d


def build_live_daily_sentiment(input_csv: str, output_csv: str) -> pd.DataFrame:
    df = pd.read_csv(input_csv)

    required_cols = {"title", "published_at_utc"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in live headlines file: {sorted(missing)}")

    df = df.copy()
    df["title"] = df["title"].apply(clean_text)

    if "summary" in df.columns:
        df["summary"] = df["summary"].apply(clean_text)
    else:
        df["summary"] = ""

    df["text"] = (df["title"] + ". " + df["summary"]).str.strip()
    df["published_at_utc"] = pd.to_datetime(df["published_at_utc"], errors="coerce", utc=True)
    df = df.dropna(subset=["published_at_utc"]).copy()
    df = df[df["text"].str.len() > 0].copy()

    # Map raw publish timestamp to an effective trading date
    df["Date"] = df["published_at_utc"].apply(map_to_effective_trading_date)

    analyzer = SentimentIntensityAnalyzer()
    scores = df["text"].apply(lambda x: analyzer.polarity_scores(x))

    df["neg"] = scores.apply(lambda s: s["neg"])
    df["neu"] = scores.apply(lambda s: s["neu"])
    df["pos"] = scores.apply(lambda s: s["pos"])
    df["compound"] = scores.apply(lambda s: s["compound"])
    df["sentiment_label"] = df["compound"].apply(label_from_compound)

    daily = (
        df.groupby("Date", as_index=False)
        .agg(
            mean_sentiment=("compound", "mean"),
            std_sentiment=("compound", "std"),
            min_sentiment=("compound", "min"),
            max_sentiment=("compound", "max"),
            mean_neg=("neg", "mean"),
            mean_neu=("neu", "mean"),
            mean_pos=("pos", "mean"),
            headline_count=("text", "count"),
        )
    )

    counts = (
        df.pivot_table(
            index="Date",
            columns="sentiment_label",
            values="text",
            aggfunc="count",
            fill_value=0,
        )
        .rename_axis(None, axis=1)
        .reset_index()
    )

    for col in ["positive", "negative", "neutral"]:
        if col not in counts.columns:
            counts[col] = 0

    counts = counts.rename(
        columns={
            "positive": "positive_count",
            "negative": "negative_count",
            "neutral": "neutral_count",
        }
    )

    out = daily.merge(
        counts[["Date", "positive_count", "negative_count", "neutral_count"]],
        on="Date",
        how="left",
    )

    out["std_sentiment"] = out["std_sentiment"].fillna(0.0)
    for col in ["positive_count", "negative_count", "neutral_count", "headline_count"]:
        out[col] = out[col].fillna(0).astype(int)

    out = out.sort_values("Date").reset_index(drop=True)

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)

    print("\n=== Live Daily Sentiment Built ===")
    print(f"Input headlines: {len(df)}")
    print(f"Daily rows: {len(out)}")
    if len(out) > 0:
        print(f"Date range: {out['Date'].min()} to {out['Date'].max()}")
    print(f"Saved to: {output_path.resolve()}")

    print("\nColumns:")
    for c in out.columns:
        print(f"  - {c}")

    print("\nEffective trading dates:")
    print(out[["Date", "headline_count"]].to_string(index=False))

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build live daily sentiment features from ingested headlines.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Path to live_headlines.csv")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Path to output live_daily_sentiment.csv")
    args = parser.parse_args()

    build_live_daily_sentiment(
        input_csv=args.input,
        output_csv=args.output,
    )


if __name__ == "__main__":
    main()