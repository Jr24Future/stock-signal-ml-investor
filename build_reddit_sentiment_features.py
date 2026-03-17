import argparse
from pathlib import Path

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text).strip()
    text = text.replace("\\n", " ").replace("\n", " ")
    return " ".join(text.split())


def label_from_compound(score: float) -> str:
    if score >= 0.05:
        return "positive"
    if score <= -0.05:
        return "negative"
    return "neutral"


def build_daily_sentiment(input_csv: str, output_csv: str, date_col: str = "Date", text_col: str = "News") -> pd.DataFrame:
    df = pd.read_csv(input_csv)

    if date_col not in df.columns:
        raise ValueError(f"Could not find date column '{date_col}'. Columns found: {list(df.columns)}")
    if text_col not in df.columns:
        raise ValueError(f"Could not find text column '{text_col}'. Columns found: {list(df.columns)}")

    df = df[[date_col, text_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[text_col] = df[text_col].apply(clean_text)
    df = df.dropna(subset=[date_col]).copy()
    df = df[df[text_col].str.len() > 0].copy()

    analyzer = SentimentIntensityAnalyzer()

    scores = df[text_col].apply(lambda x: analyzer.polarity_scores(x))
    df["neg"] = scores.apply(lambda s: s["neg"])
    df["neu"] = scores.apply(lambda s: s["neu"])
    df["pos"] = scores.apply(lambda s: s["pos"])
    df["compound"] = scores.apply(lambda s: s["compound"])
    df["sentiment_label"] = df["compound"].apply(label_from_compound)

    daily = (
        df.groupby(date_col, as_index=False)
        .agg(
            mean_sentiment=("compound", "mean"),
            std_sentiment=("compound", "std"),
            min_sentiment=("compound", "min"),
            max_sentiment=("compound", "max"),
            mean_neg=("neg", "mean"),
            mean_neu=("neu", "mean"),
            mean_pos=("pos", "mean"),
            headline_count=(text_col, "count"),
        )
        .rename(columns={date_col: "Date"})
    )

    counts = (
        df.pivot_table(
            index=date_col,
            columns="sentiment_label",
            values=text_col,
            aggfunc="count",
            fill_value=0,
        )
        .rename_axis(None, axis=1)
        .reset_index()
        .rename(columns={date_col: "Date"})
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

    print("\n=== Reddit Sentiment Features Built ===")
    print(f"Input rows: {len(df)}")
    print(f"Daily rows: {len(out)}")
    print(f"Date range: {out['Date'].min().date()} to {out['Date'].max().date()}")
    print(f"Saved to: {output_path.resolve()}")
    print("\nColumns:")
    for c in out.columns:
        print(f"  - {c}")

    return out


def main():
    parser = argparse.ArgumentParser(description="Build daily sentiment features from RedditNews.csv")
    parser.add_argument("--input", default="RedditNews.csv", help="Path to RedditNews.csv")
    parser.add_argument("--output", default="reddit_daily_sentiment.csv", help="Output aggregated sentiment CSV")
    parser.add_argument("--date-col", default="Date", help="Date column name")
    parser.add_argument("--text-col", default="News", help="Text/headline column name")
    args = parser.parse_args()

    build_daily_sentiment(
        input_csv=args.input,
        output_csv=args.output,
        date_col=args.date_col,
        text_col=args.text_col,
    )


if __name__ == "__main__":
    main()