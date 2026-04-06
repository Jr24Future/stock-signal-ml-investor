from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import yfinance as yf

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

TECH_PRED_PATH = ARTIFACTS_DIR / "predictions_technical_live.csv"
SENT_PRED_PATH = ARTIFACTS_DIR / "predictions_sentiment_live.csv"
NEWS_STATUS_PATH = ARTIFACTS_DIR / "live_news_status.json"


def get_prediction_path(mode: str) -> Path:
    if mode == "Technical + Sentiment":
        return SENT_PRED_PATH
    return TECH_PRED_PATH


def load_predictions(pred_path: Path) -> pd.DataFrame:
    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")

    df = pd.read_csv(pred_path).copy()

    if "Date" in df.columns:
        df["Date"] = df["Date"].astype(str)

    for col in ["p_sell", "p_hold", "p_buy"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(4)

    return df


def load_news_status() -> dict | None:
    if not NEWS_STATUS_PATH.exists():
        return None
    try:
        with open(NEWS_STATUS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def run_command(cmd: list[str]) -> tuple[bool, str]:
    result = subprocess.run(
        cmd,
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True,
    )

    output = ""
    if result.stdout:
        output += result.stdout
    if result.stderr:
        output += ("\n" + result.stderr) if output else result.stderr

    return result.returncode == 0, output.strip()


def run_prediction_job(mode: str) -> tuple[bool, str]:
    logs: list[str] = []

    if mode == "Technical + Sentiment":
        steps = [
            [sys.executable, "-m", "src.update_live_news"],
        ]

        for step in steps:
            ok, output = run_command(step)
            logs.append(f"$ {' '.join(step)}\n{output}")
            if not ok:
                return False, "\n\n".join(logs)

        status = load_news_status()
        sentiment_days = int(status.get("sentiment_daily_rows", 0)) if status else 0

        if sentiment_days < 2:
            logs.append(
                "\nSentiment live mode is warming up: fewer than 2 effective sentiment trading days are available yet, "
                "so prediction is skipped for now."
            )
            return True, "\n\n".join(logs)

        step = [
            sys.executable, "-m", "src.predict",
            "--sentiment-csv", r"data\processed\live_daily_sentiment.csv",
            "--model-filename", "xgboost_sentiment.pkl",
            "--manifest-filename", "manifest_sentiment.json",
            "--predictions-filename", "predictions_sentiment_live.csv",
        ]
        ok, output = run_command(step)
        logs.append(f"$ {' '.join(step)}\n{output}")
        if not ok:
            return False, "\n\n".join(logs)

        return True, "\n\n".join(logs)

    else:
        step = [
            sys.executable, "-m", "src.predict",
            "--model-filename", "xgboost_technical.pkl",
            "--manifest-filename", "manifest_technical.json",
            "--predictions-filename", "predictions_technical_live.csv",
        ]
        ok, output = run_command(step)
        logs.append(f"$ {' '.join(step)}\n{output}")
        return ok, "\n\n".join(logs)


@st.cache_data(ttl=1800)
def load_ticker_history(ticker: str, period: str = "6mo") -> pd.DataFrame:
    df = yf.download(
        tickers=ticker,
        period=period,
        auto_adjust=True,
        progress=False,
        threads=False,
    )

    if df.empty:
        return pd.DataFrame()

    df = df.reset_index().copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


st.set_page_config(page_title="Stack Investor", layout="wide")
st.title("Stack Investor")
st.caption("Live Buy / Hold / Sell predictions from the saved ML model")

model_mode = st.radio(
    "Model Mode",
    ["Technical Only", "Technical + Sentiment"],
    horizontal=True,
)

if model_mode == "Technical + Sentiment":
    st.warning(
        "Sentiment live mode needs enough recent headline history aligned with trading dates. "
        "If no valid merged rows exist yet, refresh may still fail."
    )

status = load_news_status()
st.subheader("Live News Pipeline Status")

if status is None:
    st.info("No live news status file found yet.")
else:
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Updater OK", "Yes" if status.get("ok") else "No")
    s2.metric("Stored Headlines", int(status.get("headline_rows_total", 0)))
    s3.metric("Sentiment Days", int(status.get("sentiment_daily_rows", 0)))
    s4.metric("Latest Sentiment Date", str(status.get("latest_sentiment_date", "N/A")))

    st.caption(
        f"Last news update finished at UTC: {status.get('finished_at_utc', 'N/A')}"
    )

col_a, col_b = st.columns([1, 3])

with col_a:
    if st.button("Refresh Predictions", type="primary"):
        with st.spinner(f"Running {model_mode} pipeline..."):
            ok, output = run_prediction_job(model_mode)

        st.session_state["last_refresh_output"] = output

        if ok:
            st.success("Pipeline finished successfully.")
            load_ticker_history.clear()
        else:
            st.warning("Pipeline did not complete a prediction run yet. See refresh log below.")

if "last_refresh_output" in st.session_state and st.session_state["last_refresh_output"]:
    with st.expander("Refresh log"):
        st.code(st.session_state["last_refresh_output"])

pred_path = get_prediction_path(model_mode)

if not pred_path.exists():
    if model_mode == "Technical + Sentiment":
        st.info(
            "No sentiment-live prediction file exists yet. "
            "Try Refresh Predictions again after more live headline history has been collected."
        )
    else:
        st.info(
            "No technical-live prediction file exists yet. "
            "Click Refresh Predictions to generate one."
        )
    st.stop()

try:
    df = load_predictions(pred_path)
except Exception as exc:
    st.error(str(exc))
    st.stop()

if "generated_at_utc" in df.columns and not df["generated_at_utc"].isna().all():
    last_updated = str(df["generated_at_utc"].iloc[0])
    st.info(f"Last prediction run (UTC): {last_updated}")

display_cols = ["Date", "Ticker", "signal", "p_sell", "p_hold", "p_buy"]
display_df = df[display_cols].copy()

buy_count = int((display_df["signal"] == "Buy").sum())
hold_count = int((display_df["signal"] == "Hold").sum())
sell_count = int((display_df["signal"] == "Sell").sum())

m1, m2, m3 = st.columns(3)
m1.metric("Buy Signals", buy_count)
m2.metric("Hold Signals", hold_count)
m3.metric("Sell Signals", sell_count)

st.subheader("Latest Predictions")
st.dataframe(
    display_df,
    width="stretch",
    hide_index=True,
)

st.subheader("Signal Counts")
signal_counts = display_df["signal"].value_counts().reset_index()
signal_counts.columns = ["Signal", "Count"]
st.dataframe(
    signal_counts,
    width="stretch",
    hide_index=True,
)

selected_ticker = st.selectbox("Select a ticker", sorted(display_df["Ticker"].unique()))
ticker_df = df[df["Ticker"] == selected_ticker].copy()

st.subheader(f"Details for {selected_ticker}")

top_left, top_mid, top_right = st.columns(3)
top_left.metric("Current Signal", str(ticker_df["signal"].iloc[0]))
top_mid.metric("Sell Probability", f"{float(ticker_df['p_sell'].iloc[0]):.4f}")
top_right.metric("Buy Probability", f"{float(ticker_df['p_buy'].iloc[0]):.4f}")

detail_df = ticker_df[["Date", "Ticker", "signal", "p_sell", "p_hold", "p_buy"]].copy()
st.dataframe(
    detail_df,
    width="stretch",
    hide_index=True,
)

st.subheader(f"{selected_ticker} Price Chart")
price_df = load_ticker_history(selected_ticker, period="6mo")

if price_df.empty:
    st.warning("Could not load recent price history for this ticker.")
else:
    if "Date" in price_df.columns and "Close" in price_df.columns:
        chart_df = price_df[["Date", "Close"]].copy()
        chart_df["Date"] = pd.to_datetime(chart_df["Date"], errors="coerce")
        chart_df = chart_df.dropna(subset=["Date", "Close"]).set_index("Date")
        st.line_chart(chart_df, height=350, width="stretch")

        latest_close = float(chart_df["Close"].iloc[-1])
        six_month_change = (
            (float(chart_df["Close"].iloc[-1]) / float(chart_df["Close"].iloc[0]) - 1.0) * 100.0
            if len(chart_df) > 1 else 0.0
        )

        c1, c2 = st.columns(2)
        c1.metric("Latest Close", f"{latest_close:.2f}")
        c2.metric("6-Month Change", f"{six_month_change:.2f}%")
    else:
        st.warning("Price history did not contain the expected Date/Close columns.")