# app.py
import warnings, datetime as dt
import numpy as np
import pandas as pd
import yfinance as yf

from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, SMAIndicator, MACD
from ta.volatility import BollingerBands

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import TimeSeriesSplit

import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# -------------------------
# Robust Data Loader
# -------------------------
def load(symbol, start, end, interval):
    # Normalize start/end into datetime
    if isinstance(start, dt.date) and not isinstance(start, dt.datetime):
        start = dt.datetime.combine(start, dt.time.min)
    if isinstance(end, dt.date) and not isinstance(end, dt.datetime):
        end = dt.datetime.combine(end, dt.time.min)

    # Don’t query future
    now = dt.datetime.utcnow()
    if end > now:
        end = now

    # Common index aliases
    sym = symbol.strip().upper()
    alias = {
        "SPX": "^GSPC", "GSPC": "^GSPC", "SP500": "^GSPC",
        "NDX": "^NDX", "US100": "^NDX", "NASDAQ": "^IXIC",
        "DOW": "^DJI", "DJI": "^DJI"
    }
    sym = alias.get(sym, sym)

    # Download
    df = yf.download(
        sym, start=start, end=end, interval=interval,
        auto_adjust=True, progress=False, threads=True
    )

    if df is None or df.empty:
        raise ValueError(
            f"No data returned for '{sym}' with interval='{interval}'. "
            "Try another ticker or earlier start date."
        )

    # If MultiIndex columns → flatten
    if isinstance(df.columns, pd.MultiIndex):
        df = df.droplevel(0, axis=1)

    # Normalize column names
    df = df.rename(columns=lambda c: str(c).strip().title())
    df.index = pd.to_datetime(df.index)

    return df

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Simple Quant Trader", layout="wide")
st.title("🧠 Simple Quant Trader — demo app")

with st.sidebar:
    st.header("⚙️ Settings")
    symbol = st.text_input(
        "Ticker (examples: AAPL, NVDA, ^GSPC, ^NDX, BTC-USD, EURUSD=X)", 
        "AAPL"
    )
    start = st.date_input("Start date", dt.date.today() - dt.timedelta(days=365*2))
    end = st.date_input("End date", dt.date.today())
    interval = st.selectbox("Interval", ["1d", "1h", "15m"])
    horizon = st.number_input("Prediction horizon (bars ahead)", min_value=1, max_value=10, value=1)
    prob_long = st.slider("Prob. threshold LONG", 0.0, 1.0, 0.55, 0.01)
    prob_short = st.slider("Prob. threshold SHORT", 0.0, 1.0, 0.55, 0.01)

# -------------------------
# Load Data
# -------------------------
try:
    df = load(symbol, start, end + dt.timedelta(days=1), interval)
except Exception as e:
    st.error(f"❌ Could not load data: {e}")
    st.stop()

st.success(f"Loaded {len(df)} rows for {symbol}")

# -------------------------
# Plot Price
# -------------------------
fig = go.Figure(data=[go.Candlestick(
    x=df.index,
    open=df["Open"], high=df["High"],
    low=df["Low"], close=df["Close"],
    name="Price"
)])
st.plotly_chart(fig, use_container_width=True)
