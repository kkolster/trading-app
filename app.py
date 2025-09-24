# app.py — universal fix for yfinance data

import warnings, datetime as dt, numpy as np, pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

# ---------------- UI ----------------
st.set_page_config(page_title="Simple Quant Trader", layout="wide")
st.title("🧠 Simple Quant Trader — demo app")

with st.sidebar:
    st.header("⚙️ Settings")
    symbol = st.text_input("Ticker (any valid Yahoo Finance symbol)", value="AAPL").strip()
    end = st.date_input("End date", dt.date.today())
    start = st.date_input("Start date", end - dt.timedelta(days=365*2))
    interval = st.selectbox("Interval", ["1d", "1h", "30m", "15m"], index=0)

# ---------------- Data Loader ----------------
@st.cache_data(show_spinner=True, ttl=3600)
def load_data(symbol, start, end, interval):
    df = yf.download(
        symbol,
        start=start,
        end=end + dt.timedelta(days=1),  # yfinance end is exclusive
        interval=interval,
        auto_adjust=False,
        progress=False,
    )

    if df is None or df.empty:
        return pd.DataFrame()

    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Standardize names
    df = df.rename(columns={
        "Adj Close": "AdjClose",
        "Adj_Close": "AdjClose",
    })

    # Ensure all required cols exist
    if "AdjClose" not in df.columns and "Close" in df.columns:
        df["AdjClose"] = df["Close"]

    return df

# ---------------- Chart ----------------
def make_chart(df, symbol):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        name="Price"
    ))
    fig.update_layout(
        title=f"{symbol} Candlestick",
        xaxis_rangeslider_visible=False,
        height=600
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------- Main ----------------
df = load_data(symbol, start, end, interval)

if df.empty:
    st.error("No data loaded. Check symbol or date range.")
    st.stop()

st.success(f"Loaded {len(df)} rows for {symbol}")
st.write("Columns:", list(df.columns))

# Require OHLCV
req = ["Open", "High", "Low", "Close", "AdjClose", "Volume"]
missing = [c for c in req if c not in df.columns]
if missing:
    st.error(f"Missing columns: {missing}")
    st.stop()

# Example math indicators
df["EMA50"] = df["AdjClose"].ewm(span=50).mean()
df["EMA200"] = df["AdjClose"].ewm(span=200).mean()
delta = df["AdjClose"].diff()
gain = np.where(delta > 0, delta, 0)
loss = np.where(delta < 0, -delta, 0)
avg_gain = pd.Series(gain).rolling(14).mean()
avg_loss = pd.Series(loss).rolling(14).mean()
rs = avg_gain / avg_loss
df["RSI14"] = 100 - (100 / (1 + rs))

make_chart(df, symbol)

st.subheader("Preview of last rows")
st.dataframe(df.tail())
